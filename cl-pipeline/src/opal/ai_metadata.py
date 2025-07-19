# Motivated by https://medium.com/@onkarmishra/using-langchain-for-question-answering-on-own-data-3af0a82789ed
# Thanks to Onkar Mishra for the inspiration

# https://www.dnb.de/DE/Professionell/DDC-Deutsch/DDCUebersichten/ddcUebersichten_node.html

import pandas as pd
from pathlib import Path
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
from tqdm import tqdm
import logging
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
import chromadb
import re
from collections import Counter
from wrapt_timeout_decorator import *
import json

from pipeline.taskfactory import TaskWithInputFileMonitor

import sys
sys.path.append('../src/general/')
from checkAuthorNames import NameChecker

def safe_is_empty(value):
    """Safely check if a value is empty, handling arrays/lists"""
    if value is None or value == "":
        return True
    try:
        # For scalar values, use pd.isna()
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        # If pd.isna fails (e.g., for arrays), check if it's an empty list
        if isinstance(value, list) and len(value) == 0:
            return True
    return False

def filtered(AI_response):
    # Check for deepseeg tags and remove explanations
    if "<think>" in AI_response:
        AI_response = re.sub(r'<think>.*?</think>', '', AI_response, flags=re.DOTALL)
    
    # For JSON responses, be more lenient with filtering
    if AI_response.strip().startswith('[') or AI_response.strip().startswith('{'):
        # This looks like JSON, only remove obvious non-JSON parts
        AI_response = AI_response.replace("\n", "")
        return AI_response.strip()
    
    # Remove newlines for non-JSON responses
    AI_response = AI_response.replace("\n", "")
    
    # Check for blacklist words and remove them (but not for JSON-like responses)
    blacklist = ["don't know", "weiß nicht", "weiß es nicht", "Ich kenne ",
                 "Ich sehe kein", "Es gibt kein", "Ich kann ", "Ich sehe", "Es wird keine ",
                 "Entschuldigung", "Leider kann ich", "Keine Antwort", "Die Antwort kann ich",
                 "Der Autor", "die Frage", "Ich habe keine", "Ich habe ", "Ich brauche",
                 "Bitte geben", "Das Dokument ", "Es tut mir leid", "Es handelt sich", "Es ist nicht", "Es konnte kein",
                 "I'm ready to help", "Please provide", "I'll respond with"]
    
    if any(x in AI_response for x in blacklist):
        return ""
    else:
        return AI_response.strip()

def clean_and_parse_json_response(raw_response: str):
    # Entferne evtl. Markdown-Codeblock-Markierungen wie ```json ... ```
    cleaned = re.sub(r"^```json\s*|\s*```$", "", raw_response.strip(), flags=re.DOTALL)
    
    # Remove any leading/trailing explanatory text
    cleaned = cleaned.strip()
    
    # Try to find JSON array or object in the response
    json_match = re.search(r'(\[.*?\]|\{.*?\})', cleaned, re.DOTALL)
    if json_match:
        cleaned = json_match.group(1)
    
    try:
        parsed = json.loads(cleaned)
        return parsed
    except json.JSONDecodeError as e:
        print(f"Fehler beim Parsen von '{cleaned[:100]}...': {e}")
        
        # Try to fix common JSON issues
        try:
            # Replace single quotes with double quotes
            fixed = cleaned.replace("'", '"')
            parsed = json.loads(fixed)
            return parsed
        except json.JSONDecodeError:
            # If still failing, try to extract key information manually
            print("Versuche manuelle Extraktion...")
            return None

def has_valid_dewey_classification(data, valid_notations=None) -> bool:
    try:
        # Wenn data ein String ist, parse ihn als JSON
        if isinstance(data, str):
            data = json.loads(data)
            
        if not isinstance(data, list) or len(data) == 0:
            return False

        # Optional: Gültige Dewey-Notation prüfen (3 Ziffern, optional mit Unterklassen wie 004.67)
        dewey_pattern = re.compile(r'^\d{3}(\.\d+)?$')

        for entry in data:
            print(entry)
            notation = entry.get("notation", "")
            if isinstance(notation, str) and dewey_pattern.match(notation):
                # Wenn valid_notations bereitgestellt werden, prüfe auch gegen diese
                if valid_notations is None or notation in valid_notations:
                    return True  # mindestens ein gültiger Eintrag
        return False
    except (json.JSONDecodeError, TypeError):
        return False

# Creating the chain for Question Answering
def load_qa_chain(retriever, llm, prompt):
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever, 
        chain_type="stuff",
        return_source_documents=True, 
        chain_type_kwargs={'prompt': prompt} 
    )

@timeout(240)
def get_response(query, chain):
    # Getting response from chain
    try:
        response = chain.invoke({'query': query})
        return response['result']
    except Exception as e:
        logging.error("Error processing %s: %s", query, str(e))
        return ""
    
def get_monitored_response(query, chain):
    try:
        return get_response(query, chain)
    except Exception as e:
        logging.error("Timeout of 60s %s: %s", query, str(e))
        return ""

# Lade die Prompts aus der JSON-Datei
def load_prompts(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        prompts = json.load(f)
    return prompts

class AIMetaDataExtraction(TaskWithInputFileMonitor):
    def __init__(self, config_stage, config_global):
        super().__init__(config_stage, config_global)
        stage_param = config_stage['parameters']
        self.file_file_name_inputs =  Path(config_global['raw_data_folder']) / stage_param['file_file_name_input']
        self.file_file_name_output =  Path(config_global['raw_data_folder']) / stage_param['file_file_name_output']
        self.file_folder = Path(config_global['file_folder'])
        self.file_types = stage_param['file_types']
        self.processed_data_folder = config_global['processed_data_folder']
        self.chroma_file = Path(config_global['processed_data_folder']) / "chroma_db"
        self.prompts_file = stage_param['prompts_file_name']
        self.prompts = load_prompts(self.prompts_file)
        self.dewey_classification_file = stage_param.get('dewey_classification_file', 'dewey_classification.txt')
        self.llm = stage_param['model_name']
        
        # Load Dewey classifications
        self.dewey_classifications = self.load_dewey_classifications()

    def load_dewey_classifications(self):
        """Load Dewey classifications from the configured file"""
        dewey_file = Path(self.dewey_classification_file)
        
        # If not absolute path, look relative to current working directory
        if not dewey_file.is_absolute():
            dewey_file = Path.cwd() / dewey_file
        
        if not dewey_file.exists():
            logging.warning("Dewey classification file not found at: %s. Dewey queries will not include classification reference.", dewey_file)
            return ""
        
        try:
            with open(dewey_file, 'r', encoding='utf-8') as f:
                classifications = f.read()
            logging.info("Loaded Dewey classifications from: %s", dewey_file)
            return classifications
        except (IOError, OSError) as e:
            logging.error("Error loading Dewey classifications: %s", e)
            return ""

    def get_valid_dewey_notations(self):
        """Extract valid Dewey notations from the classifications"""
        if not self.dewey_classifications:
            return set()
        
        valid_notations = set()
        for line in self.dewey_classifications.split('\n'):
            if ':' in line:
                notation = line.split(':')[0].strip()
                if notation and notation.replace('.', '').isdigit():
                    valid_notations.add(notation)
        return valid_notations

    def _normalize_affiliation(self, affiliation):
        """Normalize and clean up affiliation names"""
        if not affiliation:
            return ""
        
        # Common abbreviations and their full forms
        university_mappings = {
            # Technical Universities
            'tu dresden': 'Technische Universität Dresden',
            'tu berlin': 'Technische Universität Berlin', 
            'tu münchen': 'Technische Universität München',
            'tu darmstadt': 'Technische Universität Darmstadt',
            'tu braunschweig': 'Technische Universität Braunschweig',
            'tu clausthal': 'Technische Universität Clausthal',
            'tu dortmund': 'Technische Universität Dortmund',
            'tum': 'Technische Universität München',
            'tud': 'Technische Universität Dresden',
            'tub': 'Technische Universität Berlin',
            
            # Universities of Applied Sciences
            'fh aachen': 'FH Aachen',
            'fh köln': 'FH Köln',
            'fh münchen': 'Hochschule München',
            'hs münchen': 'Hochschule München',
            
            # Special cases
            'rwth aachen': 'RWTH Aachen',
            'kit karlsruhe': 'Karlsruher Institut für Technologie',
            'lmu münchen': 'Ludwig-Maximilians-Universität München',
            'uni hamburg': 'Universität Hamburg',
            'uni köln': 'Universität zu Köln',
            'uni frankfurt': 'Goethe-Universität Frankfurt am Main',
            'uni heidelberg': 'Ruprecht-Karls-Universität Heidelberg',
            'uni freiburg': 'Albert-Ludwigs-Universität Freiburg',
            'uni würzburg': 'Julius-Maximilians-Universität Würzburg',
            'uni göttingen': 'Georg-August-Universität Göttingen',
            'uni bonn': 'Rheinische Friedrich-Wilhelms-Universität Bonn',
            'uni mainz': 'Johannes Gutenberg-Universität Mainz',
            'uni stuttgart': 'Universität Stuttgart',
            'uni karlsruhe': 'Karlsruher Institut für Technologie',
            
            # International
            'mit': 'Massachusetts Institute of Technology',
            'stanford': 'Stanford University',
            'harvard': 'Harvard University',
            'oxford': 'University of Oxford',
            'cambridge': 'University of Cambridge'
        }
        
        # Clean the input
        cleaned = affiliation.strip()
        
        # Remove common prefixes/suffixes that don't add value
        remove_patterns = [
            r'^an der\s+',
            r'^von der\s+',
            r'^der\s+',
            r'^die\s+',
            r'^das\s+',
            r'\s*-\s*universität$',
            r'\s*university$',
            r'\s*institut$',
            r'\s*institute$'
        ]
        
        for pattern in remove_patterns:
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE).strip()
        
        # Check for direct mappings
        cleaned_lower = cleaned.lower()
        for abbrev, full_name in university_mappings.items():
            if abbrev in cleaned_lower:
                return full_name
        
        # If no mapping found, clean up common patterns
        # Capitalize first letters of words
        words = cleaned.split()
        capitalized_words = []
        
        for word in words:
            # Keep certain abbreviations in uppercase
            if word.upper() in ['TU', 'FH', 'TUM', 'TUD', 'TUB', 'RWTH', 'KIT', 'LMU', 'MIT']:
                capitalized_words.append(word.upper())
            elif word.lower() in ['universität', 'university', 'hochschule', 'institut', 'institute', 'college']:
                capitalized_words.append(word.capitalize())
            elif len(word) > 1:
                capitalized_words.append(word.capitalize())
            else:
                capitalized_words.append(word)
        
        result = ' '.join(capitalized_words)
        
        # Final cleanup - ensure consistent naming
        result = re.sub(r'\s+', ' ', result).strip()
        
        return result

    def _get_dewey_sample(self, thematic_areas=None):
        """Get a representative sample of Dewey classifications, optionally filtered by themes"""
        if not self.dewey_classifications:
            return ""
        
        lines = [line for line in self.dewey_classifications.split('\n') if ':' in line and not '[Unbesetzt]' in line]
        
        sample_lines = []
        
        # If thematic areas are provided, prioritize relevant classifications
        if thematic_areas:
            theme_keywords = {
                'informatik': ['000:', '004:', '005:', '006:'],
                'computer': ['000:', '004:', '005:', '006:'],
                'mathematik': ['510:', '511:', '512:', '513:', '514:', '515:', '516:', '517:', '518:', '519:'],
                'physik': ['530:', '531:', '532:', '533:', '534:', '535:', '536:', '537:', '538:', '539:'],
                'chemie': ['540:', '541:', '542:', '543:', '546:', '547:', '548:', '549:'],
                'biologie': ['570:', '571:', '572:', '573:', '574:', '575:', '576:', '577:', '578:', '579:'],
                'medizin': ['610:', '611:', '612:', '613:', '614:', '615:', '616:', '617:', '618:'],
                'technik': ['600:', '620:', '621:', '622:', '623:', '624:', '625:', '627:', '628:', '629:'],
                'ingenieur': ['620:', '621:', '622:', '623:', '624:', '625:', '627:', '628:', '629:'],
                'bildung': ['370:', '371:', '372:', '373:', '374:', '375:', '378:', '379:'],
                'pädagogik': ['370:', '371:', '372:', '373:', '374:', '375:', '378:', '379:'],
                'erziehung': ['370:', '371:', '372:', '373:', '374:', '375:', '378:', '379:'],
                'philosophie': ['100:', '101:', '110:', '120:', '130:', '140:', '150:', '160:', '170:', '180:', '190:'],
                'psychologie': ['150:', '151:', '152:', '153:', '154:', '155:', '156:', '158:'],
                'religion': ['200:', '210:', '220:', '230:', '240:', '250:', '260:', '270:', '280:', '290:'],
                'geschichte': ['900:', '930:', '940:', '950:', '960:', '970:', '980:', '990:'],
                'geografie': ['910:', '911:', '912:', '913:', '914:', '915:', '916:', '917:', '918:', '919:'],
                'wirtschaft': ['330:', '331:', '332:', '333:', '334:', '335:', '336:', '337:', '338:', '339:'],
                'recht': ['340:', '341:', '342:', '343:', '344:', '345:', '346:', '347:', '348:', '349:'],
                'politik': ['320:', '321:', '322:', '323:', '324:', '325:', '326:', '327:', '328:'],
                'soziologie': ['301:', '302:', '303:', '304:', '305:', '306:', '307:'],
                'sprache': ['400:', '410:', '420:', '430:', '440:', '450:', '460:', '470:', '480:', '490:'],
                'literatur': ['800:', '810:', '820:', '830:', '840:', '850:', '860:', '870:', '880:', '890:'],
                'kunst': ['700:', '710:', '720:', '730:', '740:', '750:', '760:', '770:', '780:', '790:'],
                'musik': ['780:', '781:', '782:', '783:', '784:', '785:', '786:', '787:', '788:']
            }
            
            thematic_lower = thematic_areas.lower()
            for theme, prefixes in theme_keywords.items():
                if theme in thematic_lower:
                    for line in lines:
                        if any(line.startswith(prefix) for prefix in prefixes):
                            sample_lines.append(line)
            
            # If we found thematic matches, use primarily those
            if sample_lines:
                # Add a few general categories for context
                general_categories = ['000:', '100:', '200:', '300:', '400:', '500:', '600:', '700:', '800:', '900:']
                for line in lines[:50]:
                    if any(line.startswith(cat) for cat in general_categories):
                        if line not in sample_lines:
                            sample_lines.append(line)
                
                # Limit to reasonable size
                return '\n'.join(sorted(sample_lines[:50]))
        
        # Fallback: Take every 15th classification for general coverage
        for i in range(0, len(lines), 15):
            sample_lines.append(lines[i])
        
        # Include key general categories
        important_categories = ['000:', '004:', '370:', '500:', '600:', '700:', '800:', '900:']
        for line in lines[:100]:
            if any(line.startswith(cat) for cat in important_categories):
                if line not in sample_lines:
                    sample_lines.append(line)
        
        # Remove duplicates and sort
        unique_lines = list(set(sample_lines))
        return '\n'.join(sorted(unique_lines))

    def execute_task(self):
        # vgl. https://github.com/encode/httpcore/blob/master/httpcore/_sync/http11.py
        logging.getLogger("httpcore.http11").setLevel(logging.CRITICAL)
       
        # https://github.com/langchain-ai/langchain/discussions/19256
        logging.getLogger("langchain_text_splitters.base").setLevel(logging.ERROR)
        logging.getLogger("urllib3.connectionpool").setLevel(logging.CRITICAL)
        logging.getLogger("httpx").setLevel(logging.CRITICAL)
        logging.getLogger("httpcore").setLevel(logging.CRITICAL)
        logging.getLogger("unstructured.trace").setLevel(logging.CRITICAL)

        df_files = pd.read_pickle(self.file_file_name_inputs)
        df_files = df_files[df_files['pipe:file_type'].isin(self.file_types)]

        if Path(self.file_file_name_output).exists():
            df_metadata = pd.read_pickle(self.file_file_name_output)
        else:
            df_metadata = pd.DataFrame()

        nc = NameChecker()

        embeddings = OllamaEmbeddings(
            base_url="http://localhost:11434",
            model="jina/jina-embeddings-v2-base-de"
        )

        vectorstore = Chroma(
            collection_name="oer_connected_lecturer",
            embedding_function=embeddings,
            persist_directory=str(self.chroma_file)
        )
        
        chroma_client = chromadb.PersistentClient(path=str(self.chroma_file))
        collection = chroma_client.get_or_create_collection(
            name="oer_connected_lecturer"
        )       
        name_result= collection.get()
        filenames_list = [x["filename"] for x in name_result['metadatas']]
        chunk_counter = Counter(filenames_list)

        for _, row in tqdm(df_files.iterrows(), total=df_files.shape[0]):
            # Check if the file type is supported
            if row['pipe:file_type'] not in self.file_types:
                continue
            
            file = (row['pipe:ID'] + "." + row['pipe:file_type'])
            
            # Check if we have existing metadata for this file
            existing_metadata = None
            if df_metadata.shape[0] > 0:
                existing_rows = df_metadata[df_metadata['pipe:ID'] == row['pipe:ID']]
                if existing_rows.shape[0] > 0:
                    existing_metadata = existing_rows.iloc[0]

            # Initialize metadata structure
            metadata_list_sample = {}
            metadata_list_sample['pipe:ID'] = row['pipe:ID']
            metadata_list_sample['pipe:file_type'] = row['pipe:file_type']
            
            # Copy existing metadata if available
            if existing_metadata is not None:
                for key in existing_metadata.index:
                    if key.startswith('ai:'):
                        # Ensure ai:dewey is always a list
                        if key == 'ai:dewey':
                            value = existing_metadata[key]
                            # Handle different data types safely
                            if value is None or value == "":
                                metadata_list_sample[key] = []
                            elif isinstance(value, list):
                                metadata_list_sample[key] = value
                            else:
                                # Try to check if it's NaN for scalar values
                                try:
                                    if pd.isna(value):
                                        metadata_list_sample[key] = []
                                    else:
                                        metadata_list_sample[key] = []
                                except (TypeError, ValueError):
                                    # If pd.isna fails, assume it's not NaN
                                    metadata_list_sample[key] = []
                        else:
                            metadata_list_sample[key] = existing_metadata[key]

            pages = chunk_counter.get(file, 0)
            if pages == 0:
                continue

            # Check which fields need to be processed
            needs_processing = False
            
            # Check if author processing is needed
            need_author = (existing_metadata is None or 
                          'ai:author' not in existing_metadata or 
                          safe_is_empty(existing_metadata.get('ai:author')))
            
            # Check if affiliation processing is needed
            need_affiliation = (existing_metadata is None or 
                               'ai:affilation' not in existing_metadata or 
                               safe_is_empty(existing_metadata.get('ai:affilation')))
            
            # Check if keywords_gen processing is needed
            need_keywords_gen = (existing_metadata is None or 
                                'ai:keywords_gen' not in existing_metadata or 
                                safe_is_empty(existing_metadata.get('ai:keywords_gen')))

            if not (need_author or need_affiliation or need_keywords_gen):
                continue

            retriever_with_filter = vectorstore.as_retriever( 
                search_kwargs={"filter":{"filename":file},
                               "k": pages})

            # Verwende den Template aus der JSON-Datei
            prompt = PromptTemplate.from_template(self.prompts["system_template"])
            chain = load_qa_chain(retriever_with_filter,
                                  OllamaLLM(model=self.llm, temperature=0), 
                                  prompt)

            # Process author if needed
            if need_author:
                author_query = self.prompts["author_query"].replace("{file}", file)
                authors = get_monitored_response(author_query, chain)
                metadata_list_sample['ai:author'] = authors
                metadata_list_sample['ai:revisedAuthor'] = nc.get_all_names(authors)
                needs_processing = True

            # ALWAYS process affiliation (force overwrite for improved algorithm)
            # Multi-step affiliation detection for better accuracy
            
            # Step 1: Look for explicit affiliation mentions
            affiliation_query = self.prompts["affiliation_query"].replace("{file}", file)
            affiliation_response = get_monitored_response(affiliation_query, chain)
            
            # Step 2: Enhanced affiliation search with common patterns
            enhanced_affiliation_query = f"""Suche in dem Dokument {file} nach Hinweisen auf die Herkunftsinstitution des Autors.
                
Achte dabei besonders auf:
- Universitätsnamen (vollständig oder abgekürzt)
- Hochschulnamen (TU, FH, Universität, Hochschule)
- Institutsnamen oder Fachbereiche
- E-Mail-Adressen mit Universitätsdomains (.edu, .ac.*, universitätsspezifische Domains)
- Adressen mit Universitätsbezug
- Kopf- oder Fußzeilen mit Institutionsnamen

Beispiele für deutsche Universitäten:
- TU Dresden, TU Berlin, TU München
- Universität Hamburg, Uni Köln, LMU München
- FH Aachen, Hochschule München
- RWTH Aachen, KIT Karlsruhe

Gib nur den Namen der Institution zurück. Falls mehrere gefunden werden, nimm die prominenteste.
Falls nichts gefunden wird, gib einen leeren String zurück."""
            
            enhanced_affiliation = get_monitored_response(enhanced_affiliation_query, chain)
            
            # Step 3: Email domain analysis
            email_domain_query = f"""Suche in dem Dokument {file} nach E-Mail-Adressen und analysiere die Domains.
                
Universitätsdomains können sein:
- .edu (internationale Universitäten)
- .ac.uk, .ac.at (akademische Domains)
- tu-dresden.de, uni-hamburg.de, etc.
- fh-aachen.de, hs-münchen.de, etc.

Falls du eine E-Mail-Adresse mit einer Universitätsdomain findest, 
leite daraus den Namen der Universität ab.

Beispiele:
- max.mustermann@tu-dresden.de → TU Dresden
- anna.beispiel@uni-hamburg.de → Universität Hamburg
- test@fh-aachen.de → FH Aachen

Gib nur den abgeleiteten Universitätsnamen zurück oder einen leeren String."""
            
            email_affiliation = get_monitored_response(email_domain_query, chain)
            
            # Process all responses
            affiliations = []
            for response in [affiliation_response, enhanced_affiliation, email_affiliation]:
                filtered_response = filtered(response)
                if filtered_response and len(filtered_response.strip()) > 3:
                    affiliations.append(filtered_response.strip())
            
            # Step 4: Validation and normalization
            final_affiliation = ""
            if affiliations:
                # Take the most detailed/longest response that seems valid
                valid_affiliations = []
                
                for affiliation in affiliations:
                    # Check if affiliation contains university-related keywords
                    university_keywords = [
                        'universität', 'university', 'hochschule', 'institut', 'college',
                        'tu ', 'fh ', 'uni ', 'rwth', 'kit', 'lmu', 'tum',
                        'fachhochschule', 'technische universität', 'akademie'
                    ]
                    
                    affiliation_lower = affiliation.lower()
                    if any(keyword in affiliation_lower for keyword in university_keywords):
                        # Verify the affiliation appears in the document
                        suchergebnisse = collection.get(
                            where={"filename": {"$eq": file}},
                            include=["documents"]
                        )
                        content = " ".join(suchergebnisse['documents']).replace("\n", " ").lower()
                        
                        # More flexible matching - check for parts of the affiliation
                        affiliation_words = affiliation_lower.split()
                        if len(affiliation_words) >= 2:
                            # Check if at least 2 significant words appear in content
                            significant_words = [w for w in affiliation_words if len(w) > 3]
                            matches = sum(1 for word in significant_words if word in content)
                            if matches >= min(2, len(significant_words)):
                                valid_affiliations.append(affiliation)
                        elif affiliation_lower in content:
                            valid_affiliations.append(affiliation)
                
                # Select the best affiliation
                if valid_affiliations:
                    # Prefer the longest/most detailed one
                    final_affiliation = max(valid_affiliations, key=len)
                    
                    # Clean up the result
                    final_affiliation = self._normalize_affiliation(final_affiliation)
            
            metadata_list_sample['ai:affilation'] = final_affiliation
            
            # Affiliation processing completed - always mark as needing processing
            needs_processing = True

            # Process keywords_gen if needed
            if need_keywords_gen:
                keywords_gen_query = self.prompts["keywords_gen_query"].replace("{file}", file)
                keywords2 = get_monitored_response(keywords_gen_query, chain)
                metadata_list_sample['ai:keywords_gen'] = filtered(keywords2)
                needs_processing = True

            # Process other fields if they don't exist yet (keeping original behavior for these)
            if existing_metadata is None or 'ai:title' not in existing_metadata or safe_is_empty(existing_metadata.get('ai:title')):
                title_query = self.prompts["title_query"].replace("{file}", file)
                title = get_monitored_response(title_query, chain)
                metadata_list_sample['ai:title'] = filtered(title)
                needs_processing = True

            if existing_metadata is None or 'ai:type' not in existing_metadata or safe_is_empty(existing_metadata.get('ai:type')):
                document_type_query = self.prompts["document_type_query"].replace("{file}", file)
                document_type = get_monitored_response(document_type_query, chain)
                metadata_list_sample['ai:type'] = filtered(document_type)
                needs_processing = True

            if existing_metadata is None or 'ai:keywords_ext' not in existing_metadata or safe_is_empty(existing_metadata.get('ai:keywords_ext')):
                keywords_ext_query = self.prompts["keywords_ext_query"].replace("{file}", file)
                keywords = get_monitored_response(keywords_ext_query, chain)
                metadata_list_sample['ai:keywords_ext'] = filtered(keywords)
                needs_processing = True

            if existing_metadata is None or 'ai:keywords_dnb' not in existing_metadata or safe_is_empty(existing_metadata.get('ai:keywords_dnb')):
                keywords_dnb_query = self.prompts["keywords_dnb_query"].replace("{file}", file)
                keywords3 = get_monitored_response(keywords_dnb_query, chain)
                metadata_list_sample['ai:keywords_dnb'] = filtered(keywords3)
                needs_processing = True

            # ALWAYS process Dewey classification (force overwrite for improved algorithm)
            # Two-step Dewey classification process
            
            # Step 1: Get thematic assessment from LLM
            thematic_query = f"""Analysiere den Inhalt des Dokuments {file} und beschreibe die Hauptthemen.

Welche fachlichen Bereiche werden behandelt? 
Nenne 3-5 Hauptthemen oder Fachbereiche in Stichworten (z.B. "Informatik", "Mathematik", "Pädagogik", "Geschichte", etc.).

Antworte nur mit den Themenbereichen, getrennt durch Kommas."""
            
            thematic_areas = get_monitored_response(thematic_query, chain)
            thematic_areas_filtered = filtered(thematic_areas)
            
            print(f"DEBUG - Step 1 - Thematic areas: {thematic_areas_filtered}")
            
            # Step 2: Map themes to official Dewey classifications
            if thematic_areas_filtered:
                dewey_sample = self._get_dewey_sample(thematic_areas_filtered)
                mapping_query = f"""Basierend auf den identifizierten Themenbereichen: "{thematic_areas_filtered}"

Ordne diese Themen zu passenden Dewey-Dezimalklassifikationen zu.

Verwende NUR die folgenden offiziellen Dewey-Klassifikationen:

{dewey_sample}

Wähle maximal 3 passende Klassifikationen aus der obigen Liste aus.
Antworte NUR mit einem gültigen JSON-Array im Format:
[{{"notation": "XXX", "label": "Beschreibung", "score": 0.X}}]

Keine Erklärungen oder zusätzlicher Text."""
                
                dewey = get_monitored_response(mapping_query, chain)
            else:
                # Fallback to original approach if step 1 fails
                dewey_query_base = self.prompts["dewey_query"].replace("{file}", file)
                if self.dewey_classifications:
                    dewey_sample = self._get_dewey_sample()
                    dewey = get_monitored_response(f"{dewey_query_base}\n\nVerfügbare Klassifikationen:\n{dewey_sample}", chain)
                else:
                    dewey = get_monitored_response(dewey_query_base, chain)
            
            dewey_answer = filtered(dewey)
            print(f"DEBUG - Step 2 - Raw Dewey response: {dewey_answer[:200]}...")

            # Always initialize as empty list
            metadata_list_sample['ai:dewey'] = []
            
            if dewey_answer != "":
                print(f"DEBUG - Filtered Dewey response: {dewey_answer}")
                
                # Try to clean and parse the response
                dewey_parsed = clean_and_parse_json_response(dewey_answer)
                print(f"DEBUG - Parsed JSON: {dewey_parsed}")
                
                if dewey_parsed is not None:
                    valid_notations = self.get_valid_dewey_notations()
                    print(f"DEBUG - Valid notations count: {len(valid_notations)}")
                    
                    if has_valid_dewey_classification(dewey_parsed, valid_notations):
                        metadata_list_sample['ai:dewey'] = dewey_parsed
                        print(f"DEBUG - Successfully assigned Dewey: {dewey_parsed}")
                    else:
                        print("DEBUG - No valid Dewey classification found")
                else:
                    print("DEBUG - Failed to parse JSON response")
            else:
                print("DEBUG - Empty Dewey response after filtering")
            
            needs_processing = True

            # Skip if no processing was needed
            if not needs_processing:
                continue

            # Build output string with only changed fields
            output_lines = [f"File      : {(row['pipe:ID'] + '.' + row['pipe:file_type'])}"]
            
            # Always show author info (either newly processed or existing)
            if need_author and 'ai:author' in metadata_list_sample:
                # Show newly processed author
                author_text = f"Author    : {metadata_list_sample['ai:author']}"
                if 'ai:revisedAuthor' in metadata_list_sample:
                    author_text += f" / {metadata_list_sample['ai:revisedAuthor']}"
                output_lines.append(author_text)
            elif existing_metadata is not None and 'ai:author' in existing_metadata:
                # Show existing author info
                author_text = f"Author    : {existing_metadata['ai:author']}"
                if 'ai:revisedAuthor' in existing_metadata:
                    author_text += f" / {existing_metadata['ai:revisedAuthor']}"
                output_lines.append(author_text)
            
            # Always show affiliation (since we always process it now)
            if 'ai:affilation' in metadata_list_sample:
                output_lines.append(f"Affilation: {metadata_list_sample['ai:affilation']}")
            
            # Only show fields that were actually processed conditionally
            if need_keywords_gen and 'ai:keywords_gen' in metadata_list_sample:
                output_lines.append(f"Keywords2 : {metadata_list_sample['ai:keywords_gen']}")
            
            # Show other processed fields
            if 'ai:title' in metadata_list_sample and (existing_metadata is None or 'ai:title' not in existing_metadata or safe_is_empty(existing_metadata.get('ai:title'))):
                output_lines.append(f"Title     : {metadata_list_sample['ai:title']}")
            
            if 'ai:type' in metadata_list_sample and (existing_metadata is None or 'ai:type' not in existing_metadata or safe_is_empty(existing_metadata.get('ai:type'))):
                output_lines.append(f"Typ       : {metadata_list_sample['ai:type']}")
            
            if 'ai:keywords_ext' in metadata_list_sample and (existing_metadata is None or 'ai:keywords_ext' not in existing_metadata or safe_is_empty(existing_metadata.get('ai:keywords_ext'))):
                output_lines.append(f"Keywords  : {metadata_list_sample['ai:keywords_ext']}")
            
            if 'ai:keywords_dnb' in metadata_list_sample and (existing_metadata is None or 'ai:keywords_dnb' not in existing_metadata or safe_is_empty(existing_metadata.get('ai:keywords_dnb'))):
                output_lines.append(f"Keywords3 : {metadata_list_sample['ai:keywords_dnb']}")
            
            # Always show Dewey classification (since we always process it now)
            if 'ai:dewey' in metadata_list_sample:
                output_lines.append(f"Dewey     : {metadata_list_sample['ai:dewey']}")
            
            # Print the formatted output
            print("\n" + "\n            ".join(output_lines) + "\n")

            # Update or add metadata
            if existing_metadata is not None:
                # Remove existing row and add updated row to avoid broadcasting issues
                df_metadata = df_metadata[df_metadata['pipe:ID'] != row['pipe:ID']]
                df_aux = pd.DataFrame([metadata_list_sample])        
                df_metadata = pd.concat([df_metadata, df_aux], ignore_index=True)
            else:
                # Ensure ai:dewey is always a list for new rows
                if 'ai:dewey' not in metadata_list_sample:
                    metadata_list_sample['ai:dewey'] = []
                    
                # Add new row
                df_aux = pd.DataFrame([metadata_list_sample])        
                df_metadata = pd.concat([df_metadata, df_aux], ignore_index=True)
            
            df_metadata.to_pickle(self.file_file_name_output)

        df_metadata.reset_index(drop=True, inplace=True)
        df_metadata.to_pickle(self.file_file_name_output)
