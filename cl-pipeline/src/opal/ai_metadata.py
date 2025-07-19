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
    # Remove newlines
    AI_response = AI_response.replace("\n", "")
    # Check for blacklist words and remove them
    blacklist = ["don't know", "weiß nicht", "weiß es nicht", "Ich kenne ",
                 "Ich sehe kein", "Es gibt kein", "Ich kann ", "Ich sehe", "Es wird keine ",
                 "Entschuldigung", "Leider kann ich", "Keine Antwort", "Die Antwort kann ich",
                 "Der Autor", "die Frage", "Ich habe keine", "Ich habe ", "Ich brauche",
                 "Bitte geben", "Das Dokument ", "Es tut mir leid", "Es handelt sich", "Es ist nicht", "Es konnte kein"]
    if any(x in AI_response for x in blacklist):
        return ""
    else:
        return AI_response.strip()

def clean_and_parse_json_response(raw_response: str):
    # Entferne evtl. Markdown-Codeblock-Markierungen wie ```json ... ```
    cleaned = re.sub(r"^```json\s*|\s*```$", "", raw_response.strip(), flags=re.DOTALL)

    try:
        parsed = json.loads(cleaned)
        return parsed
    except json.JSONDecodeError as e:
        print("Fehler beim Parsen:", e)
        return None

def has_valid_dewey_classification(data) -> bool:
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
        logging.error(f"Error processing {query}: {str(e)}")
        return ""
    
def get_monitored_response(query, chain):
    try:
        return get_response(query, chain)
    except Exception as e:
        logging.error(f"Timeout of 60s {query}: {str(e)}")
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
        self.llm = stage_param['model_name']

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

            # Process affiliation if needed
            if need_affiliation:
                affiliation_query = self.prompts["affiliation_query"].replace("{file}", file)
                affilation = get_monitored_response(affiliation_query, chain)

                # Prüfe, ob die Affiliation im Text des Dokuments vorkommt.
                filtered_affilation = filtered(affilation)
                content = ""
                if filtered_affilation:
                    suchergebnisse = collection.get(
                        where={"filename": {"$eq": file}},
                        include=["documents"]
                    )
                    content = "".join(suchergebnisse['documents']).replace("\n", "")
                if filtered_affilation in content:
                    metadata_list_sample['ai:affilation'] = filtered_affilation
                else:
                    metadata_list_sample['ai:affilation'] = ""
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

            if existing_metadata is None or 'ai:dewey' not in existing_metadata or safe_is_empty(existing_metadata.get('ai:dewey')):
                dewey_query = self.prompts["dewey_query"].replace("{file}", file)
                dewey = get_monitored_response(dewey_query, chain)
                dewey_answer = filtered(dewey)
                
                # Always initialize as empty list
                metadata_list_sample['ai:dewey'] = []
                
                if dewey_answer != "":
                    dewey_answer = clean_and_parse_json_response(dewey_answer)
                    if dewey_answer is not None:
                        if has_valid_dewey_classification(dewey_answer):
                            metadata_list_sample['ai:dewey'] = dewey_answer
                
                needs_processing = True
            else:
                # Ensure existing dewey is always a list
                if 'ai:dewey' not in metadata_list_sample:
                    metadata_list_sample['ai:dewey'] = []

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
            
            # Only show fields that were actually processed
            if need_affiliation and 'ai:affilation' in metadata_list_sample:
                output_lines.append(f"Affilation: {metadata_list_sample['ai:affilation']}")
            
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
            
            if 'ai:dewey' in metadata_list_sample and (existing_metadata is None or 'ai:dewey' not in existing_metadata or safe_is_empty(existing_metadata.get('ai:dewey'))):
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
