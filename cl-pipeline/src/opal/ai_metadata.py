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

def has_valid_dewey_classification(json_response: str) -> bool:
    try:
        data = json.loads(json_response)
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

        #llm_gemma = OllamaLLM(model="gemma3:27b", temperature=0)

        for _, row in tqdm(df_files.iterrows(), total=df_files.shape[0]):
            #Check if the ai metadata already exists for the file 
            if df_metadata.shape[0] > 0:
                if df_metadata[df_metadata['pipe:ID'] == row['pipe:ID']].shape[0] > 0:
                    continue

            metadata_list_sample = {}
            metadata_list_sample['pipe:ID'] = row['pipe:ID']
            metadata_list_sample['pipe:file_type'] = row['pipe:file_type']
            if row['pipe:file_type'] not in self.file_types:
                continue
            
            file = (row['pipe:ID'] + "." + row['pipe:file_type'])

            pages = chunk_counter.get(file, 0)
            if pages == 0:
                continue

            retriever_with_filter = vectorstore.as_retriever( 
                search_kwargs={"filter":{"filename":file},
                               "k": pages})

            # Verwende den Template aus der JSON-Datei
            prompt = PromptTemplate.from_template(self.prompts["system_template"])
            chain = load_qa_chain(retriever_with_filter,
                                  OllamaLLM(model=self.llm, temperature=0), 
                                  prompt)

            # Ersetze {file} in der Anfrage
            author_query = self.prompts["author_query"].replace("{file}", file)
            authors = get_monitored_response(author_query, chain)

            metadata_list_sample['ai:author'] = authors
            metadata_list_sample['ai:revisedAuthor'] = nc.get_all_names(authors)

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

            title_query = self.prompts["title_query"].replace("{file}", file)
            title = get_monitored_response(title_query, chain)
            metadata_list_sample['ai:title'] = filtered(title)

            document_type_query = self.prompts["document_type_query"].replace("{file}", file)
            document_type = get_monitored_response(document_type_query, chain)
            metadata_list_sample['ai:type'] = filtered(document_type)

            keywords_ext_query = self.prompts["keywords_ext_query"].replace("{file}", file)
            keywords = get_monitored_response(keywords_ext_query, chain)
            metadata_list_sample['ai:keywords_ext'] = filtered(keywords)

            keywords_gen_query = self.prompts["keywords_gen_query"].replace("{file}", file)
            keywords2 = get_monitored_response(keywords_gen_query, chain)
            metadata_list_sample['ai:keywords_gen'] = filtered(keywords2)

            keywords_dnb_query = self.prompts["keywords_dnb_query"].replace("{file}", file)
            keywords3 = get_monitored_response(keywords_dnb_query, chain)
            metadata_list_sample['ai:keywords_dnb'] = filtered(keywords3)

            dewey_query = self.prompts["dewey_query"].replace("{file}", file)
            dewey = get_monitored_response(dewey_query, chain)
            dewey_answer = filtered(dewey)
            if dewey_answer != "":
                dewey_answer = clean_and_parse_json_response(dewey_answer)

            if dewey_answer is not None:
                if has_valid_dewey_classification(dewey_answer):
                    metadata_list_sample['ai:dewey'] = dewey_answer
                else:
                    metadata_list_sample['ai:dewey'] = ""
            else:
                metadata_list_sample['ai:dewey'] = ""

            print(f"""
            File      : {(row['pipe:ID'] + "." + row['pipe:file_type'])}
            Author    : {authors} / {metadata_list_sample['ai:revisedAuthor']}
            Affilation: {metadata_list_sample['ai:affilation']}
            Title     : {filtered(title)}
            Typ       : {filtered(document_type)}
            Keywords  : {filtered(keywords)}
            Keywords3 : {filtered(keywords3)}
            Dewey     : {dewey_answer} 
            """)

            # Teste ob alle dict einträge deren Keys in check_keys genannt sind leer sind
            check_keys = ["ai:keywords_gen", "ai:keywords_ext", "ai:keywords_dnb", "ai:affilation", "ai:author", "ai:title", "ai:type"]
            if all(metadata_list_sample[key] == "" for key in check_keys):
                continue

            df_aux = pd.DataFrame([metadata_list_sample])        

            df_metadata = pd.concat([ df_metadata, df_aux])
            df_metadata.to_pickle(self.file_file_name_output)

        df_metadata.reset_index(drop=True, inplace=True)
        df_metadata.to_pickle(self.file_file_name_output)
