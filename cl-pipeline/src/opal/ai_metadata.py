# Motivated by https://medium.com/@onkarmishra/using-langchain-for-question-answering-on-own-data-3af0a82789ed
# Thanks to Onkar Mishra for the inspiration

import pandas as pd
from pathlib import Path

import numpy as np
import faiss
from typing import List, Dict, Any, Optional
from langchain import PromptTemplate
from langchain_core.documents import Document
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_core.vectorstores import VectorStore
from langchain.chains import RetrievalQA
from tqdm import tqdm
import logging
from langchain.embeddings.base import Embeddings
from wrapt_timeout_decorator import *
from pathlib import Path
from langchain.vectorstores import Chroma
import chromadb
import re
from collections import Counter
from wrapt_timeout_decorator import *

from pipeline.taskfactory import TaskWithInputFileMonitor

import sys
sys.path.append('../src/general/')
from checkAuthorNames import NameChecker

def filtered(AI_response):
    # Check for deepseeg tags and remove explanations
    if "<think>" in AI_response:
        AI_response = re.sub(r'<think>.*?</think>', '', AI_response, flags=re.DOTALL)
    # Check for blacklist words and remove them
    blacklist = ["don't know", "weiß nicht", "weiß es nicht", "Ich kenne ",
                 "Ich sehe kein", "Es gibt kein", "Ich kann ", "Ich sehe",
                 "Entschuldigung", "Leider kann ich", "Keine Antwort",
                 "Der Autor", "die Frage", "Ich habe keine", "Ich habe ",
                 "Bitte geben", "Das Dokument ", "Es tut mir leid", "Es handelt sich"]
    if any(x in AI_response for x in blacklist):
        return ""
    else:
        return AI_response.strip()


# Creating the chain for Question Answering
def load_qa_chain(retriever, llm, prompt):
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever, 
        chain_type="stuff",
        return_source_documents=True, 
        chain_type_kwargs={'prompt': prompt} 
    )

@timeout(120)
def get_response(query, chain):
    # Getting response from chain
    try:
        response = chain({'query': query})
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

template = """
### System:
Du bist ein freundlicher KI-Agent, der darauf trainiert ist, Fragen
zu beantworten, die auf den Inhalten von individuellen 
Dokumenten basieren. Die Fragen des Benutzers dürfen 
nur mithilfe dieser Dokumente beantwortet werden. 
Wenn Du die Antwort nicht kennst, antworte mit einem 
leeren Text. Versuche nicht, eine 
Antwort zu erfinden. Sage niemals Danke, dass Du 
gern hilfst, dass du ein KI-Agent sind usw. 

### Context:
{context}

### User:
{question}

### Response:
"""

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
        self.temp_document_path = Path(config_global['processed_data_folder']) / "documents.pkl"

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
            #model="all-minilm",
            model="jina/jina-embeddings-v2-base-de"
            #model = load_embedding_model(model_path="all-MiniLM-L6-v2")
            #show_progress=True
        )

        vectorstore = Chroma(
            collection_name="oer_connected_lecturer",
            embedding_function=embeddings,
            persist_directory=str(self.chroma_file)
        )
        
        chroma_client = chromadb.PersistentClient(path=str(self.chroma_file))
        collection = chroma_client.get_or_create_collection(name="oer_connected_lecturer")       
        name_result= collection.get()
        filenames_list = [x["filename"] for x in name_result['metadatas']]
        chunk_counter = Counter(filenames_list)

        llm_deepseek = Ollama(model="deepseek-r1", temperature=0)
        llm_llama = Ollama(model="llama3.1", temperature=0)
        #llm = Ollama(model="llama3.3:70b-instruct-q2_K", temperature=0)
        #llm = Ollama(model="phi4:14b-q8_0")

        metadata_list = []
        for index, row in tqdm(df_files.iterrows(), total=df_files.shape[0]):

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

            prompt = PromptTemplate.from_template(template)
            chain = load_qa_chain(retriever_with_filter, llm_llama, prompt)

            author = get_monitored_response(f"""
                Wer ist der Autor oder die Autoren der Datei {file}. Vermeide alle zusätzlichen Informationen 
                und antworte  einfach mit dem Namen des Autors. Füge  nicht etwas wie `Der 
                Autor des Dokuments ist` hinzu. Bitte antworte  auf Deutsch.""", chain)
            metadata_list_sample['ai:author'] = filtered(author)

            name_result = nc.get_validated_name(filtered(author))
            if name_result is not None:
                df_files.at[index, 'ai:revisedAuthor'] = f"{name_result.Vorname}/{name_result.Familienname}"
            else:
                df_files.at[index, 'ai:revisedAuthor'] = ""

            affilation = get_monitored_response(f"""
                An welcher Institution (Universität, Hochschule) entstand das Dokument {file}?.
                Antworte einfach mit dem Namen der Universität und gegebenenfalls des 
                Fachbereichs oder der Fakultät, getrennt durch Komma. 
                Beginne nicht mit „Die Universität“. Bitte antworten Sie auf Deutsch.""", chain)
            metadata_list_sample['ai:affilation'] = filtered(affilation)

            title = get_monitored_response(f"""
                Wie lautet die Überschrift oder der Titel des Dokumentes {file}? Antworte
                einfach mit dem Titel und nicht in einem Satz. Bitte antworte auf Deutsch.""", chain)
            metadata_list_sample['ai:title'] = filtered(title)

            document_type = get_monitored_response(f"""
                Welcher Typ von Material liegt bei Dokument {file} vor? Unterscheide zwischen 
                Aufgabenblatt, Vorlesungsfolien, wissenschaftliches Paper, Buch, Buchauszug, 
                Seminararbeit, Doktorarbeit, Dokumentation, Tutorial usw. Antworte 
                nur mit einer Typbezeichnung.""", chain)
            metadata_list_sample['ai:type'] = filtered(document_type)

            keywords = get_monitored_response(f"""
                Bitte extrahiere mindestens 10 deutsche Schlagworte aus dem Dokument {file}.
                Bitte gib nur eine Liste deutscher Schlagworte zurück, die für eine 
                bibliothekarische Erschließung geeignet sind. Die Schlagworte sollten 
                präzise und spezifisch sein. Antworte einfach 
                durch eine durch Kommas getrennte Liste. Fügen nicht etwas einleitendes 
                wie `Hier sind 10 deutsche Schlagworte` etc. hinzu. Bitte antworte 
                auf Deutsch.""", chain)
            metadata_list_sample['ai:keywords_ext'] = filtered(keywords)

            keywords2 = get_monitored_response(f"""
                Generiere mindestens 10 deutsche Schlagworte, die den Inhalt des
                Dokumentes {file} repräsentieren.
                Bitte gib nur eine Liste deutscher Schlagworte zurück, die für eine 
                bibliothekarische Erschließung geeignet sind. Die Schlagworte sollten 
                möglichst präzise und spezifisch sein. Antworte einfach 
                durch eine durch Kommas getrennte Liste. Füge nicht etwas 
                einleitendes wie `Hier  sind 10 deutsche Schlagworte` etc. hinzu. Bitte 
                antworte auf Deutsch.""", chain)
            metadata_list_sample['ai:keywords_gen'] = filtered(keywords2)

            keywords3 = get_monitored_response(f"""
                Du bist ein erfahrener Bibliothekar und sollst das Dokument {file}
                inhaltlich erschließen. Ordne dem Inhalt 10 Schlagworte der Gemeinsame 
                Normdatei (GND) zu. Nutzen Sie dafür den Bestand der GND und 
                fügen Sie keine eigenen Schlagworte hinzu. Füge nicht etwas 
                einleitendes wie `Hier sind 10 Schlagwörter der Gemeinsamen 
                Normdatei (GND)` oder `Ich habe die Schlagwörter entnommen und 
                mit denen der GND abgeglichen.` etc. hinzu. Antworten einfach 
                durch eine mit Kommas getrennten Liste der Worte. 
                Bitte antworte auf Deutsch.""", chain)
            metadata_list_sample['ai:keywords_dnb'] = filtered(keywords3)

            dewey = get_monitored_response(f"""
                Bitte ordne das Dokument {file} entsprechend der 
                Dewey-Dezimalklassifizierung zu. Antworten Sie nur mit der 
                Klassifizierungsnummer. Wenn Du keine eindeutige Zuordnung findest,
                antworten mit einem leeren String. Erkläre nicht, 
                dass Du keine Klassifizierung finden kannst. Bitte antworte auf 
                Deutsch.""", chain)
            metadata_list_sample['ai:dewey'] = filtered(dewey)

            if name_result is not None:
                revised_author = f"{name_result.Vorname}/{name_result.Familienname}"
            else:
                revised_author = ""
            print(f"""
            File     : {(row['pipe:ID'] + "." + row['pipe:file_type'])}
            Author   : {filtered(author)} / {revised_author}
            Title    : {filtered(title)}
            Typ      : {filtered(document_type)}
            Keywords : {filtered(keywords)}
            Keywords3: {filtered(keywords3)}
            """)

            metadata_list=[]
            metadata_list.append(metadata_list_sample)
            df_aux = pd.DataFrame(metadata_list)        

            df_metadata = pd.concat([ df_metadata, df_aux])
            df_metadata.to_pickle(self.file_file_name_output)

        df_metadata.reset_index(drop=True, inplace=True)
        df_metadata.to_pickle(self.file_file_name_output)
