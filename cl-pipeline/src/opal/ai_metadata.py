# Motivated by https://medium.com/@onkarmishra/using-langchain-for-question-answering-on-own-data-3af0a82789ed
# Thanks to Onkar Mishra for the inspiration

import pandas as pd
from pathlib import Path
import numpy as np
from typing import List, Dict, Any, Optional
from langchain import PromptTemplate
from langchain_core.documents import Document
#from langchain_community.llms import Ollama
from langchain_ollama import OllamaLLM
from langchain_core.vectorstores import VectorStore
from langchain.chains import RetrievalQA
from tqdm import tqdm
import logging
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
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
You are an AI agent trained to answer questions based solely on the content of the document.
Please only answer using information found within the document.
If the answer cannot be found in the document, return an empty text.
Do not invent answers or provide any additional information.
Avoid any introductions or explanations such as "I am an AI agent" or "Thank you for asking."
Only answer based on the content of the document.

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
        collection = chroma_client.get_or_create_collection(
            name="oer_connected_lecturer"
        )       
        name_result= collection.get()
        filenames_list = [x["filename"] for x in name_result['metadatas']]
        chunk_counter = Counter(filenames_list)

        #llm_deepseek = OllamaLLM(model="deepseek-r1", temperature=0)
        #llm_llama = OllamaLLM(model="llama3.1", temperature=0)
        llm_gemma = OllamaLLM(model="gemma3:27b", temperature=0)
        #llm = OllamaLLM(model="llama3.3:70b-instruct-q2_K", temperature=0)
        #llm = OllamaLLM(model="phi4:14b-q8_0")

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

            prompt = PromptTemplate.from_template(template)
            chain = load_qa_chain(retriever_with_filter, llm_gemma, prompt)

            author = get_monitored_response(f"""
                Who is the author (or authors) of the document {file}?
                Do not include any additional information. Just reply with the name(s) only — no 
                explanations, no phrases like "The author is".
                Your answer should be in German.""", chain)
            metadata_list_sample['ai:author'] = filtered(author)

            name_result = nc.get_validated_name(filtered(author))
            metadata_list_sample['ai:revisedAuthor'] = ""
            if name_result is not None:
                metadata_list_sample['ai:revisedAuthor'] = f"{name_result.Vorname}/{name_result.Familienname}"

            affilation = get_monitored_response(f"""
                On which university or university of applied sciences was the document {file} written?
                Only look at the first page. If you cannot find any university name, return an 
                empty string — no explanation.
                If you do find one, return only the name, in German — no extra words like 
                “The university is” or “This was written at ...""", chain)

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

            title = get_monitored_response(f"""
                What is the title or main heading of the document {file}?
                Look at the first page only.
                If no title is found, return an empty string — do not write "unknown" or anything else.
                If a title is found, return only the title — no introductory phrases or explanations.
                Your answer should be in German.""", chain)
            metadata_list_sample['ai:title'] = filtered(title)

            document_type = get_monitored_response(f"""
                You are given a document {file}. Determine what type of material it is.
                Choose exactly one of the following categories (German label), based on the English description:
                + "Aufgabenblatt" = Exercise sheet: a list of tasks or problems, typically used in class or for homework.
                + "Vorlesungsfolien" = Lecture slides: visual slides used in lectures or presentations.
                + "Skript" = Lecture script: structured written notes for a lecture, often textbook-like.
                + "Paper" = Scientific paper: academic or peer-reviewed research article.
                + "Buch" = Book: a full-length published book.
                + "Buchauszug" = Book excerpt: a section or chapter taken from a book.
                + "Seminararbeit" = Seminar paper: a short academic paper submitted for a seminar (e.g., 5–15 pages).
                + "Bachelorarbeit" = Bachelor's thesis: final thesis for a bachelor's degree.
                + "Masterarbeit" = Master's thesis: final thesis for a master's degree.
                + "Doktorarbeit" = Doctoral dissertation: dissertation written to obtain a doctoral degree.
                + "Dokumentation" = Documentation: user, technical, or project documentation.
                + "Tutorial" = Tutorial: instructional guide or how-to with practical steps.
                + "Präsentation" = Presentation: general presentation document, not necessarily lecture-related.
                + "Poster" = Poster: academic or scientific poster used at a conference.
                + "Protokoll" = Protocol / Report: a record of an experiment, meeting, or session.
                + "Sonstiges" = Other: if no category fits clearly.
                If you are not sure or cannot determine the type confidently, return an empty string.
                Otherwise, return only the German category name listed above — no explanation or additional text.
                The output must be in German.""", chain)
            metadata_list_sample['ai:type'] = filtered(document_type)

            keywords = get_monitored_response(f"""
                Extract at least 10 precise German keywords from the document {file}.
                The keywords should be suitable for library cataloging (bibliothekarische Erschließung).
                Focus on specific and content-relevant terms — avoid generic words or phrases.
                Return a comma-separated list of keywords in German, with no introduction or explanation.
                Output only the list, in German.""", chain)
            metadata_list_sample['ai:keywords_ext'] = filtered(keywords)

            keywords2 = get_monitored_response(f"""
                Generate at least 10 precise German keywords describing the content of the document {file}.
                The keywords should be suitable for library cataloging (bibliothekarische Erschließung).
                Focus on specific and content-relevant terms — avoid generic words or phrases.
                Return a comma-separated list of keywords in German, with no introduction or explanation.
                Output only the list, in German.""", chain)
            metadata_list_sample['ai:keywords_gen'] = filtered(keywords2)

            keywords3 = get_monitored_response(f"""
                Assign 10 keywords from the Gemeinsame Normdatei (GND) to the document {file}.
                If no specific GND keywords are available for certain concepts, use broader terms where possible, even if they are more general.
                Combine keywords into chains to specify specialized concepts more accurately, especially when only general terms are available.
                Return only a comma-separated list of the GND keywords — no introduction or explanation.
                The answer must be in German.""", chain)
            metadata_list_sample['ai:keywords_dnb'] = filtered(keywords3)

            dewey = get_monitored_response(f"""
                Assign the document {file} to the corresponding Dewey Decimal Classification.
                Respond first with the classification number, followed by a comma and the class name.
                If no clear classification can be found, return an empty string.
                Do not explain why no classification is found.
                The answer must be in German.""", chain)
            dewey_answer = filtered(dewey)
            metadata_list_sample['ai:dewey'] = ""
            metadata_list_sample['ai:dewey_name'] = ""
            if dewey_answer:
                dewey = dewey_answer.split(",")
                if len(dewey) == 2:
                    metadata_list_sample['ai:dewey'] = dewey[0].strip()
                    metadata_list_sample['ai:dewey_name'] = dewey[1].strip()

            print(f"""
            File      : {(row['pipe:ID'] + "." + row['pipe:file_type'])}
            Author    : {filtered(author)} / {metadata_list_sample['ai:revisedAuthor']}
            Affilation: {metadata_list_sample['ai:affilation']}
            Title     : {filtered(title)}
            Typ       : {filtered(document_type)}
            Keywords  : {filtered(keywords)}
            Keywords3 : {filtered(keywords3)}
            Dewey     : {metadata_list_sample['ai:dewey']} ({metadata_list_sample['ai:dewey_name']})
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
