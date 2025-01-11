# Motivated by https://medium.com/@onkarmishra/using-langchain-for-question-answering-on-own-data-3af0a82789ed
# Thanks to Onkar Mishra for the inspiration

import pandas as pd
from pathlib import Path

from langchain import PromptTemplate
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_community.document_loaders import (UnstructuredPowerPointLoader, 
                                                  UnstructuredExcelLoader,
                                                  UnstructuredMarkdownLoader,
                                                  UnstructuredWordDocumentLoader,
                                                  PyMuPDFLoader,
                                                  DirectoryLoader)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from tqdm import tqdm
import logging
from wrapt_timeout_decorator import *

from pipeline.taskfactory import TaskWithInputFileMonitor


# Responsible for splitting the documents into several chunks
def split_docs(documents, chunk_size=1000, chunk_overlap=20):
    
    # Initializing the RecursiveCharacterTextSplitter with
    # chunk_size and chunk_overlap
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    # Splitting the documents into chunks
    chunks = text_splitter.split_documents(documents=documents)
    
    # returning the document chunks
    return chunks

# function for loading the embedding model
def load_embedding_model(model_path, normalize_embedding=True):
    return HuggingFaceEmbeddings(
        model_name=model_path,
        model_kwargs={'device':'cpu'}, # here we will run the model with CPU only
        encode_kwargs = {
            'normalize_embeddings': normalize_embedding # keep True to compute cosine similarity
        }
    )

# Function for creating embeddings using FAISS
def create_embeddings(chunks, embedding_model, storing_path="vectorstore"):
    # Creating the embeddings using FAISS
    vectorstore = FAISS.from_documents(chunks, embedding_model)
    
    # Saving the model in current directory
    vectorstore.save_local(storing_path)
    
    # returning the vectorstore
    return vectorstore

# Creating the chain for Question Answering
def load_qa_chain(retriever, llm, prompt):
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever, # here we are using the vectorstore as a retriever
        chain_type="stuff",
        return_source_documents=True, # including source documents in output
        chain_type_kwargs={'prompt': prompt} # customizing the prompt
    )
    
def get_response(query, chain):
    # Getting response from chain
    response = chain({'query': query})
    return response['result']


template = """
### System:
You are an respectful and honest assistant. You have to answer the user's questions using only the context \
provided to you. If you don't know the answer, just say you don't know. Don't try to make up an answer.
Never say thank you, that you are happy to help, that you are an AI agent, etc. Just answer directly.

### Context:
{context}

### User:
{question}

### Response:
"""

# Define a dictionary to map file extensions to their respective loaders
loaders = {
    'pdf': PyMuPDFLoader,
    'pptx': UnstructuredPowerPointLoader,
    'md': UnstructuredMarkdownLoader,
    'docx': UnstructuredWordDocumentLoader,
    'xlsx': UnstructuredExcelLoader
}

def get_loader_for_file_type(file_type, file_path):
    loader_class = loaders[file_type]
    # Baseloader seams not to work with current Pathlib objects
    return loader_class(file_path=str(file_path))

@timeout(60)
def prepare_retriever(file_path, file_type, embed):
    loader = get_loader_for_file_type(file_type, file_path)
    try:
        docs = loader.load()
    except:
        return None

    documents = split_docs(documents=docs)
    if len(documents) == 0:
        return None 
    
    vectorstore = create_embeddings(documents, embed)
    retriever = vectorstore.as_retriever()
    return retriever

def filtered(AI_response):
    blacklist = ["don't know", "weiß nicht", "weiß es nicht"]
    if any(x in AI_response for x in blacklist):
        return ""
    else:
        return AI_response

class AIMetaDataExtraction(TaskWithInputFileMonitor):
    def __init__(self, config_stage, config_global):
        super().__init__(config_stage, config_global)
        stage_param = config_stage['parameters']
        self.file_file_name_inputs =  Path(config_global['raw_data_folder']) / stage_param['file_file_name_input']
        self.file_file_name_output =  Path(config_global['raw_data_folder']) / stage_param['file_file_name_output']
        self.file_folder = Path(config_global['file_folder'])
        self.file_types = stage_param['file_types']

    def execute_task(self):

        logging.getLogger('urllib3').setLevel(logging.CRITICAL)
        logging.getLogger('unstructured').setLevel(logging.CRITICAL)

        df_files = pd.read_pickle(self.file_file_name_inputs)

        if Path(self.file_file_name_output).exists():
            df_metadata = pd.read_pickle(self.file_file_name_output)
        else:
            df_metadata = pd.DataFrame()

        llm = Ollama(model="llama3", temperature=0)
        embed = load_embedding_model(model_path="all-MiniLM-L6-v2")

        for file_type in self.file_types:
            if file_type not in loaders:
                raise ValueError(f"Loader for file type '{file_type}' not found.")

        metadata_list = []
        for index, row in tqdm(df_files.iterrows(), total=df_files.shape[0]):
            # Check if the ai metadata already exists for the file 
            if df_metadata.shape[0] > 0:
                if df_metadata[df_metadata['pipe:ID'] == row['pipe:ID']].shape[0] > 0:
                    continue

            metadata_list_sample = {}
            metadata_list_sample['pipe:ID'] = row['pipe:ID']
            metadata_list_sample['pipe:file_type'] = row['pipe:file_type']
            if row['pipe:file_type'] not in self.file_types:
                continue

            file_path = self.file_folder / (row['pipe:ID'] + "." + row['pipe:file_type'])
            try:
                retriever = prepare_retriever(file_path, row['pipe:file_type'], embed)
            except:
                print("Stopped due to timeout!")
                continue

            if retriever== None:
                continue

            prompt = PromptTemplate.from_template(template)
            chain = load_qa_chain(retriever, llm, prompt)

            file = (row['pipe:ID'] + "." + row['pipe:file_type'])
            author = get_response(f"Who is the author of the document {file}. Avoid all additional information, just answer by authors name. Do not add something like 'The autor of the document is'. Please answer in German.", chain)
            metadata_list_sample['ai:author'] = filtered(author)

            affilation = get_response(f"Which university published the document {file}?. Just answer by the name of the university and the department, separated by comma. Do not start with `Die Universität`. Please answer in German.", chain)
            metadata_list_sample['ai:affilation'] = filtered(affilation)

            title = get_response(f"Extract the title of the document {file}. Just answer by the title. Please answer in German.", chain)
            metadata_list_sample['ai:title'] = filtered(title)

            keywords = get_response(f"Please extract 10 Keywords from {file}? Just answer by a list separted by commas. Answer in German.", chain)
            metadata_list_sample['ai:keywords'] = filtered(keywords)

            keywords2 = get_response(f"Please use 10 keywords to describe the content of {file}? Just answer by a list separted by commas. Answer in German.", chain)
            metadata_list_sample['ai:keywords2'] = filtered(keywords2)

            dewey = get_response(f"Please assign the document {file} according to the dewey decimal classification. Answer with the classification number only. Do not explain that you are not able to find a classification.", chain)
            metadata_list_sample['ai:dewey'] = filtered(dewey)

            print(filtered(author) + "-" + filtered(dewey))
            print(filtered(keywords))
            print(filtered(keywords2))
            print(filtered(affilation))

            metadata_list=[]
            metadata_list.append(metadata_list_sample)
            df_aux = pd.DataFrame(metadata_list)        

            df_metadata = pd.concat([ df_metadata, df_aux])
            df_metadata.to_pickle(self.file_file_name_output)

        df_metadata.reset_index(drop=True, inplace=True)
        df_metadata.to_pickle(self.file_file_name_output)