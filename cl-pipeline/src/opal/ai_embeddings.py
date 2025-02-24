from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import (UnstructuredPowerPointLoader, 
                                                  UnstructuredExcelLoader,
                                                  UnstructuredMarkdownLoader,
                                                  UnstructuredWordDocumentLoader,
                                                  PyMuPDFLoader,
                                                  DirectoryLoader)
from langchain_community.vectorstores import FAISS
import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

import pandas as pd
from pathlib import Path
from tqdm import tqdm
import logging
from wrapt_timeout_decorator import *
import pickle

from pipeline.taskfactory import TaskWithInputFileMonitor


# Define a dictionary to map file extensions to their respective loaders
loaders = {
    'pdf': PyMuPDFLoader,
    'pptx': UnstructuredPowerPointLoader,
    'md': UnstructuredMarkdownLoader,
    'docx': UnstructuredWordDocumentLoader,
    'xlsx': UnstructuredExcelLoader
}

def get_loader_for_file_type(file_path): 
    loader_class = loaders[file_path.suffix[1:]]
    # Baseloader seams not to work with current Pathlib objects
    return loader_class(file_path=str(file_path))

@timeout(60)
def load_and_split_document(file_path, text_splitter):
    loader = get_loader_for_file_type(file_path)
    try:
        docs = loader.load()
        chunks = text_splitter.split_documents(docs)
        for chunk in chunks:
            chunk.metadata['filename'] = file_path.name
        return chunks

    except Exception as e:
        logger.error(f"Error processing {file_path}: {str(e)}")
        return None

class AIEmbeddingsGeneration(TaskWithInputFileMonitor):
    def __init__(self, config_stage, config_global):
        super().__init__(config_stage, config_global)
        stage_param = config_stage['parameters']
        self.file_file_name_inputs =  Path(config_global['raw_data_folder']) / stage_param['file_file_name_input']
        self.file_file_name_output =  Path(config_global['raw_data_folder']) / stage_param['file_file_name_output']
        self.file_types = stage_param['file_types']
        self.file_folder = Path(config_global['file_folder'])
        self.content_folder = Path(config_global['content_folder'])
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

        df_files = pd.read_pickle(self.file_file_name_inputs)

        df_files = df_files[df_files['pipe:file_type'].isin(self.file_types)]
        #df_files = df_files.iloc[0:100]
        logging.info(f"Found {len(df_files)} files of type {self.file_types}")

        embeddings = OllamaEmbeddings(
            base_url="http://localhost:11434",
            #model="all-minilm",
            model = "jina/jina-embeddings-v2-base-de"
            #show_progress=True
        )

        # Initialize text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )

        chroma_client = chromadb.PersistentClient(path=str(self.chroma_file))
        collection = chroma_client.get_or_create_collection(name="oer_connected_lecturer")
        result = collection.get()
        processed_files = set([x["filename"] for x in result['metadatas']])
        logging.info(f"{len(processed_files)} in database found")

        # Process each file in the DataFrame
        for id, row in tqdm(df_files.iterrows(), total=len(df_files)):       
            file_path = self.file_folder / (row['pipe:ID'] + "." + row['pipe:file_type'])
            if file_path.name in processed_files:
                continue

            try:
               split_docs = load_and_split_document(file_path, text_splitter)
            except:
               print("Stopped due to timeout!")
               continue
            if split_docs:
                for idx, doc in enumerate(split_docs):
                    if "page" not in doc.metadata:
                        doc.metadata["page"] = 0
                    collection.add(
                        ids = [str(id) + "_" + str(idx)],
                        documents = doc.page_content, 
                        embeddings = embeddings.embed_query(doc.page_content),
                        metadatas = {
                            "filename": doc.metadata['filename'],
                            "page": doc.metadata['page'],
                        }   
                    )
            