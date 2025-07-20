from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import (UnstructuredPowerPointLoader, 
                                                  UnstructuredExcelLoader,
                                                  UnstructuredMarkdownLoader,
                                                  UnstructuredWordDocumentLoader,
                                                  PyMuPDFLoader,
                                                  DirectoryLoader)
import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

import pandas as pd
from pathlib import Path
from tqdm import tqdm
import logging
from wrapt_timeout_decorator import *
import pickle
import re

from pipeline.taskfactory import TaskWithInputFileMonitor

# Import zentrale Logging-Konfiguration
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))
from pipeline_logging import setup_stage_logging


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

# Nach der load_and_split_document Funktion hinzufügen
def clean_text_content(text):
    """Bereinigt Text von unnötigen Elementen."""
    # Leerzeichen normalisieren
    text = re.sub(r'\s+', ' ', text)
    # Entfernen von typischen Header/Footer-Mustern
    text = re.sub(r'(?i)(confidential|intern|seite \d+|page \d+|www\.[\w\.]+)', '', text)
    # Sonderzeichen bereinigen, deutsche Umlaute beibehalten
    text = re.sub(r'[^\w\s\.,;:!?\(\)\[\]\{\}äöüÄÖÜß-]', '', text)
    return text.strip()

def is_useful_chunk(text, min_length=100, min_words=15):
    """Prüft, ob ein Textabschnitt nützlichen Inhalt enthält."""
    if len(text) < min_length:
        return False
        
    # Prüfen, ob genügend Wörter vorhanden sind
    word_count = len(text.split())
    if word_count < min_words:
        return False
    
    # Prüfen auf Vorhandensein vollständiger Sätze
    if not re.search(r'[A-ZÄÖÜ][^.!?]+[.!?]', text):
        return False
        
    return True

# In load_and_split_document modifizieren
@timeout(60)
def load_and_split_document(file_path, text_splitter):
    if file_path.suffix[1:] not in loaders:
        logger.warning(f"Unsupported file type: {file_path.suffix} for {file_path.name}")
        return None
        
    loader = get_loader_for_file_type(file_path)
    try:
        docs = loader.load()
        chunks = text_splitter.split_documents(docs)
        
        # Filtern und Bereinigen der Chunks
        filtered_chunks = []
        for chunk in chunks:
            chunk.page_content = clean_text_content(chunk.page_content)
            if is_useful_chunk(chunk.page_content):
                chunk.metadata['filename'] = file_path.name
                filtered_chunks.append(chunk)
                
        return filtered_chunks

    except Exception as e:
        logger.error(f"Error processing {file_path}: {str(e)}")
        return None

class AIEmbeddingsGeneration(TaskWithInputFileMonitor):
    def __init__(self, config_stage, config_global):
        super().__init__(config_stage, config_global)
        
        # Setup zentrale Logging-Konfiguration
        self.logger_configurator = setup_stage_logging(config_global)
        
        stage_param = config_stage['parameters']
        self.file_name_inputs =  Path(config_global['raw_data_folder']) / stage_param['file_name_input']
        self.file_name_output =  Path(config_global['raw_data_folder']) / stage_param['file_name_output']
        self.file_types = stage_param['file_types']
        self.file_folder = Path(config_global['file_folder'])
        self.content_folder = Path(config_global['content_folder'])
        self.processed_data_folder = config_global['processed_data_folder']
        self.chroma_file = Path(config_global['processed_data_folder']) / "chroma_db"
        
        # LLM configuration from config file
        self.llm_config = stage_param.get('llm_config', {})
        self.base_url = self.llm_config.get('base_url', 'http://localhost:11434')
        self.embedding_model = self.llm_config.get('embedding_model', 'jina/jina-embeddings-v2-base-de')
        self.collection_name = self.llm_config.get('collection_name', 'oer_connected_lecturer')

    def execute_task(self):
        # Logging wird jetzt zentral konfiguriert - keine lokalen Einstellungen mehr nötig

        df_files = pd.read_pickle(self.file_name_inputs)

        df_files = df_files[df_files['pipe:file_type'].isin(self.file_types)]
        #df_files = df_files.iloc[0:100]
        logging.info(f"Found {len(df_files)} files of type {self.file_types}")

        embeddings = OllamaEmbeddings(
            base_url=self.base_url,
            model=self.embedding_model
        )

        # Verbesserte Textsplitter-Konfiguration
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,               # Größere Chunks für mehr Kontext
            chunk_overlap=150,            # Größerer Überlapp für Kontexterhalt
            length_function=len,
            separators=[
                "\n\n",                   # Absätze bevorzugen
                "\n",                     # Dann Zeilenumbrüche
                ". ",                     # Dann Satzenden
                ", ",                     # Dann Kommas
                " ",                      # Dann Leerzeichen
                ""                        # Notfalls Zeichen trennen
            ]
        )

        chroma_client = chromadb.PersistentClient(path=str(self.chroma_file))
        collection = chroma_client.get_or_create_collection(name=self.collection_name)
        result = collection.get()
        processed_files = set([x["filename"] for x in result['metadatas']])
        logging.info(f"{len(processed_files)} in database found")

        # Batch-Größe definieren
        BATCH_SIZE = 16  # Anpassen basierend auf verfügbarem RAM

        # Batch-Container initialisieren
        batch_ids = []
        batch_documents = []
        batch_metadatas = []

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

                    # Zum Batch hinzufügen
                    batch_ids.append(str(id) + "_" + str(idx))
                    batch_documents.append(doc.page_content)
                    batch_metadatas.append({
                        "filename": doc.metadata['filename'],
                        "page": doc.metadata['page'],
                    })

                    # Wenn Batch-Größe erreicht ist, verarbeiten
                    if len(batch_ids) >= BATCH_SIZE:
                        # Batch-Embeddings erzeugen
                        batch_embeddings = embeddings.embed_documents(batch_documents)

                        # In Datenbank einfügen
                        collection.add(
                            ids=batch_ids,
                            documents=batch_documents,
                            embeddings=batch_embeddings,
                            metadatas=batch_metadatas
                        )

                        # Batch zurücksetzen
                        batch_ids = []
                        batch_documents = []
                        batch_metadatas = []

        # Restliche Dokumente verarbeiten
        if batch_ids:
            batch_embeddings = embeddings.embed_documents(batch_documents)
            collection.add(
                ids=batch_ids,
                documents=batch_documents,
                embeddings=batch_embeddings,
                metadatas=batch_metadatas
            )
            