try:
    from langchain_ollama import OllamaEmbeddings
except ImportError:
    from langchain_community.embeddings import OllamaEmbeddings
import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_core.documents import Document

import pandas as pd
from pathlib import Path
from tqdm import tqdm
import logging
from wrapt_timeout_decorator import *
import re

from pipeline.taskfactory import TaskWithInputFileMonitor

# Import zentrale Logging-Konfiguration
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))
from pipeline_logging import setup_stage_logging


def clean_text_content(text):
    """Bereinigt Text von unnötigen Elementen, erhält Markdown-Struktur."""
    # Horizontale Leerzeichen normalisieren (Zeilenumbrüche beibehalten)
    text = re.sub(r'[ \t]+', ' ', text)
    # Übermäßige Leerzeilen reduzieren
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Entfernen von typischen Header/Footer-Mustern
    text = re.sub(r'(?i)(confidential|intern|seite \d+|page \d+|www\.[\w\.]+)', '', text)
    # Sonderzeichen bereinigen, Markdown-Zeichen (#, *, |, >) und deutsche Umlaute beibehalten
    text = re.sub(r'[^\w\s\.,;:!?\(\)\[\]\{\}äöüÄÖÜß#*|>\-]', '', text)
    return text.strip()


def is_useful_chunk(text, min_length=100, min_words=15):
    """Prüft, ob ein Textabschnitt nützlichen Inhalt enthält."""
    if len(text) < min_length:
        return False

    word_count = len(text.split())
    if word_count < min_words:
        return False

    # Prüfen auf Vorhandensein vollständiger Sätze
    if not re.search(r'[A-ZÄÖÜ][^.!?]+[.!?]', text):
        return False

    return True


@timeout(60)
def load_and_split_markdown(md_file_path, header_splitter, text_splitter, source_filename):
    """
    Liest eine Markdown-Datei und splittet sie in zwei Stufen:
    Stufe 1: MarkdownHeaderTextSplitter — splittet an Überschriften
    Stufe 2: RecursiveCharacterTextSplitter — für zu große Abschnitte
    """
    try:
        with open(str(md_file_path), 'r', encoding='utf-8') as f:
            md_content = f.read()
    except Exception as e:
        logging.error(f"Error reading {md_file_path}: {e}")
        return None

    if not md_content.strip():
        return None

    # Stufe 1: Split nach Markdown-Überschriften
    header_docs = header_splitter.split_text(md_content)

    # Stufe 2: Zu große Abschnitte weiter splitten
    final_chunks = []
    for doc in header_docs:
        if len(doc.page_content) > text_splitter._chunk_size:
            sub_chunks = text_splitter.split_text(doc.page_content)
            for chunk_text in sub_chunks:
                chunk_text = clean_text_content(chunk_text)
                if is_useful_chunk(chunk_text):
                    metadata = dict(doc.metadata)
                    metadata['filename'] = source_filename
                    metadata['page'] = 0
                    final_chunks.append(Document(page_content=chunk_text, metadata=metadata))
        else:
            cleaned = clean_text_content(doc.page_content)
            if is_useful_chunk(cleaned):
                metadata = dict(doc.metadata)
                metadata['filename'] = source_filename
                metadata['page'] = 0
                final_chunks.append(Document(page_content=cleaned, metadata=metadata))

    return final_chunks if final_chunks else None


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

        # Filter only validated LiaScript files (if validation column exists)
        if 'pipe:is_valid_liascript' in df_files.columns:
            initial_count = len(df_files)
            df_files = df_files[df_files['pipe:is_valid_liascript'] == True]
            filtered_count = len(df_files)
            logging.info(f"Filtered files: {initial_count} -> {filtered_count} (removed {initial_count - filtered_count} invalid files)")

        df_files = df_files[df_files['pipe:file_type'].isin(self.file_types)]
        logging.info(f"Found {len(df_files)} files of type {self.file_types}")

        embeddings = OllamaEmbeddings(
            base_url=self.base_url,
            model=self.embedding_model,
        )

        # Stufe 1: Markdown-bewusstes Splitting nach Überschriften
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]
        header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on,
            strip_headers=False,
        )

        # Stufe 2: Größenbasiertes Splitting für zu große Abschnitte
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150,
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
        # Paginated get to avoid SQLite variable limit
        all_metadatas = []
        batch_size = 5000
        offset = 0
        while True:
            result = collection.get(limit=batch_size, offset=offset, include=["metadatas"])
            if not result['metadatas']:
                break
            all_metadatas.extend(result['metadatas'])
            if len(result['metadatas']) < batch_size:
                break
            offset += batch_size
        processed_files = set(x["filename"] for x in all_metadatas)
        logging.info(f"{len(processed_files)} in database found")

        # Batch-Größe definieren
        BATCH_SIZE = 16

        # Batch-Container initialisieren
        batch_ids = []
        batch_documents = []
        batch_metadatas = []

        # Process each file in the DataFrame
        for id, row in tqdm(df_files.iterrows(), total=len(df_files)):
            source_filename = row['pipe:ID'] + "." + row['pipe:file_type']
            if source_filename in processed_files:
                continue

            # Markdown-Datei aus content_folder lesen (statt Originaldatei erneut zu laden)
            md_file_path = self.content_folder / (row['pipe:ID'] + ".md")
            if not md_file_path.exists():
                logging.warning(f"Markdown content file not found: {md_file_path}")
                continue

            try:
                split_docs = load_and_split_markdown(
                    md_file_path, header_splitter, text_splitter, source_filename
                )
            except:
                print("Stopped due to timeout!")
                continue

            if split_docs:
                for idx, doc in enumerate(split_docs):
                    # Zum Batch hinzufügen
                    batch_ids.append(str(id) + "_" + str(idx))
                    batch_documents.append(doc.page_content)
                    batch_metadatas.append({
                        "filename": doc.metadata['filename'],
                        "page": doc.metadata['page'],
                        "chunk_index": idx,
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
