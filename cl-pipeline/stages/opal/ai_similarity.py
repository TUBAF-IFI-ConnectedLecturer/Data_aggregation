# Motivated by https://github.com/olonok69/LLM_Notebooks/blob/main/langchain/Langchain_chromadb_Ollama_doc_similarity.ipynb
# Thanks to olonok69 for the inspiration

import pandas as pd
from pathlib import Path
from tqdm import tqdm
import logging
from typing import List, Tuple
import numpy as np
import chromadb
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity

#from langchain_ollama import OllamaEmbeddings
from langchain_community.embeddings import OllamaEmbeddings

from sentence_transformers import SentenceTransformer

from pipeline.taskfactory import TaskWithInputFileMonitor

# Import zentrale Logging-Konfiguration
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))
from pipeline_logging import setup_stage_logging

class DocumentSimilarity(TaskWithInputFileMonitor):
    def __init__(self, config_stage, config_global):
        super().__init__(config_stage, config_global)
        
        # Setup zentrale Logging-Konfiguration
        self.logger_configurator = setup_stage_logging(config_global)
        
        stage_param = config_stage['parameters']
        self.file_name_inputs =  Path(config_global['raw_data_folder']) / stage_param['file_name_input']
        self.file_name_output =  Path(config_global['processed_data_folder']) / stage_param['file_name_output']
        self.file_folder = Path(config_global['file_folder'])
        self.processed_data_folder = config_global['processed_data_folder']
        self.chroma_file = Path(config_global['processed_data_folder']) / "chroma_db"
        
        # LLM configuration from config file
        self.llm_config = stage_param.get('llm_config', {})
        self.collection_name = self.llm_config.get('collection_name', 'oer_connected_lecturer')

    def execute_task(self):
        # Logging wird jetzt zentral konfiguriert

        df_files = pd.read_pickle(self.file_name_inputs)

        logging.info("Reading db files")
        chroma_client = chromadb.PersistentClient(path=str(self.chroma_file))
        collection = chroma_client.get_collection(
            name=self.collection_name
        )       
        results = collection.get(include=["embeddings", "metadatas", "documents"])
        logging.info("Starting similarity analysis")
        aggregated_embeddings = []
        filenames = []
        for index, row in tqdm(df_files.iterrows(), total=df_files.shape[0]):

            file = (row['pipe:ID'] + "." + row['pipe:file_type'])

            # 1. Hole alle Chunks, deren Metadatum `filename` entspricht
            results = collection.get(
                where={"filename": file},
                include=["embeddings"]
            )
            
            if len(results["embeddings"]) == 0:
                continue  # überspringe leere Dokumente

            # 2. Aggregiere die Embeddings (z. B. Mittelwert)
            emb = np.mean(np.vstack(results["embeddings"]), axis=0)

            # 3. Speichern
            aggregated_embeddings.append(emb)
            filenames.append(row['pipe:ID'])

        emb_matrix = np.vstack(aggregated_embeddings)
        similarity_matrix = cosine_similarity(emb_matrix)
        df_similarity = pd.DataFrame(similarity_matrix, index=filenames, columns=filenames)

        df_similarity.to_pickle(self.file_name_output)
