# Motivated by https://github.com/olonok69/LLM_Notebooks/blob/main/langchain/Langchain_chromadb_Ollama_doc_similarity.ipynb
# Thanks to olonok69 for the inspiration

import pandas as pd
from pathlib import Path
from tqdm import tqdm
import logging
from typing import List, Tuple
import numpy as np

#from langchain_ollama import OllamaEmbeddings
from langchain_community.embeddings import OllamaEmbeddings

from sentence_transformers import SentenceTransformer

from pipeline.taskfactory import TaskWithInputFileMonitor

class DocumentSimilarity(TaskWithInputFileMonitor):
    def __init__(self, config_stage, config_global):
        super().__init__(config_stage, config_global)
        stage_param = config_stage['parameters']
        self.file_file_name_inputs =  Path(config_global['raw_data_folder']) / stage_param['file_file_name_input']
        self.file_file_name_output =  Path(config_global['raw_data_folder']) / stage_param['file_file_name_output']
        self.content_folder = Path(config_global['content_folder'])
        self.file_folder = Path(config_global['file_folder'])

    def get_embedding_model(self, model_name: str = "nomic-embed-text"):
        embeddings = OllamaEmbeddings(
            model=model_name,
            model_kwargs={
               "device": "gpu",
            }
        )
        return embeddings

    def get_embeddings_batch(self, embeddings_model, texts: List[str]) -> np.ndarray:
        """
        Generiert Embeddings für eine Liste von Texten
        """
        embeddings = embeddings_model.embed_documents(texts)
        return np.array(embeddings)

    def calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Berechnet die Kosinus-Ähnlichkeit zwischen zwei Embeddings
        """
        return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))

    def calculate_similarity_matrix(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Berechnet die Ähnlichkeitsmatrix für alle Embedding-Paare
        """
        normalized = embeddings / np.linalg.norm(embeddings, axis=1)[:, np.newaxis]
        return np.dot(normalized, normalized.T)

    def get_most_similar_files(self, correlation_matrix, file_names, n_similar=3):
            
        # Erstelle leere Listen für die Ergebnisse
        base_files = []
        similar_files = []
        similarity_scores = []

        # Iteriere über alle Dateien
        for i, file in enumerate(file_names):
            # Hole die Ähnlichkeitswerte für die aktuelle Datei
            similarities = correlation_matrix[i, :]
            
            # Setze den Ähnlichkeitswert mit sich selbst auf -1 um ihn auszuschließen
            similarities_without_self = similarities.copy()
            similarities_without_self[i] = -1
            
            # Finde die Indizes der n ähnlichsten Dateien
            most_similar_indices = np.argsort(similarities_without_self)[-n_similar:][::-1]
            
            # Füge die Ergebnisse den Listen hinzu
            for similar_idx in most_similar_indices:
                base_files.append(file)
                similar_files.append(file_names[similar_idx])
                similarity_scores.append(similarities[similar_idx])
        
        # Erstelle das Ergebnis-DataFrame
        results_df = pd.DataFrame({
            'ai:filename': base_files,
            'ai:filename_similar': similar_files,
            'ai:similarity': similarity_scores
        })
        
        return results_df

    def execute_task(self):
        logging.getLogger("urllib3").propagate = False

        embeddings_model = self.get_embedding_model()

        df_content = pd.read_pickle(self.file_file_name_inputs)

        if Path(self.file_file_name_output).exists():
            df_similarity = pd.read_pickle(self.file_file_name_output)
        else:
            df_similarity = pd.DataFrame()

        content=[]
        file_names = []
        logging.info("Reading content files")
        for _, row in tqdm(df_content.iterrows(), total=df_content.shape[0]):
            file_path = self.content_folder / (row['pipe:ID'] + ".txt")
            with open(file_path, 'r', encoding='utf-8') as file:
                raw_text = file.read()
                processed_text = raw_text.replace('\n', ' ').strip()
            content.append(processed_text)
            file_names.append(row['pipe:ID'])

        limit = len(content)   # Added for testing purposes to limit the number of files
        content = content[:limit]
        file_names = file_names[:limit]

        logging.info("Generating embeddings")
        embeddings = self.get_embeddings_batch(embeddings_model, content)
        logging.info("Calculating similarity matrix")
        similarity_matrix = self.calculate_similarity_matrix(embeddings)
        correlation_matrix = similarity_matrix.astype(np.float32)
        logging.info("Determining most similar files")
        df_similarity = self.get_most_similar_files(similarity_matrix, file_names)

        df_similarity.reset_index(drop=True, inplace=True)
        df_similarity.to_pickle(self.file_file_name_output)
        df_similarity.to_csv(self.file_file_name_output.with_suffix('.csv'), index=False)

