import pandas as pd
from pathlib import Path
from tqdm import tqdm
import urllib.request
import urllib.error
import os
import hashlib
import warnings

from langchain.schema import Document
import json
from typing import Iterable
import logging
from tqdm import tqdm
from wrapt_timeout_decorator import *

from langchain_community.document_loaders import (UnstructuredPowerPointLoader, 
                                                  UnstructuredExcelLoader,
                                                  UnstructuredMarkdownLoader,
                                                  UnstructuredWordDocumentLoader,
                                                  PyMuPDFLoader)

from langdetect import detect_langs
from pipeline.taskfactory import TaskWithInputFileMonitor

# Define a dictionary to map file extensions to their respective loaders
loaders = {
    'pdf':  PyMuPDFLoader,
    'pptx': UnstructuredPowerPointLoader,
    'md':   UnstructuredMarkdownLoader,
    'docx': UnstructuredWordDocumentLoader,
    'xlsx': UnstructuredExcelLoader
}

def get_loader_for_file_type(file_type, file_path):
    loader_class = loaders[file_type]
    # Baseloader seams not to work with current Pathlib objects
    return loader_class(file_path=str(file_path))

@timeout(120)
def provide_content(file_path, file_type):
    loader = get_loader_for_file_type(file_type, file_path)
    try:
        docs = loader.load()
    except:
        docs = None
    return docs

class ExtractFileContent(TaskWithInputFileMonitor):
    def __init__(self, config_stage, config_global):
        super().__init__(config_stage, config_global)
        stage_param = config_stage['parameters']
        self.json_file_folder = Path(config_global['raw_data_folder'])
        self.file_file_name_inputs =  Path(config_global['raw_data_folder']) / stage_param['file_file_name_input']
        self.file_file_name_output =  Path(config_global['raw_data_folder']) / stage_param['file_file_name_output']
        self.file_folder = Path(config_global['file_folder'])
        self.content_folder = Path(config_global['content_folder'])
        self.file_types = stage_param['file_types']

    def execute_task(self):

        logging.getLogger().setLevel(logging.ERROR)
        df_files = pd.read_pickle(self.file_file_name_inputs)

        if Path(self.file_file_name_output).exists():
            df_content = pd.read_pickle(self.file_file_name_output)
        else:
            df_content = pd.DataFrame()

        for file_type in self.file_types:
            if file_type not in loaders:
                raise ValueError(f"Loader for file type '{file_type}' not found.")

        warnings.filterwarnings("ignore", category=UserWarning, module="langchain")

        for index, row in tqdm(df_files.iterrows(), total=df_files.shape[0]):
            # Check if the ai metadata already exists for the file 
            if df_content.shape[0] > 0:
                if df_content[df_content['pipe:ID'] == row['pipe:ID']].shape[0] > 0:
                    continue

            if row['pipe:file_type'] not in self.file_types:
                continue

            file_path = self.file_folder / (row['pipe:ID'] + "." + row['pipe:file_type'])
            try:
                docs = provide_content(file_path, row['pipe:file_type'])
            except:
                print("Stopped due to timeout!")
                continue

            if docs == None:
                continue
            content = "".join(doc.page_content for doc in docs)
            
            if content != "":
                file_path = self.content_folder / (row['pipe:ID'] + ".txt")
                with open(file_path, 'w') as f:
                    f.write(content)

                content_list_sample = {}
                content_list_sample['pipe:ID'] = row['pipe:ID']
                content_list_sample['pipe:file_type'] = row['pipe:file_type']
                hash_object = hashlib.sha256(str(content).encode('utf-8'))
                hex_dig = hash_object.hexdigest()
                content_list_sample['pipe:content_hash'] = hex_dig
                content_list_sample['pipe:content_pages'] = len(docs)
                content_list_sample['pipe:content_words'] = len(content.split())
                
                try:
                    languages = detect_langs(content)
                except:
                    languages = []
                if languages:
                    most_probable = max(languages, key=lambda lang: lang.prob)
                    language, probability = most_probable.lang, most_probable.prob
                else:
                    language, probability = None, None
                content_list_sample['pipe:most_prob_language'] = language
                content_list_sample['pipe:language_probability'] = probability 

                df_aux = pd.DataFrame([content_list_sample])
                if df_aux.isna().all().all():
                    print(df_aux)
                    raise ValueError("Empty dataframe")

                df_content = pd.concat([ df_content, df_aux])
                df_content.to_pickle(self.file_file_name_output)
                # just for testing
                df_content.to_csv(self.file_file_name_output.with_suffix('.csv'))

        logging.info(f"Finished extracting content of {df_content.shape[0]} files")
        df_content.reset_index(drop=True, inplace=True)
        df_content.to_pickle(self.file_file_name_output)
        