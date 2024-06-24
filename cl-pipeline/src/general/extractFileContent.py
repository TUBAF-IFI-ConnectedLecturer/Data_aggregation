import pandas as pd
from pathlib import Path
from tqdm import tqdm
import urllib.request
import urllib.error
import os

from  langchain.schema import Document
import json
from typing import Iterable
import logging

from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders.pdf import PyMuPDFLoader
from langchain_community.document_loaders import (UnstructuredPowerPointLoader, 
                                                 UnstructuredMarkdownLoader,
                                                 UnstructuredWordDocumentLoader)


from pipeline.taskfactory import TaskWithInputFileMonitor

# https://medium.com/@kamaljp/loading-pdf-data-into-langchain-to-use-or-not-to-use-unstructured-4eb220a15f4d


def save_docs_to_jsonl(array:Iterable[Document], file_path:str)->None:
    with open(file_path, 'w') as jsonl_file:
        for doc in array:
            jsonl_file.write(doc.json() + '\n')

def load_docs_from_jsonl(file_path)->Iterable[Document]:
    array = []
    with open(file_path, 'r') as jsonl_file:
        for line in jsonl_file:
            data = json.loads(line)
            obj = Document(**data)
            array.append(obj)
    return array

# Define a dictionary to map file extensions to their respective loaders
loaders = {
    'pdf': PyMuPDFLoader,
    'pptx': UnstructuredPowerPointLoader,
    'md': UnstructuredMarkdownLoader,
    'docx': UnstructuredWordDocumentLoader,
}

def create_directory_loader(file_type, directory_path):
    return DirectoryLoader(
        path=directory_path,
        glob=f"**/*{file_type}",
        loader_cls=loaders[file_type],
        show_progress=True,
        silent_errors=True
)

class ExtractFileContent(TaskWithInputFileMonitor):
    def __init__(self, config_stage, config_global):
        super().__init__(config_stage, config_global)
        stage_param = config_stage['parameters']
        self.json_file_folder = Path(config_global['raw_data_folder'])
        self.file_file_name_inputs =  Path(config_global['raw_data_folder']) / stage_param['file_file_name_input']
        self.file_file_name_output =  Path(config_global['raw_data_folder']) / stage_param['file_file_name_output']
        self.file_folder = Path(config_global['file_folder'])
        self.file_types = stage_param['file_types']

    def execute_task(self):

        #set environment variables
        os.environ["UNSTRUCTURED_NARRATIVE_TEXT_CAP_THRESHOLD"] = "1.0"
        
        logging.getLogger().setLevel(logging.ERROR)
        df_files = pd.read_pickle(self.file_file_name_inputs)

        for file_type in self.file_types:
            if file_type not in loaders:
                raise ValueError(f"Loader for file type '{file_type}' not found.")
            
            loader = create_directory_loader(file_type, self.file_folder)
            documents = loader.load()
            file_path_json = self.json_file_folder / ("content_" + file_type + ".json")
            save_docs_to_jsonl(documents, file_path_json)