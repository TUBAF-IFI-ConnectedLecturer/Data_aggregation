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

class FilterFilesByContent(TaskWithInputFileMonitor):
    def __init__(self, config_stage, config_global):
        super().__init__(config_stage, config_global)
        stage_param = config_stage['parameters']
        self.json_file_folder = Path(config_global['raw_data_folder'])
        self.file_file_name_inputs =  Path(config_global['raw_data_folder']) / stage_param['file_file_name_input']
        self.file_file_content = Path(config_global['raw_data_folder']) / stage_param['file_file_content']
        self.file_file_name_output =  Path(config_global['raw_data_folder']) / stage_param['file_file_name_output']

    def execute_task(self):

        df_files = pd.read_pickle(self.file_file_name_inputs)
        df_content = pd.read_pickle(self.file_file_content)

        # eliminate duplicates
        df_content = df_content.drop_duplicates(subset=["pipe:content_hash"])

        # eliminate files with less than 90 words
        df_content = df_content[df_content["pipe:content_words"]>90]

        # elinimate files with less than 90% language probability and not german
        df_content = df_content[(df_content["pipe:language_probability"] > 0.9) & (df_content["pipe:most_prob_language"] == "de")]

        df_files_filtered = df_files[df_files["pipe:ID"].isin(df_content["pipe:ID"])]

        logging.info(f"Filtered {df_files_filtered.shape[0]} relevant files by content.")
        df_files_filtered.reset_index(drop=True, inplace=True)
        df_files_filtered.to_pickle(self.file_file_name_output)
        