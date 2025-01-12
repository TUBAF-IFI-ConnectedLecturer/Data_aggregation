import pandas as pd
from pathlib import Path
import urllib.request
import os
import json
import logging
from tqdm import tqdm

from pipeline.taskfactory import Task, loggedExecution

import sys
sys.path.append('../src/general/')
from checkAuthorNames import NameChecker

class CollectOPALOERdocuments(Task):
    def __init__(self, config_stage, config_global):
        super().__init__(config_stage, config_global)
        stage_param = config_stage['parameters']
        self.json_file =  Path(config_global['raw_data_folder']) / stage_param['json_file_name']
        self.file_file =  Path(config_global['raw_data_folder']) / stage_param['file_file_name']
        self.repo_file_name =  Path(config_global['raw_data_folder']) / stage_param['repo_file_name']
        self.json_url = stage_param['json_url']

    def execute_task(self):
        if not self.json_file.exists():
            logging.info(f"OPAL data set already downloaded.")
          
            try:
                urllib.request.urlretrieve(self.json_url, self.json_file)
            except Exception as e:
                logging.error(f"Download failed: {e}")
                return
        
        logging.info(f"OPAL data set successfully downloaded.")

        with open(self.json_file) as f:
            raw_data = json.load(f)

        df_files = pd.DataFrame(raw_data['files'])
        renaming_dict = {
            'filename': 'opal:filename',
            'oer_permalink': 'opal:oer_permalink',
            'license': 'opal:license',
            'creator': 'opal:author',
            'title': 'opal:title',
            'comment': 'opal:comment',
            'language': 'opal:language',
            'publicationMonth': 'opal:publicationMonth',
            'publicationYear': 'opal:publicationYear',
        }
        df_files = df_files.rename(columns=renaming_dict)
        # Remove all columns that not in the renaming_dict
        df_files = df_files[renaming_dict.values()]

        df_files = df_files[renaming_dict.values()]

        # Evaluate author names
        logging.getLogger('urllib3').setLevel(logging.CRITICAL)
        logging.getLogger("urllib3").propagate = False
        logging.getLogger('requests').setLevel(logging.CRITICAL)

        nc = NameChecker()
        df_files.loc[:, 'opal:revisedAuthor'] = ""
        for index, row in tqdm(df_files.iterrows(), total=df_files.shape[0]):
            if row['opal:author'] != "":
                result = nc.get_validated_name(row['opal:author'])
                if result is not None:
                    df_files.at[index, 'opal:revisedAuthor'] = f"{result.Vorname}/{result.Familienname}"
                    #print(f"{result.Vorname} {result.Familienname}")

        df_files.to_pickle(self.file_file)