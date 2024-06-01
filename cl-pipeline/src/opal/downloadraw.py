import pandas as pd
from pathlib import Path
import urllib.request
import os
import json
import logging

from pipeline.taskfactory import Task, loggedExecution

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
        df_lr = pd.DataFrame(raw_data['learning_resources'])    
    
        df_files.to_pickle(self.file_file)
        df_lr.to_pickle(self.repo_file_name)