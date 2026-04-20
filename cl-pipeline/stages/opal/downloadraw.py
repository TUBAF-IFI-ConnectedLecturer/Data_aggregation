import pandas as pd
from pathlib import Path
import urllib.request
import os
import json
import logging
from tqdm import tqdm

from pipeline.taskfactory import Task, loggedExecution

# Import zentrale Logging-Konfiguration
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))
from pipeline_logging import setup_stage_logging

import sys
sys.path.append('../src/general/')
from checkAuthorNames import NameChecker

class CollectOPALOERdocuments(Task):
    def __init__(self, config_stage, config_global):
        super().__init__(config_stage, config_global)

        # Setup zentrale Logging-Konfiguration
        self.logger_configurator = setup_stage_logging(config_global)

        stage_param = config_stage['parameters']
        self.json_file =  Path(config_global['raw_data_folder']) / stage_param['json_file_name']
        self.file_file =  Path(config_global['raw_data_folder']) / stage_param['file_name']
        self.repo_file_name =  Path(config_global['raw_data_folder']) / stage_param['repo_file_name']
        self.json_url = stage_param['json_url']
        self.force_run = stage_param['force_run']
        self.skip_name_check = stage_param.get('skip_name_check', False)  # Optional: disable name checking

    @loggedExecution
    def execute_task(self):
        if self.json_file.exists() and not self.force_run:
            logging.debug(f"OPAL data set already exists.")
            return

        if not self.force_run:
            logging.info(f"OPAL data set is missing.")
        else:
            logging.info(f"Forcing download of a new OPAL data set.")
          
        try:
            urllib.request.urlretrieve(self.json_url, self.json_file)
        except Exception as e:
            logging.error(f"Download of {self.json_file.name} failed: {e}")
            return
        
        logging.debug(f"OPAL data set successfully downloaded.")

        with open(self.json_file) as f:
            raw_data = json.load(f)

        logging.debug(f"Extract metadata from json data set.")
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

        # Evaluate author names - Logging wird jetzt zentral konfiguriert
        # NOTE: This is CPU-intensive and should ideally be a separate stage after GPU operations
        # For now, this can be disabled via config to test GPU utilization

        logging.debug("Running name AI based name checks.")
        nc = NameChecker()
        df_files.loc[:, 'opal:revisedAuthor'] = ""
        count = 0

        # Check if name checking is enabled (can be disabled for performance testing)
        if self.skip_name_check:
            logging.debug("Name checking disabled (skipping for performance testing)")
        else:
            for index, row in tqdm(df_files.iterrows(), total=df_files.shape[0]):
                if row['opal:author'] != "":
                    try:
                        result = nc.get_all_names(row['opal:author'])
                        if result is not None:
                            count = count + 1
                            # Defensively convert to string - handles lists, dicts, etc.
                            if isinstance(result, str):
                                result_str = result
                            elif isinstance(result, list):
                                result_str = ",".join(str(x) for x in result)
                            else:
                                result_str = str(result)
                            df_files.at[index, 'opal:revisedAuthor'] = result_str
                    except Exception as e:
                        logging.warning(f"Error processing author '{row['opal:author']}': {e}")
                        continue

        logging.debug(f"{count} names validated.")
        df_files.to_pickle(self.file_file)
