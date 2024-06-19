import pandas as pd
from pathlib import Path
from tqdm import tqdm
import urllib.request
import urllib.error
import os

from pipeline.taskfactory import TaskWithInputFileMonitor

class DownloadPdfs(TaskWithInputFileMonitor):
    def __init__(self, config_stage, config_global):
        super().__init__(config_stage, config_global)
        stage_param = config_stage['parameters']
        self.file_file_name_inputs =  Path(config_global['raw_data_folder']) / stage_param['file_file_name_input']
        self.file_file_name_output =  Path(config_global['raw_data_folder']) / stage_param['file_file_name_output']
        self.pdf_folder = Path(config_global['pdf_file_folder'])

    def execute_task(self):

        df_files = pd.read_pickle(self.file_file_name_inputs)
        df_files['download_date'] = pd.to_datetime('now')

        for index, row in tqdm(df_files.iterrows(), total=df_files.shape[0]):
            if row['file_type'] != 'pdf':
                continue
            pdf_path = self.pdf_folder / (row['ID'] + ".pdf")
            df_files.loc[index, "error_download"] = 'none'
            if (not os.path.exists(pdf_path)):
                try:
                    pdf_filename, headers = urllib.request.urlretrieve(row['oer_permalink'], pdf_path)
                except urllib.error.HTTPError as e:
                    print("Error while downloading.")
                    if e.code == 404:
                        df_files.loc[index, "error_download"] = 'Missing(404)'
                    else:
                        df_files.loc[index, "error_download"] = f'Unknown download error {e.code}'
                except:
                    df_files.loc[index, "error_download"] = 'Unknown error'

        df_files.to_pickle(self.file_file_name_output)