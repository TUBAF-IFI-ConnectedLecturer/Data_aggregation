import pandas as pd
from pathlib import Path
from tqdm import tqdm
import urllib.request
import urllib.error
import os

from pipeline.taskfactory import TaskWithInputFileMonitor

class DownloadOERFromOPAL(TaskWithInputFileMonitor):
    def __init__(self, config_stage, config_global):
        super().__init__(config_stage, config_global)
        stage_param = config_stage['parameters']
        self.file_folder = Path(config_global['file_folder'])
        self.raw_data_folder = Path(config_global['raw_data_folder'])
        self.file_file_name_inputs =  self.raw_data_folder / stage_param['file_file_name_input']
        self.file_file_name_output =  self.raw_data_folder / stage_param['file_file_name_output']
        self.file_types = stage_param['file_types']

    def execute_task(self):

        df_files = pd.read_pickle(self.file_file_name_inputs)

        download_list = []
        for index, row in tqdm(df_files.iterrows(), total=df_files.shape[0]):
            download_list_sample = {}

            if row['pipe:file_type'] not in self.file_types:
                continue
            file_path = self.file_folder / (row['pipe:ID'] + "." + row['pipe:file_type'])

            download_list_sample['pipe:ID'] = row['pipe:ID']
            download_list_sample['pipe:file_path'] = file_path
            download_list_sample['pipe:error_download'] = 'none'

            if (not os.path.exists(file_path)):
                download_list_sample['pipe:download_date'] = pd.to_datetime('now')
                try:
                    pdf_filename, headers = urllib.request.urlretrieve(row['oer_permalink'], file_path)
                except urllib.error.HTTPError as e:
                    print("Error while downloading.")
                    if e.code == 404:
                        download_list_sample["pipe:error_download"] = 'Missing(404)'
                    else:
                        download_list_sample["pipe:error_download"] = f'Unknown download error {e.code}'
                except:
                    download_list_sample["pipe:error_download"] = 'Unknown error'
            else:
                download_list_sample['pipe:download_date'] = pd.to_datetime(os.path.getmtime(file_path), unit='s')
            download_list.append(download_list_sample)

        df_download_list = pd.DataFrame(download_list)
        df_download_list.to_pickle(self.file_file_name_output)
