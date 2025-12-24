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
        self.file_name_inputs =  self.raw_data_folder / stage_param['file_name_input']
        self.file_name_output =  self.raw_data_folder / stage_param['file_name_output']
        self.file_types = stage_param['file_types']

        # Optional: Maximum number of downloads per file type
        self.max_downloads_per_type = stage_param.get('max_downloads_per_type', None)
        # Optional: Maximum total downloads (across all file types)
        self.max_total_downloads = stage_param.get('max_total_downloads', None)

    def execute_task(self):

        df_files = pd.read_pickle(self.file_name_inputs)

        # Apply stratified sampling if limits are specified
        if self.max_downloads_per_type or self.max_total_downloads:
            df_files = self._apply_download_limits(df_files)

        download_list = []
        for index, row in tqdm(df_files.iterrows(), total=df_files.shape[0]):
            if row['pipe:file_type'] not in self.file_types:
                continue

            # Start with all original row data
            download_list_sample = row.to_dict()

            file_path = self.file_folder / (row['pipe:ID'] + "." + row['pipe:file_type'])

            # Add/update download-specific fields
            download_list_sample['pipe:file_path'] = file_path
            download_list_sample['pipe:error_download'] = 'none'

            if (not os.path.exists(file_path)):
                download_list_sample['pipe:download_date'] = pd.to_datetime('now')
                try:
                    _, _ = urllib.request.urlretrieve(row['opal:oer_permalink'], file_path)
                except urllib.error.HTTPError as e:
                    print(f"Error while downloading {file_path.name}: HTTP {e.code}")
                    if e.code == 404:
                        download_list_sample["pipe:error_download"] = 'Missing(404)'
                    else:
                        download_list_sample["pipe:error_download"] = f'Unknown download error {e.code}'
                except Exception as e:
                    print(f"Error while downloading {file_path.name}: {str(e)}")
                    download_list_sample["pipe:error_download"] = 'Unknown error'
            else:
                download_list_sample['pipe:download_date'] = pd.to_datetime(os.path.getmtime(file_path), unit='s')
            download_list.append(download_list_sample)

        df_download_list = pd.DataFrame(download_list)
        df_download_list.to_pickle(self.file_name_output)

    def _apply_download_limits(self, df_files):
        """
        Apply stratified sampling to limit downloads per file type and total downloads.

        Args:
            df_files: DataFrame with all available files

        Returns:
            DataFrame with limited files based on sampling strategy
        """
        # Filter for requested file types
        df_filtered = df_files[df_files['pipe:file_type'].isin(self.file_types)]

        if self.max_downloads_per_type:
            # Stratified sampling: limit each file type separately
            sampled_dfs = []
            for file_type in self.file_types:
                df_type = df_filtered[df_filtered['pipe:file_type'] == file_type]
                if len(df_type) > self.max_downloads_per_type:
                    df_type = df_type.head(self.max_downloads_per_type)
                sampled_dfs.append(df_type)
            df_filtered = pd.concat(sampled_dfs, ignore_index=True)

        if self.max_total_downloads and len(df_filtered) > self.max_total_downloads:
            # Apply total download limit
            df_filtered = df_filtered.head(self.max_total_downloads)

        return df_filtered
