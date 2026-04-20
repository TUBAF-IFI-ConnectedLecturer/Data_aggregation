import pandas as pd
from pathlib import Path
from tqdm import tqdm
import urllib.request
import urllib.error
import os
import sys
import logging

from pipeline.taskfactory import TaskWithInputFileMonitor

# Import improved HTTP client with connection pooling and retry
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))
from error_handling.http_client import PooledHTTPClient, get_default_client, close_default_client
from error_handling.retry_strategy import RetryStrategy, RetryConfig

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

        # Initialize HTTP client with connection pooling
        # 120s timeout for large file downloads
        self.http_client = get_default_client() if hasattr(self, 'http_client') else None

        # Create a new client with longer timeout for this download task
        from error_handling.http_client import PooledHTTPClient
        self.http_client = PooledHTTPClient(
            timeout=120,  # 2 minutes for large files
            max_retries=3,
            pool_connections=5,
            pool_maxsize=5
        )

        # Initialize retry strategy for resilient downloads
        self.retry_config = RetryConfig(
            max_retries=3,
            initial_delay=2.0,  # Start with 2 seconds
            max_delay=30.0,     # Cap at 30 seconds
            exponential_base=2.0,
            add_jitter=True
        )
        self.retry_strategy = RetryStrategy(self.retry_config)

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
                # Use improved download method with retries
                error = self._download_with_retry(row['opal:oer_permalink'], file_path)
                download_list_sample['pipe:error_download'] = error
            else:
                download_list_sample['pipe:download_date'] = pd.to_datetime(os.path.getmtime(file_path), unit='s')
            download_list.append(download_list_sample)

        df_download_list = pd.DataFrame(download_list)
        df_download_list.to_pickle(self.file_name_output)

        # Clean up HTTP client
        close_default_client()

    def _download_with_retry(self, url: str, file_path: Path) -> str:
        """
        Download file with retry strategy and connection pooling.

        Args:
            url: URL to download from
            file_path: Path where to save the file

        Returns:
            Error string ('none' for success, error description otherwise)
        """
        def download_file():
            # Don't pass timeout here - it's already configured in http_client
            response = self.http_client.get(url)
            response.raise_for_status()

            # Write to file
            with open(file_path, 'wb') as f:
                f.write(response.content)
            return 'none'

        try:
            return self.retry_strategy.call_with_retry(download_file)
        except urllib.error.HTTPError as e:
            error_msg = f'HTTP{e.code}'
            if e.code == 404:
                error_msg = 'Missing(404)'
            logging.error(f"HTTP Error downloading {file_path.name}: {error_msg}")
            return error_msg
        except Exception as e:
            error_msg = f'Error: {type(e).__name__}'
            logging.error(f"Error downloading {file_path.name}: {error_msg}")
            return error_msg

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
