import pandas as pd
from pathlib import Path
import logging

from pipeline.taskfactory import TaskWithInputFileMonitor
from pipeline.taskfactory import loggedExecution


class MergeLiaScriptData(TaskWithInputFileMonitor):
    """
    Merge multiple LiaScript data sources into a consolidated DataFrame.

    Merges on pipe:ID:
    - LiaScript_files_validated.p (base file data with repo info)
    - LiaScript_metadata.p (header metadata: lia:* fields)
    - LiaScript_features.p (feature analysis: feature:* fields)
    - LiaScript_commits.p (commit history, optional)

    Output: A single consolidated pickle with all columns joined on pipe:ID.
    """

    def __init__(self, config_stage, config_global):
        super().__init__(config_stage, config_global)
        stage_param = config_stage['parameters']
        self.data_folder = Path(config_global['raw_data_folder'])

        # Input files
        self.lia_files_name = self.data_folder / stage_param['lia_files_name_input']
        self.lia_metadata_name = self.data_folder / stage_param['lia_metadata_name_input']
        self.lia_features_name = self.data_folder / stage_param['lia_features_name_input']
        self.lia_commits_name = self.data_folder / stage_param.get('lia_commits_name_input', 'LiaScript_commits.p')

        # Output file
        self.merged_output_name = self.data_folder / stage_param['merged_output_name']
        self.force_run = stage_param.get('force_run', False)

    @loggedExecution
    def execute_task(self):
        # Load base file data
        df_files = pd.read_pickle(self.lia_files_name)
        logging.info(f"Loaded {len(df_files)} files from {self.lia_files_name}")

        # Filter to validated files
        if 'pipe:is_valid_liascript' in df_files.columns:
            df_files = df_files[df_files['pipe:is_valid_liascript'] == True]
            logging.info(f"After validation filter: {len(df_files)} files")

        # Merge header metadata
        if self.lia_metadata_name.exists():
            df_meta = pd.read_pickle(self.lia_metadata_name)
            # Drop 'id' column from metadata to avoid conflicts
            if 'id' in df_meta.columns:
                df_meta = df_meta.drop(columns=['id'])
            # Drop columns that already exist in base (except pipe:ID)
            dup_cols = [c for c in df_meta.columns if c in df_files.columns and c != 'pipe:ID']
            df_meta = df_meta.drop(columns=dup_cols, errors='ignore')
            df_files = df_files.merge(df_meta, on='pipe:ID', how='left')
            logging.info(f"Merged {len(df_meta)} metadata records ({len(df_meta.columns)-1} fields)")
        else:
            logging.warning(f"Metadata file not found: {self.lia_metadata_name}")

        # Merge feature analysis
        if self.lia_features_name.exists():
            df_features = pd.read_pickle(self.lia_features_name)
            # Drop columns that already exist in base (except pipe:ID)
            dup_cols = [c for c in df_features.columns if c in df_files.columns and c != 'pipe:ID']
            df_features = df_features.drop(columns=dup_cols, errors='ignore')
            df_files = df_files.merge(df_features, on='pipe:ID', how='left')
            logging.info(f"Merged {len(df_features)} feature records ({len(df_features.columns)-1} fields)")
        else:
            logging.warning(f"Features file not found: {self.lia_features_name}")

        # Merge commit data (optional)
        if self.lia_commits_name.exists():
            df_commits = pd.read_pickle(self.lia_commits_name)
            commit_cols = ['pipe:ID', 'commit_count', 'author_count',
                           'first_commit', 'last_commit', 'contributors_list']
            available = [c for c in commit_cols if c in df_commits.columns]
            if 'pipe:ID' in available:
                # Drop columns that already exist
                dup_cols = [c for c in available if c in df_files.columns and c != 'pipe:ID']
                commit_data = df_commits[available].drop(columns=dup_cols, errors='ignore')
                df_files = df_files.merge(commit_data, on='pipe:ID', how='left')
                logging.info(f"Merged {len(df_commits)} commit records")
        else:
            logging.info(f"Commits file not found (optional): {self.lia_commits_name}")

        # Save consolidated output
        df_files.to_pickle(self.merged_output_name)
        # Also save as CSV for inspection
        df_files.to_csv(self.merged_output_name.with_suffix('.csv'), index=False)

        logging.info(f"Consolidated dataset: {len(df_files)} rows, {len(df_files.columns)} columns")
        logging.info(f"Output: {self.merged_output_name}")
