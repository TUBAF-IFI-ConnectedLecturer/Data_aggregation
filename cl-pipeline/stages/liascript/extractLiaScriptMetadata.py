import pandas as pd
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import logging

from pipeline.taskfactory import TaskWithInputFileMonitor
from pipeline.taskfactory import loggedExecution


class ExtractLiaScriptMetadata(TaskWithInputFileMonitor):

    def __init__(self, config_stage, config_global):
        super().__init__(config_stage, config_global)
        stage_param = config_stage['parameters']
        self.data_folder = Path(config_global['raw_data_folder'])
        self.file_folder = Path(config_global['file_folder'])
        
        self.lia_files_name =  Path(config_global['raw_data_folder']) / stage_param['lia_files_name_input']
        self.lia_metadata_name =  Path(config_global['raw_data_folder']) / stage_param['lia_metadata_name']
        

    @loggedExecution
    def execute_task(self):

        df_files = pd.read_pickle(Path(self.lia_files_name))

        # Filter only validated LiaScript files
        if 'pipe:is_valid_liascript' in df_files.columns:
            initial_count = len(df_files)
            df_files = df_files[df_files['pipe:is_valid_liascript'] == True]
            filtered_count = len(df_files)
            logging.info(f"Filtered files: {initial_count} -> {filtered_count} (removed {initial_count - filtered_count} invalid files)")

        if Path(self.lia_metadata_name).exists():
            df_meta = pd.read_pickle(Path(self.lia_metadata_name))
        else:
            df_meta = pd.DataFrame()

        for index, row in tqdm(df_files.iterrows(), total=df_files.shape[0]):
            # Check if metadata already exists for this file
            if df_meta.shape[0] > 0:
                if df_meta[df_meta['pipe:ID'] == row['pipe:ID']].shape[0] > 0:
                    continue

            # file starts with <!--
            if row['liaIndi_comment_in_beginning']:
                # Use pipe:ID for consistent file naming
                file_name = f"{row['pipe:ID']}.md"
                file_path = self.file_folder / file_name

                if not file_path.exists():
                    logging.warning(f"File not found: {file_path}")
                    continue

                with open(file_path, "r", encoding='utf-8') as f:
                    content = f.read()

                meta_data = self.extract_metadata(content)
                if meta_data is not None:
                    meta_data['pipe:ID'] = row['pipe:ID']
                    meta_data['id'] = row['id']  # Keep old ID for backwards compatibility
                    df_meta = pd.concat([df_meta, pd.DataFrame([meta_data])])
                    df_meta.to_pickle(Path(self.lia_metadata_name))

        df_meta.reset_index(drop=True, inplace=True)
        df_meta.to_pickle(Path(self.lia_metadata_name))


    def extract_metadata(self, content):
        # extract first text block starting with <!-- and ending with -->
        start = content.find("<!--")
        end = content.find("-->")
        if start == -1 or end == -1:
            return None
        intro = content[start:end]

        meta_data = {}

        meta_data_labels = ["title:", "comment:", "author:", "language:", "icon:", "mode:", "narrator:", "script:", "link:", "script:"]

        for line in intro.split("\n"):
            for label in meta_data_labels:
                if label in line:
                    meta = line.split(label)[1].strip()
                    if label in meta_data:
                        meta_data[label].append(meta)
                    else:
                        meta_data[label] = [meta]
        
        # Link is more complex - it covers multiple lines and can be in different formats
        # 1. starting with "link:" in each line
        # 2. starting with "link:" in first line and then empty line

        link_found=False
        for line in intro.split("\n"):
            # find first "link:"
            if not link_found and "import:" in line:
                link_found=True

            if link_found:
                if "import:" in line:
                    link_adress = line.split("import:")[1].strip()
                else:
                    link_adress = line.strip()

                if link_adress != "":
                    if "import:" in meta_data:
                        meta_data["import:"].append(link_adress)
                    else:
                        meta_data["import:"] = [link_adress]

            # find empty line after link
            if link_found and (line.strip() == "" or "script:" in line or "link:" in line or "title:" in line):
                break

        return meta_data
