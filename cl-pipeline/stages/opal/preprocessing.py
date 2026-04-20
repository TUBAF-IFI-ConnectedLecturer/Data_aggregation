import pandas as pd
from pathlib import Path
import logging

from pipeline.taskfactory import TaskWithInputFileMonitor

# extraktion der unique ids für die einzelnen Dokumente
def extractUniqueID(url):
    return url.removeprefix("https://bildungsportal.sachsen.de/opal/oer/")

# Supported file types organized by category
SUPPORTED_FILE_TYPES = {
    # Documents - fully supported
    'documents': {'pdf', 'docx', 'doc', 'docm', 'odt', 'rtf', 'dotx'},
    # Spreadsheets - fully supported
    'spreadsheets': {'xlsx', 'xls', 'xlsm', 'ods', 'csv'},
    # Presentations - fully supported
    'presentations': {'pptx', 'ppt', 'pptm', 'odp', 'potx', 'ppsx', 'ott'},
    # Plain text and markup
    'text': {'md', 'txt', 'html', 'htm', 'xml', 'json', 'yaml', 'yml', 'srt', 'vtt'},
    # Code and notebooks
    'code': {'py', 'java', 'cpp', 'c', 'js', 'jsx', 'ts', 'tsx', 'cs', 'r', 'sql', 'sh', 'bash', 'html', 'css', 'ipynb', 'rmd', 'hs', 'pl', 'ino'},
    # Images
    'images': {'jpg', 'jpeg', 'png', 'gif', 'svg', 'bmp', 'tif', 'tiff', 'webp', 'ico', 'heic'},
    # Media
    'media': {'mp3', 'mp4', 'wav', 'ogg', 'm4a', 'flac', 'aac', 'mov', 'avi', 'mkv', 'webm', 'fls'},
    # Educational/Technical
    'technical': {'h5p', 'ggb', 'svg', 'eps', 'tex', 'rds', 'prj', 'dwg', 'stl', 'nb', 'odg'},
    # Archives (often contain course materials)
    'archives': {'zip', 'rar', '7z', 'tar', 'gz', 'epub'},
}

# Flatten to single set for quick lookup
ALL_SUPPORTED_TYPES = set()
for category_types in SUPPORTED_FILE_TYPES.values():
    ALL_SUPPORTED_TYPES.update(category_types)

# Explicitly exclude junk file types
EXCLUDED_TYPES = {
    'exe', 'dll', 'bin', 'so', 'o', 'a',  # Executables/Libraries
    'pyc', 'pyo', 'class', 'o',  # Compiled
    'db', 'sqlite', 'mdb', 'accdb',  # Databases (binary)
    'vhd', 'vmdk', 'iso',  # Virtual machines
    'mov', 'flv',  # Proprietary video (often corrupted)
}

def extract_file_type(filename):
    """
    Extract and validate file type from filename.

    Returns:
        str: Lowercase file extension if valid and supported, None otherwise
    """
    if not isinstance(filename, str) or not filename.strip():
        return None

    # Get extension after last dot
    parts = filename.rsplit('.', 1)
    if len(parts) != 2:
        return None  # No extension

    ext = parts[1].lower().strip()

    # Reject obviously invalid extensions
    if not ext or len(ext) > 10 or ' ' in ext:
        return None

    # Check against supported types
    if ext in ALL_SUPPORTED_TYPES and ext not in EXCLUDED_TYPES:
        return ext
    else:
        return None

class Preprocessing(TaskWithInputFileMonitor):
    def __init__(self, config_stage, config_global):
        super().__init__(config_stage, config_global)
        stage_param = config_stage['parameters']
        self.file_name_inputs =  Path(config_global['raw_data_folder']) / stage_param['file_name_input']
        self.file_name_output =  Path(config_global['raw_data_folder']) / stage_param['file_name_output']

    def execute_task(self):

        df_files = pd.read_pickle(self.file_name_inputs)
        initial_count = len(df_files)

        # Extract unique ID
        df_files['pipe:ID'] = df_files['opal:oer_permalink'].apply(extractUniqueID)

        # Extract and validate file extension
        df_files['pipe:file_type'] = df_files['opal:filename'].apply(extract_file_type)

        # Remove rows with invalid/unsupported file types
        df_files = df_files[df_files['pipe:file_type'].notna()]

        logging.info(f"File type filtering: {initial_count} → {len(df_files)} ({100*len(df_files)/initial_count:.1f}%)")
        logging.debug(f"Supported file types: {sorted(ALL_SUPPORTED_TYPES)}")

        # Standardize naming convention: rename opal:author to opal:author_raw
        if 'opal:author' in df_files.columns:
            df_files['opal:author_raw'] = df_files['opal:author']
            # Keep old column for backward compatibility
            # df_files = df_files.drop(columns=['opal:author'])

        df_files.to_pickle(self.file_name_output)