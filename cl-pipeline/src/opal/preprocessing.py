import pandas as pd
from pathlib import Path

from pipeline.taskfactory import TaskWithInputFileMonitor

# extraktion der unique ids für die einzelnen Dokumente
def extractUniqueID(url):
    return url.removeprefix("https://bildungsportal.sachsen.de/opal/oer/")

# identifikation der pdf Dokumente
def getLastValue(aList):
    if  isinstance(aList, list):
        return aList[-1]
    else:
        return "unknown"

class Preprocessing(TaskWithInputFileMonitor):
    def __init__(self, config_stage, config_global):
        super().__init__(config_stage, config_global)
        stage_param = config_stage['parameters']
        self.file_file_name_inputs =  Path(config_global['raw_data_folder']) / stage_param['file_file_name_input']
        self.file_file_name_output =  Path(config_global['raw_data_folder']) / stage_param['file_file_name_output']

    def execute_task(self):

        df_files = pd.read_pickle(self.file_file_name_inputs)
        # Extract unique ID
        df_files['pipe:ID'] = df_files['opal:oer_permalink'].apply(extractUniqueID)
       
        # Extract file extension
        df_files['pipe:file_type'] = df_files['opal:filename'].str.split('.').apply(getLastValue).str.lower()

        
        #df_files = df_files[['ID', 'file_type', 'known_creator']]

        df_files.to_pickle(self.file_file_name_output)