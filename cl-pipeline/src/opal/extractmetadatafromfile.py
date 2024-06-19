import pandas as pd
from pathlib import Path
from tqdm import tqdm
from pypdf import PdfReader
import os
import warnings
import logging
import pymupdf4llm

from pipeline.taskfactory import TaskWithInputFileMonitor

class ExtractMetaDataFromFile(TaskWithInputFileMonitor):
    def __init__(self, config_stage, config_global):
        super().__init__(config_stage, config_global)
        stage_param = config_stage['parameters']
        self.file_file_name_inputs =  Path(config_global['raw_data_folder']) / stage_param['file_file_name_input']
        self.file_file_name_output =  Path(config_global['raw_data_folder']) / stage_param['file_file_name_output']
        self.pdf_folder = Path(config_global['pdf_file_folder'])

    def execute_task(self):

        warnings.filterwarnings("ignore")
        local_logger = logging.getLogger("pypdf")
        local_logger.setLevel(logging.ERROR)

        df_files = pd.read_pickle(self.file_file_name_inputs)

        df_pdf_meta = pd.DataFrame(columns = ['ID', 'file_exist', 'meta_data_error',
                                              'meta_data_error_msg',
                                              'author', 'creator', 'subject', 'title',
                                              'creation_date', 'modification_date', 'pages'])
        df_pdfs = df_files[df_files['file_type'] == 'pdf']

        for index, row in tqdm(df_pdfs.iterrows(), total=df_pdfs.shape[0]):
            pdf_sample = {}
            pdf_sample["ID"] = row['ID']
            pdf_sample["file_exist"] = False
            pdf_sample["meta_data_error"] = False
            pdf_sample["meta_data_error_msg"] = ""
            pdf_path = self.pdf_folder / (row['ID'] + ".pdf")
            if (os.path.exists(pdf_path)):
                pdf_sample["file_exist"] = True
                try:
                    pdf = PdfReader(pdf_path)
                except Exception as e:
                    pdf_sample["meta_data_error_msg"] = e
                
                if pdf is not None:
                    try:
                        metadata = pdf.metadata
                    except Exception as e:
                        pdf_sample["meta_data_error_msg"] = e

                if metadata is not None:
                    if metadata.author is None:
                        pdf_sample["author"]  = metadata.author
                    else:
                        pdf_sample["author"]  = ""
                    pdf_sample["author"]  = metadata.author
                    pdf_sample["creator"] = metadata.creator
                    pdf_sample["subject"] = metadata.subject
                    try:
                        pdf_sample["modification_date"] = metadata.modification_date
                    except:
                        pdf_sample["modification_date"] = None
                    try:
                        pdf_sample["creation_date"] = metadata.creation_date
                    except:
                        pdf_sample["creation_date"] = None
                    try:
                        pdf_sample["pages"] = len(pdf.pages)
                    except:
                        pdf_sample["pages"] = None
                    if metadata.title is None:
                        pdf_sample["title"] = None
                    else:
                        pdf_sample["title"] = str(metadata.title) 
                else:
                    pdf_sample["meta_data_error_msg"] = "empty metadata"

            df_pdf_meta = pd.concat([df_pdf_meta, pd.DataFrame([pdf_sample])], 
                                     axis=0, ignore_index=True)

        df_pdf_meta.to_pickle(self.file_file_name_output)


class ExtractMarkdownFromPdf(TaskWithInputFileMonitor):
    def __init__(self, config_stage, config_global):
        super().__init__(config_stage, config_global)
        stage_param = config_stage['parameters']
        self.file_file_name_inputs =  Path(config_global['raw_data_folder']) / stage_param['file_file_name_input']
        self.file_file_name_output =  Path(config_global['raw_data_folder']) / stage_param['file_file_name_output']
        self.pdf_folder = Path(config_global['pdf_file_folder'])

    def execute_task(self):

        df_files = pd.read_pickle(self.file_file_name_inputs)

        df_pdf_meta = pd.DataFrame(columns = ['ID', 'valid_transformation', 'file_content'])
        df_pdfs = df_files[df_files['file_type'] == 'pdf']

        bar = tqdm(df_pdfs.iterrows(), total=df_pdfs.shape[0])
        for index, row in bar:
            pdf_sample = {}
            pdf_sample["ID"] = row['ID']

            pdf_path = self.pdf_folder / (row['ID'] + ".pdf")
            text_path = self.pdf_folder / (row['ID'] + ".txt")
            if (os.path.exists(pdf_path)):
                if not os.path.exists(text_path):
                    print("Converting " + row['ID'] + " ...")
                    try:
                        md_text = pymupdf4llm.to_markdown(str(pdf_path))
                        Path(text_path).write_bytes(md_text.encode())
                        pdf_sample["valid_transformation"] = True
                    except:
                        print("Error in conversion of " + row['ID'] + " to markdown.")
                        pdf_sample["valid_transformation"] = False

                    if pdf_sample["valid_transformation"]:
                        pdf_sample["file_content"] = md_text
                else:
                    print(row['ID'] + " already exists")
                    with open(text_path, 'r') as file:
                        pdf_sample["file_content"] = file.read()
            df_pdf_meta = pd.concat([df_pdf_meta, pd.DataFrame([pdf_sample])], 
                                     axis=0, ignore_index=True)
            bar.refresh()
            
        df_pdf_meta.to_pickle(self.file_file_name_output)