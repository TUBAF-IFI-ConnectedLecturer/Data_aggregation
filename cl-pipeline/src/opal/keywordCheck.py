import pandas as pd
from pathlib import Path
from tqdm import tqdm
import requests
import logging
import json
import spacy

from pipeline.taskfactory import TaskWithInputFileMonitor

def get_lobid_response(word):
    params = {
        'q': word,
        'filter': 'type:SubjectHeading OR type:SubjectHeadingSensoStricto',
        'format': 'json',
    }
    response = requests.get('https://lobid.org/gnd/search', params=params)
    return response

def receive_lobid_keywords(word):
    response = get_lobid_response(word)
    try:
        data = response.json()
    except:
        return None

    if data['member']:
        lobid = {}
        lobid['totalItems'] = data['totalItems']
        result = data['member'][0]
        if 'sameAs' in result.keys():
            lobid['sameAs_link'] = result['sameAs'][0]['id']      
        lobid['gnd_link'] = result['id']
        dewey2 = 'relatedDdcWithDegreeOfDeterminacy2'
        if dewey2 in result.keys():
            lobid['ddc_D2'] = result[dewey2][0]['id']
        dewey3 = 'relatedDdcWithDegreeOfDeterminacy3'
        if dewey3 in result.keys():
            lobid['ddc_D3'] = result[dewey3][0]['id'] 
        lobid['perferedNames'] = str([entry['preferredName'] for entry in data['member'] if entry['preferredName'] != "None"])
    else:
        return None
    return lobid


class GNDKeywordCheck(TaskWithInputFileMonitor):
    def __init__(self, config_stage, config_global):
        super().__init__(config_stage, config_global)
        stage_param = config_stage['parameters']
        self.file_file_name_inputs =  Path(config_global['raw_data_folder']) / stage_param['file_file_name_input']
        self.file_file_name_output =  Path(config_global['raw_data_folder']) / stage_param['file_file_name_output']
        self.file_folder = Path(config_global['file_folder'])
        self.processed_data_folder = config_global['processed_data_folder']
        self.keyword_list_path = Path(config_global['processed_data_folder']) / "keyword_list"

    def execute_task(self):

        logging.getLogger("urllib3.connectionpool").setLevel(logging.CRITICAL)

        df_metadata = pd.read_pickle(self.file_file_name_inputs)
        keyword_columns = [col for col in df_metadata.columns if 'keyword' in col]

        if Path(self.file_file_name_output).exists():
            df_checkedKeywords = pd.read_pickle(self.file_file_name_output)
        else:
            df_checkedKeywords = pd.DataFrame()

        if Path(self.keyword_list_path.with_suffix('.p')).exists():
            df_keywords = pd.read_pickle(self.keyword_list_path.with_suffix('.p'))
        else:
            df_keywords = pd.DataFrame()

        nlp = spacy.load("de_core_news_md")

        for _, row in tqdm(df_metadata.iterrows(), total=df_metadata.shape[0]):

            #Check if gnd check is already done
            if df_checkedKeywords.shape[0] > 0:
                if df_checkedKeywords[df_checkedKeywords['pipe:ID'] == row['pipe:ID']].shape[0] > 0:
                    continue
            
            all_keywords_list = []
            for keyword_column in keyword_columns:
                keywords = row[keyword_column].split(",")
                if keywords:
                    all_keywords_list.extend(keywords)
                
            if all_keywords_list == []:
                continue

            all_keywords_list = [keyword.strip() for keyword in all_keywords_list]

            keyword_list = []
            for keyword in all_keywords_list:
                keyword_sample = {}
                keyword_sample['raw_keyword'] = keyword

                if " " not in keyword:
                    doc = nlp(keyword)
                    keyword_sample['lemma'] = "".join([token.lemma_ for token in doc])
                    lobid = receive_lobid_keywords(keyword_sample['lemma'])
                else:
                    keyword_sample['lemma'] = ""
                    lobid = receive_lobid_keywords(keyword)
                
                if lobid:
                    keyword_sample['is_gnd'] = True
                    lobbid_keys = ['gnd_link', 'sameAs_link', 'ddc_D2', 'ddc_D3', 'perferedNames', 'totalItems']
                    for key in lobbid_keys:
                        if key in lobid.keys():
                            keyword_sample[key] = lobid[key]
                else:
                    keyword_sample['is_gnd'] = False
                keyword_sample['count'] = 0
                if df_keywords.shape[0] == 0:
                    df_keywords = pd.DataFrame(keyword_sample, index=[0])
                else:
                    if keyword not in df_keywords['raw_keyword'].values:
                        new_index = df_keywords.index.max() + 1
                        new_row = pd.DataFrame([keyword_sample], index=[new_index])
                        df_keywords = pd.concat([df_keywords, new_row])

                df_keywords.loc[df_keywords['raw_keyword'] == keyword, 'count'] += 1
                keyword_id = (df_keywords['raw_keyword'] == keyword).idxmax()
                
                keyword_list_sample = {}
                keyword_list_sample['pipe:ID'] = row['pipe:ID']
                keyword_list_sample['pipe:file_type'] = row['pipe:file_type']
                keyword_list_sample['keyword_id'] = keyword_id
                keyword_list.append(keyword_list_sample)

            df_aux = pd.DataFrame(keyword_list)        
            df_checkedKeywords = pd.concat([df_checkedKeywords, df_aux])
            df_checkedKeywords.to_pickle(self.file_file_name_output)
            df_keywords.to_csv(self.keyword_list_path.with_suffix('.csv'))  
            df_keywords.to_pickle(self.keyword_list_path.with_suffix('.p'))
            
        df_checkedKeywords.reset_index(drop=True, inplace=True)
        df_checkedKeywords.to_pickle(self.file_file_name_output)
