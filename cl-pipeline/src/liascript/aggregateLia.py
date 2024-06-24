import pandas as pd
from pathlib import Path
from tqdm import tqdm
import os
from github import Github
from github import Auth

from pipeline.taskfactory import Task, loggedExecution
import logging
from langdetect import detect_langs

# TODO
# Warum taucht dieser Datensatz in unserer Auswahl auf?
# https://raw.githubusercontent.com/djplaner/memex/master/docs/sense/Design/pedagogy-before-technology.md


def extract_liafile_meta_data(file):
    if not file.name.endswith(".md"):
        print("-", end="")
        return None 
    if file.encoding != "base64":
        print( "b", end="")
        return None 
    content = file.decoded_content.decode()
    if (not content.lstrip().startswith('<!--')) and \
       (not "https://LiaScript.github.io/course/?" in content):
        print( "n", end="")
        return None 
    
    repo_data = {}
    repo_data['repo_name'] = file.repository.name
    repo_data['repo_user'] = file.repository.owner.login
    repo_data['repo_url'] = file.repository.html_url
    repo_data['file_name'] = file.name
    repo_data['file_download_url'] = file.download_url
    repo_data['file_html_url'] = file.html_url
    repo_data['file_content'] = content
    print("*", end="")
    return repo_data


def explore_potential_lia_files(github_handle, data_folder,
         repo_data_file_name, course_data_file_name):

    repo_data_file = Path(data_folder) / repo_data_file_name
    course_data_file = Path(data_folder) / course_data_file_name

    df_repos = pd.read_pickle(Path(repo_data_file))

    if Path(course_data_file).exists():
        df_courses = pd.read_pickle(Path(course_data_file))
    else:
        df_courses = pd.DataFrame()

    for i, row in df_repos.iterrows():
        print(f"{i}/{df_repos.shape[0]} - {row['user'] + '/' + row['name']} - ", end='')
        if df_courses.shape[0] > 0:
            if df_courses[(df_courses['repo_user'] == row['user']) &
                          (df_courses['repo_name'] == row['name'])].shape[0] > 0:
                print(" already aggregated")
                continue
        repo = github_handle.get_repo(row['user'] + '/' + row['name'])
        # aggregate all files in repo
        try:   
            contents = repo.get_contents("")
            files = []
            while contents:
                file_content = contents.pop(0)
                if file_content.type == "dir":
                    contents.extend(repo.get_contents(file_content.path))
                else:
                    files.append(file_content)
        except:
            print("no files")
            continue
        # identify liascript files and extract meta data
        if len(files):
            print(f"{len(files)} files ", end="")
            course_list = []
            for file in files:
                repo_data = extract_liafile_meta_data(file)
                if repo_data is not None:
                    course_list.append(repo_data)

            if len(course_list) > 0:     
                df_aux = pd.DataFrame(course_list)        
                df_courses = pd.concat([ df_courses, df_aux])
                df_courses.to_pickle(Path(course_data_file))
                df_courses.to_csv(Path(course_data_file).with_suffix('.csv'))
            print("")

    df_courses.reset_index(drop=True, inplace=True)
    df_courses.to_pickle(Path(course_data_file))
    
def identify_languages(s):
    langs = detect_langs(s['file_content']) 
    s['file_lang'] = langs[0].lang
    s['file_lang_prob'] = langs[0].prob
    return s

def extract_commit_statistics(s, github_handle):
    repo_signature = s['repo_user']+ "/" + s['repo_name']
    remaining = s['file_download_url'].split(repo_signature)[1]
    print(s['file_download_url'])
    if "/master/" in remaining:
        remaining=remaining.replace("/master/", "")
    if "/main/" in remaining:
        remaining=remaining.replace("/main/", "")
    print(remaining)

    repo = github_handle.get_repo(repo_signature)
    commits = repo.get_commits(path=remaining)

    feature_list = []
    for commit in commits:
        sample = {}
        try:
            sample['author'] = commit.author.login
        except:
            sample['author'] = "unknown"
        sample['date'] = commit.commit.author.date
        feature_list.append(sample)

    df_features = pd.DataFrame(feature_list)
    if df_features.shape[0] > 0:
        df_features['date']=pd.to_datetime(df_features['date'])
        s['contributors_list'] = df_features.author.to_list()
        s['author_count'] = len(df_features.author.unique())
        s['commit_count'] = df_features.shape[0]
        s['first_commit'] = df_features.date.min()
        s['last_commit']  = df_features.date.max()
        s['commit_hist_extracted'] = True
    else:
        s['contributors_list'] = []
        s['author_count'] = 0
        s['commit_count'] = 0

    print(s['commit_count'], s['author_count'])
    return s


class AggregateLiaScriptFiles(Task):
    def __init__(self, config_stage, config_global):
        super().__init__(config_stage, config_global)
        stage_param = config_stage['parameters']
        self.data_folder = Path(config_global['raw_data_folder'])
        self.repo_data_file_name = stage_param['repo_data_file_name_input']
        
        self.file_file_name =  Path(config_global['raw_data_folder']) / stage_param['repo_data_file_name_input']
        self.lia_files_name =  Path(config_global['raw_data_folder']) / stage_param['lia_files_name']
        
        github_api_token =os.environ["GITHUB_API_KEY"]
        auth = Auth.Token(github_api_token)
        self.github_handle = Github(auth=auth)
        logging.getLogger("urllib3").propagate = False

    @loggedExecution
    def execute_task(self):
        explore_potential_lia_files(
            github_handle=self.github_handle,
            data_folder=self.data_folder,
            repo_data_file_name=self.repo_data_file_name,
            course_data_file_name=self.lia_files_name
        )

