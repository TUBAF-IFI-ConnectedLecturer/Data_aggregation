import pandas as pd
from pathlib import Path
from tqdm import tqdm
import os
from github import Github
from github import Auth

from pipeline.taskfactory import Task, TaskWithInputFileMonitor, loggedExecution
import logging
from langdetect import detect_langs

# TODO
# Warum taucht dieser Datensatz in unserer Auswahl auf?
# https://raw.githubusercontent.com/djplaner/memex/master/docs/sense/Design/pedagogy-before-technology.md


def extract_liafile_meta_data(file):

    if not file.name.endswith(".md"):
        print("-", end="")
        return None, None 
    
    if file.encoding != "base64":
        print( "b", end="")
        return None, None
    
    try:
        # Versuche UTF-8 Dekodierung
        content = file.decoded_content.decode('utf-8')
    except UnicodeDecodeError:
        try:
            # Fallback: Versuche latin-1 (kann alle Byte-Werte dekodieren)
            content = file.decoded_content.decode('latin-1')
        except UnicodeDecodeError:
            try:
                # Fallback: UTF-8 mit Fehlerbehandlung
                content = file.decoded_content.decode('utf-8', errors='ignore')
                print("i", end="")  # Markiere ignorierte Zeichen
            except:
                print("x", end="")  # Markiere nicht dekodierbare Datei
                return None, None
    
    meta_data = extract_data(file, content)
    meta_data['liaIndi_Lia_in_filename'] = False
    meta_data['liaIndi_liaTemplates_used'] = False
    meta_data['liaIndi_liascript_in_content'] = False
    meta_data['liaIndi_lia_button'] = False
    meta_data['liaIndi_comment_in_beginning'] = False

    if "liascript" in file.name.lower():
        meta_data['liaIndi_Lia_in_filename'] = True

    if "liascript" in content.lower():
        meta_data['liaIndi_liascript_in_content'] = True

    if "liaTemplates" in content:
        meta_data['liaIndi_liaTemplates_used'] = True

    if "https://LiaScript.github.io/course/?" in content:
        meta_data['liaIndi_lia_button'] = True

    if content.lstrip().startswith("<!--"):
        meta_data['liaIndi_comment_in_beginning'] = True
        
    return meta_data, content


def extract_data(file, content):
    repo_data = {}
    repo_data['id'] = hash(file.download_url)
    repo_data['repo_name'] = file.repository.name
    repo_data['repo_user'] = file.repository.owner.login
    repo_data['repo_url'] = file.repository.html_url
    repo_data['file_name'] = file.name
    repo_data['file_download_url'] = file.download_url
    repo_data['file_html_url'] = file.html_url
    print("*", end="")
    return repo_data


def explore_potential_lia_files(github_handle, data_folder,
         repo_data_file_name, course_data_file_name, file_folder, blacklist_indices=None):

    repo_data_file = Path(data_folder) / repo_data_file_name
    course_data_file = Path(data_folder) / course_data_file_name
    storage_folder = Path(file_folder)

    df_repos = pd.read_pickle(Path(repo_data_file))
    
    # Blacklist für Repository-Indizes
    if blacklist_indices is None:
        blacklist_indices = []

    if Path(course_data_file).exists():
        df_courses = pd.read_pickle(Path(course_data_file))
    else:
        df_courses = pd.DataFrame()

    print()

    for i, row in df_repos.iterrows():
        #print(f"{i}/{df_repos.shape[0]} - {row['user'] + '/' + row['name']} - ", end='')
        print(f"{i}/{df_repos.index.max()} - {row['user'] + '/' + row['name']} - ", end='')
        
        # Prüfe Blacklist
        if i in blacklist_indices:
            print("skipped (blacklisted)")
            continue
            
        if df_courses.shape[0] > 0:
            if df_courses[(df_courses['repo_user'] == row['user']) &
                          (df_courses['repo_name'] == row['name'])].shape[0] > 0:
                print(" already aggregated")
                continue
        repo = github_handle.get_repo(row['user'] + '/' + row['name'])
        # aggregate all files in repo recursively
        # https://pygithub.readthedocs.io/en/latest/examples/Repository.html#get-all-of-the-contents-of-the-repository-recursively
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
                repo_data, content = extract_liafile_meta_data(file)
                if repo_data is not None:
                    file_name = f"{repo_data['repo_user']}_{repo_data['repo_name']}_{repo_data['file_name']}"
                    file_path = storage_folder / file_name
                    with open(file_path, "w") as f:
                        f.write(content)
                    course_list.append(repo_data)

            if len(course_list) > 0:     
                df_aux = pd.DataFrame(course_list)        
                df_courses = pd.concat([ df_courses, df_aux])
                df_courses.to_pickle(Path(course_data_file))
                df_courses.to_csv(Path(course_data_file).with_suffix('.csv'))
            print("")

    df_courses.reset_index(drop=True, inplace=True)
    df_courses.to_pickle(Path(course_data_file))
    

class AggregateLiaScriptFiles(TaskWithInputFileMonitor):
    def __init__(self, config_stage, config_global):
        super().__init__(config_stage, config_global)
        stage_param = config_stage['parameters']
        self.data_folder = Path(config_global['raw_data_folder'])
        self.repo_data_file_name = stage_param['repo_data_file_name_input']
        self.file_folder = Path(config_global['file_folder'])
        
        self.file_file_name =  Path(config_global['raw_data_folder']) / stage_param['repo_data_file_name_input']
        self.lia_files_name =  Path(config_global['raw_data_folder']) / stage_param['lia_files_name']
        
        # Get GitHub API token (should be loaded by run_pipeline.py)
        github_api_token = os.environ.get("GITHUB_API_KEY")
        if not github_api_token:
            raise ValueError("GITHUB_API_KEY environment variable is not set. Make sure run_pipeline.py loads the .env file.")
        
        auth = Auth.Token(github_api_token)
        self.github_handle = Github(auth=auth)
        logging.getLogger("urllib3").propagate = False

    @loggedExecution
    def execute_task(self):
        blacklist = [41, 580, 597, 598, 600, 607, 617, 640, 641, 647, 649, 683, 689, 712, 724, 725, 726, 727, 728, 729, 732, 733, 734, 741, 750, 809,
                     763, 764, 765, 776, 799, 801, 816, 825, 842, 843, 850, 869, 880, 881, 897, 898, 900, 932, 933, 934]
        explore_potential_lia_files(
            github_handle=self.github_handle,
            data_folder=self.data_folder,
            repo_data_file_name=self.repo_data_file_name,
            course_data_file_name=self.lia_files_name,
            file_folder=self.file_folder,
            blacklist_indices=blacklist
        )

