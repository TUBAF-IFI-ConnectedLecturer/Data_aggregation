import pandas as pd
from pathlib import Path
from tqdm import tqdm
import os
import hashlib
from github import Github
from github import Auth

from pipeline.taskfactory import Task, TaskWithInputFileMonitor, loggedExecution
import logging
from langdetect import detect_langs
from datetime import datetime, timedelta

# TODO
# Warum taucht dieser Datensatz in unserer Auswahl auf?
# https://raw.githubusercontent.com/djplaner/memex/master/docs/sense/Design/pedagogy-before-technology.md

# Custom logging filter to improve GitHub retry messages
class GithubRetryFilter(logging.Filter):
    """Filter to convert GitHub backoff seconds to human-readable format"""

    def filter(self, record):
        if hasattr(record, 'msg'):
            msg = str(record.msg)
            # Handle "Setting next backoff to X.Xs" messages
            if 'Setting next backoff to' in msg and record.name == 'github.GithubRetry':
                try:
                    # Extract seconds from message like "Setting next backoff to 1182.8533s"
                    parts = msg.split('Setting next backoff to ')
                    if len(parts) > 1:
                        seconds_str = parts[1].replace('s', '').strip()
                        seconds = float(seconds_str)

                        # Convert to human-readable format
                        if seconds < 60:
                            readable = f"{int(seconds)} Sekunden"
                        elif seconds < 3600:
                            minutes = int(seconds / 60)
                            remaining_secs = int(seconds % 60)
                            readable = f"{minutes} Minuten und {remaining_secs} Sekunden"
                        else:
                            hours = int(seconds / 3600)
                            minutes = int((seconds % 3600) / 60)
                            readable = f"{hours} Stunden und {minutes} Minuten"

                        # Calculate restart time
                        restart_time = datetime.now() + timedelta(seconds=seconds)
                        restart_str = restart_time.strftime('%H:%M:%S')

                        record.msg = f"GitHub API Rate-Limit erreicht. Warte {readable} (bis ca. {restart_str} Uhr)"
                except (ValueError, IndexError):
                    pass  # Keep original message if parsing fails

            # Suppress the "Restarting queries at ..." message as it's now included above
            elif 'Restarting queries at' in msg and record.name == 'github.GithubRetry':
                return False  # Don't show this message

        return True


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
    meta_data['liaIndi_import_statement'] = False
    meta_data['liaIndi_narrator_statement'] = False
    meta_data['liaIndi_version_statement'] = False
    meta_data['liaIndi_video_syntax'] = False
    meta_data['liaIndi_lia_in_h1'] = False

    if "liascript" in file.name.lower():
        meta_data['liaIndi_Lia_in_filename'] = True

    if "liascript" in content.lower():
        meta_data['liaIndi_liascript_in_content'] = True

    if "liaTemplates" in content:
        meta_data['liaIndi_liaTemplates_used'] = True

    # Check for LiaScript button/badge or course link
    if ("https://LiaScript.github.io/course/?" in content or
        "LiaScript/LiaScript/master/badges/course.svg" in content):
        meta_data['liaIndi_lia_button'] = True

    # Video syntax is very specific to LiaScript
    if "!?[" in content:
        meta_data['liaIndi_video_syntax'] = True

    # Check for "LiaScript" in main heading (first # line)
    lines = content.split('\n')
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('#') and not stripped.startswith('##'):
            if 'liascript' in stripped.lower():
                meta_data['liaIndi_lia_in_h1'] = True
            break

    if content.lstrip().startswith("<!--"):
        meta_data['liaIndi_comment_in_beginning'] = True

        # Check for LiaScript-specific header content (within first 2000 chars)
        header_section = content[:2000].lower()

        # import: statement is a very strong indicator
        if "import:" in header_section or "import :" in header_section:
            meta_data['liaIndi_import_statement'] = True

        # narrator: statement is a strong indicator
        if "narrator:" in header_section or "narrator :" in header_section:
            meta_data['liaIndi_narrator_statement'] = True

        # version: statement is a moderate indicator (combined with others)
        if "version:" in header_section or "version :" in header_section:
            meta_data['liaIndi_version_statement'] = True
        
    return meta_data, content


def extract_data(file, content):
    repo_data = {}
    # Create consistent pipe:ID using hash of download URL
    hash_object = hashlib.sha256(file.download_url.encode('utf-8'))
    repo_data['pipe:ID'] = hash_object.hexdigest()[:16]  # Use first 16 chars of hash
    repo_data['pipe:file_type'] = 'md'  # Mark as markdown file
    repo_data['id'] = hash(file.download_url)  # Keep old ID for backwards compatibility
    repo_data['repo_name'] = file.repository.name
    repo_data['repo_user'] = file.repository.owner.login
    repo_data['repo_url'] = file.repository.html_url
    repo_data['file_name'] = file.name
    repo_data['file_download_url'] = file.download_url
    repo_data['file_html_url'] = file.html_url
    # Add license information from repository
    repo = file.repository
    repo_data['repo_license_spdx'] = repo.license.spdx_id if repo.license else None
    repo_data['repo_license_name'] = repo.license.name if repo.license else None
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
                    # Use pipe:ID for consistent file naming
                    file_name = f"{repo_data['pipe:ID']}.md"
                    file_path = storage_folder / file_name
                    with open(file_path, "w", encoding='utf-8') as f:
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

        self.file_name =  Path(config_global['raw_data_folder']) / stage_param['repo_data_file_name_input']
        self.lia_files_name =  Path(config_global['raw_data_folder']) / stage_param['lia_files_name']

        # Optional: Repository indices to exclude from processing
        self.exclude_repo_indices = stage_param.get('exclude_repo_indices', [])
        
        # Get GitHub API token (should be loaded by run_pipeline.py)
        github_api_token = os.environ.get("GITHUB_API_KEY")
        if not github_api_token:
            raise ValueError("GITHUB_API_KEY environment variable is not set. Make sure run_pipeline.py loads the .env file.")
        
        auth = Auth.Token(github_api_token)
        self.github_handle = Github(auth=auth)
        logging.getLogger("urllib3").propagate = False

        # Add custom filter for GitHub retry messages
        github_logger = logging.getLogger("github.GithubRetry")
        github_logger.addFilter(GithubRetryFilter())

    @loggedExecution
    def execute_task(self):
        explore_potential_lia_files(
            github_handle=self.github_handle,
            data_folder=self.data_folder,
            repo_data_file_name=self.repo_data_file_name,
            course_data_file_name=self.lia_files_name,
            file_folder=self.file_folder,
            blacklist_indices=self.exclude_repo_indices
        )

