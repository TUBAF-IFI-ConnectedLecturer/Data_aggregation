import pandas as pd
from pathlib import Path
from tqdm import tqdm
import os
from github import Github
from github import Auth

from pipeline.taskfactory import loggedExecution
from pipeline.taskfactory import TaskWithInputFileMonitor
import logging
from datetime import datetime, timedelta

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

def explore_potential_lia_commits(github_handle, data_folder,
         course_data_file_name, commits_data_file_name):

    course_data_file = Path(data_folder) / course_data_file_name
    commit_data_file = Path(data_folder) / commits_data_file_name

    df_courses = pd.read_pickle(Path(course_data_file))

    if Path(commit_data_file).exists():
        df_commits = pd.read_pickle(Path(commit_data_file))
    else:
        df_commits = pd.DataFrame()

    for i, row in df_courses.iterrows():
        print(f"{i}/{df_courses.shape[0]} - {row['file_download_url']} - ", end='')
        if df_commits.shape[0] > 0:
            if df_commits[(df_commits['file_download_url'] == row['file_download_url'])].shape[0] > 0:
                print(" already aggregated")
                continue

        row = extract_commit_statistics(row, github_handle)
        df_commits = pd.concat([df_commits, pd.DataFrame([row])])
        df_commits.to_pickle(Path(commit_data_file))
        print(f" Commits={row['commit_count']} Authors={row['author_count']}")

    df_commits.reset_index(drop=True, inplace=True)
    df_commits.to_pickle(Path(commit_data_file))
    

def extract_commit_statistics(s, github_handle):
    repo_signature = s['repo_user']+ "/" + s['repo_name']
    remaining = s['file_download_url'].split(repo_signature)[1]
    if "/master/" in remaining:
        remaining=remaining.replace("/master/", "")
    if "/main/" in remaining:
        remaining=remaining.replace("/main/", "")

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
    return s


class AggregateLiaScriptCommits(TaskWithInputFileMonitor):
    def __init__(self, config_stage, config_global):
        super().__init__(config_stage, config_global)
        stage_param = config_stage['parameters']
        self.data_folder = Path(config_global['raw_data_folder'])
        
        self.lia_files_name =  Path(config_global['raw_data_folder']) / stage_param['lia_files_name_input']
        self.lia_commits_name =  Path(config_global['raw_data_folder']) / stage_param['lia_commits_name']
        
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
        explore_potential_lia_commits(
            github_handle=self.github_handle,
            data_folder=self.data_folder,
            course_data_file_name=self.lia_files_name,
            commits_data_file_name=self.lia_commits_name
        )

