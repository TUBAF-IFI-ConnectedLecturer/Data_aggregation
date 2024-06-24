import pandas as pd
from pathlib import Path
from tqdm import tqdm
import os
from github import Github
from github import Auth

from pipeline.taskfactory import Task, loggedExecution
import logging

def get_repo_meta_data(repo):
    repository = {
       'user': repo.owner.login,
       'name': repo.name,
       'repo_url': repo.html_url,
       'created_at': repo.created_at,
       'updated_at': repo.updated_at,
       'stars': repo.stargazers_count,
       'forks': repo.forks_count,
       'is_fork': repo.fork,
       'watchers': repo.watchers_count,
       'contributors_per_repo': repo.get_contributors().totalCount,
    }
    return repository

def search_repositories(github_handle, queries):
    # Step one: search for repositories

    repository_list = []
    for query_param in queries:

      query = f"{query_param['keyword']} {query_param['focus']}"
      print(query)
      repositories = github_handle.search_repositories(query)

      for repo in tqdm(repositories, total=repositories.totalCount):
          repository = get_repo_meta_data(repo)
          repository['searched_type'] = 'repository'
          repository['validity'] = query_param['validity']
          repository_list.append(repository)
      
    return repository_list

def search_code(github_handle, queries):
    # Step two: search for repositories with the keyword "liascript" in files
    repository_list = []
    for query_param in queries:
        query = f"{query_param['keyword']}"
        print(query)
        files = github_handle.search_code(query)
        for code in tqdm(files, total=files.totalCount):
            repository = get_repo_meta_data(code.repository)
            repository['searched_type'] = 'code'
            repository['validity'] = query_param['validity']
            repository_list.append(repository)

    return repository_list

def search_lia_repos(github_handle, data_folder, data_file):
    repository_list = []
    search_queries = [
        {"keyword": '"liascript"', "focus": " in:topic", "validity": 1},
        {"keyword": '"liascript-course"', "focus": " in:topic", "validity": 1},
        {"keyword": '"liascript"', "focus": " in:description", "validity": 1},
        {"keyword": '" liascript "', "focus": " in:readme", "validity": 1},
        {"keyword": 'liascript', "focus": " in:path,file", "validity": 2},
    ]
    print("Searching for repositories")
    repos = search_repositories(github_handle, search_queries)
    repository_list.extend(repos)

    search_queries = [
        {"keyword": '" liascript "', "validity": 1},
        {"keyword": '"https://github.com/LiaTemplates/"', "validity": 1},
        {"keyword": '"liascript"', "validity": 2},
    ]
    print("Searching for files")
    code = search_code(github_handle, search_queries)
    repository_list.extend(code)

    df = pd.DataFrame(repository_list)
    df_droped = df.sort_values('validity', ascending=True).drop_duplicates(subset=['name','user'], keep='first').reset_index(drop=True)

    print(f"{df.shape} -> {df_droped.shape}")
    df = df_droped

    black_list = ['LiaScript', 'TUBAF-IfI-LiaScript','LiaPlayground', 'SebastianZug', 
                  'andre-dietrich', 'LiaBooks', 'LiaTemplates', 'TUBAF-IUZ-LiaScript',
                  'markjjacob', 'HueblerPatricia']

    df['internal'] = False
    df.loc[df.user.isin(black_list), 'internal'] = True

    data_file = Path().resolve().parent / Path(data_folder) / data_file
    df.to_pickle(Path(str(data_file)))

    print(f"Found {df.shape[0]} relevant repositories referencing 'liascript' in different ways")


class CrawlGithubForLiaScript(Task):
    def __init__(self, config_stage, config_global):
        super().__init__(config_stage, config_global)
        stage_param = config_stage['parameters']
        self.data_folder = Path(config_global['raw_data_folder'])
        self.repo_data_file_name = stage_param['repo_data_file_name']
        self.repo_file_name =  Path(config_global['raw_data_folder']) / stage_param['repo_data_file_name']
        github_api_token =os.environ["GITHUB_API_KEY"]
        auth = Auth.Token(github_api_token)
        self.github_handle = Github(auth=auth)
        logging.getLogger("urllib3").propagate = False

    @loggedExecution
    def execute_task(self):
        run = False
        if self.repo_file_name.is_file():
            logging.info(f"OPAL data set already downloaded.")
            if "force_run" in self.parameters:
                if self.parameters['force_run'] == True:
                    logging.info(f"Forcing run of task")
                    run = True
        else:
            run = True
    
        if run:
            search_lia_repos(
                github_handle=self.github_handle, 
                data_folder=self.data_folder,
                data_file=self.repo_data_file_name
            )

