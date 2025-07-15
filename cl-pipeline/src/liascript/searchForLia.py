import pandas as pd
from pathlib import Path
from tqdm import tqdm
import os
import time
from github import Github
from github import Auth
from dotenv import load_dotenv

from pipeline.taskfactory import Task, loggedExecution
import logging

# Load environment variables from .env file in run directory only for direct execution
if __name__ == "__main__":
    env_path = Path(__file__).parent.parent.parent / 'run' / '.env'
    load_dotenv(env_path)

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

def search_repositories_by_year(github_handle, base_query, year, validity):
    """Search repositories for a specific year to avoid the 1000 result limit"""
    query = f"{base_query} created:{year}-01-01..{year}-12-31"
    print(f"  Searching year {year}: {query}")
    
    repository_list = []
    try:
        repositories = github_handle.search_repositories(query)
        print(f"    Found {repositories.totalCount} repositories for year {year}")
        
        for repo in tqdm(repositories, desc=f"Year {year}", leave=False):
            repository = get_repo_meta_data(repo)
            repository['searched_type'] = 'repository'
            repository['validity'] = validity
            repository['search_year'] = year
            repository_list.append(repository)
            
    except Exception as e:
        print(f"    Error searching year {year}: {e}")
    
    # Small delay to be nice to the API
    time.sleep(0.5)
    return repository_list

def search_repositories(github_handle, queries):
    # Step one: search for repositories
    repository_list = []
    current_year = 2025  # Current year
    start_year = 2017    # Start year for LiaScript-related repositories
    
    for query_param in queries:
        query = f"{query_param['keyword']} {query_param['focus']}"
        print(f"Searching for repositories: {query}")
        
        # First, try a general search to see if we hit the 1000 limit
        repositories = github_handle.search_repositories(query)
        total_count = repositories.totalCount
        print(f"Total repositories found: {total_count}")
        
        if total_count < 1000:
            # If less than 1000 results, we can get all results with one query
            print("Getting all results in one query...")
            for repo in tqdm(repositories, total=total_count, desc="Processing repositories"):
                repository = get_repo_meta_data(repo)
                repository['searched_type'] = 'repository'
                repository['validity'] = query_param['validity']
                repository_list.append(repository)
        else:
            # If we hit the 1000 limit, search year by year
            print(f"Hit 1000 result limit. Searching year by year from {start_year} to {current_year}...")
            
            # For repository searches, we need to modify the base query to remove the focus part
            # because we'll add the year constraint
            base_query = query_param['keyword'] + " " + query_param['focus']
            
            for year in range(start_year, current_year + 1):
                year_results = search_repositories_by_year(
                    github_handle, 
                    base_query, 
                    year, 
                    query_param['validity']
                )
                repository_list.extend(year_results)
        
        print(f"Completed search for: {query}")
        print(f"Total repositories collected so far: {len(repository_list)}")
      
    return repository_list

def search_code_with_pagination(github_handle, base_query, validity):
    """Search code using different strategies to get all results beyond 1000 limit"""
    repository_list = []
    
    # Strategy: Split by file size
    size_ranges = [
        "size:0..1000",
        "size:1001..5000", 
        "size:5001..20000",
        "size:>20000"
    ]
    
    for size_range in size_ranges:
        query = f"{base_query} {size_range}"
        print(f"  Searching with size filter: {query}")
        
        try:
            files = github_handle.search_code(query)
            print(f"    Found {files.totalCount} files with {size_range}")
            
            for code in tqdm(files, desc=f"Size {size_range}", leave=False):
                repository = get_repo_meta_data(code.repository)
                repository['searched_type'] = 'code'
                repository['validity'] = validity
                repository['size_range'] = size_range
                repository_list.append(repository)
                
        except Exception as e:
            print(f"    Error searching with {size_range}: {e}")
        
        time.sleep(0.5)  # Rate limiting
    
    return repository_list

def search_code(github_handle, queries):
    # Step two: search for repositories with the keyword "liascript" in files
    repository_list = []
    
    for query_param in queries:
        base_query = f"{query_param['keyword']}"
        print(f"Searching for code: {base_query}")
        
        # First, try a general search to see if we hit the 1000 limit
        try:
            files = github_handle.search_code(base_query)
            total_count = files.totalCount
            print(f"Total files found: {total_count}")
            
            if total_count < 1000:
                # If less than 1000 results, we can get all results with one query
                print("Getting all results in one query...")
                for code in tqdm(files, total=total_count, desc="Processing files"):
                    repository = get_repo_meta_data(code.repository)
                    repository['searched_type'] = 'code'
                    repository['validity'] = query_param['validity']
                    repository_list.append(repository)
            else:
                # If we hit the 1000 limit, use pagination strategies
                print(f"Hit 1000 result limit. Using alternative search strategies...")
                
                pagination_results = search_code_with_pagination(
                    github_handle, 
                    base_query, 
                    query_param['validity']
                )
                repository_list.extend(pagination_results)
                
        except Exception as e:
            print(f"Error in main search for {base_query}: {e}")
        
        print(f"Completed search for: {base_query}")
        print(f"Total repositories collected so far: {len(repository_list)}")

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

    ###########
    df = df[~df.internal]

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
        
        # Get GitHub API token (should be loaded by run_pipeline.py)
        github_api_token = os.environ.get("GITHUB_API_KEY")
        if not github_api_token:
            raise ValueError("GITHUB_API_KEY environment variable is not set. Make sure run_pipeline.py loads the .env file.")
        
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

if __name__ == "__main__":
    github_api_token = os.environ.get("GITHUB_API_KEY")
    if not github_api_token:
        print("Error: GITHUB_API_KEY environment variable is not set.")
        print("Please check your .env file in the run directory.")
        exit(1)
    
    print(f"Using GitHub token: {github_api_token[:10]}...")
    
    try:
        auth = Auth.Token(github_api_token)
        github_handle = Github(auth=auth)
        user = github_handle.get_user()
        print(f"Successfully authenticated as: {user.login}")
    except Exception as e:
        print(f"Authentication failed: {e}")
        print("Please check if your GitHub token is valid and has the necessary permissions.")
        exit(1)

    data_folder = Path().resolve().parent / "data"
    data_file = "lia_repos.p"
    search_lia_repos(
        github_handle=github_handle, 
        data_folder=data_folder,
        data_file=data_file
    )
