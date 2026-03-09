"""
Pipeline stage: Collect GitHub user profile data for all repository owners.

Fetches location, company, bio, email and other profile metadata from GitHub
for each unique user in the repositories dataset. This data can be used for
geographical analysis and author affiliation mapping.

Output columns:
    login, name, company, location, bio, email, blog,
    twitter_username, followers, following, public_repos,
    created_at, updated_at, profile_url, type
"""

import pandas as pd
from pathlib import Path
from tqdm import tqdm
import os
import time
from github import Github, GithubException
from github import Auth

from pipeline.taskfactory import TaskWithInputFileMonitor
from pipeline.taskfactory import loggedExecution
import logging
from datetime import datetime, timedelta


# Custom logging filter to improve GitHub retry messages
class GithubRetryFilter(logging.Filter):
    """Filter to convert GitHub backoff seconds to human-readable format"""

    def filter(self, record):
        if hasattr(record, 'msg'):
            msg = str(record.msg)
            if 'Setting next backoff to' in msg and record.name == 'github.GithubRetry':
                try:
                    parts = msg.split('Setting next backoff to ')
                    if len(parts) > 1:
                        seconds_str = parts[1].replace('s', '').strip()
                        seconds = float(seconds_str)

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

                        restart_time = datetime.now() + timedelta(seconds=seconds)
                        restart_str = restart_time.strftime('%H:%M:%S')

                        record.msg = f"GitHub API Rate-Limit erreicht. Warte {readable} (bis ca. {restart_str} Uhr)"
                except (ValueError, IndexError):
                    pass

            elif 'Restarting queries at' in msg and record.name == 'github.GithubRetry':
                return False

        return True


def get_user_profile(github_handle, login):
    """Fetch profile metadata for a single GitHub user/org."""
    try:
        user = github_handle.get_user(login)
        return {
            'login': user.login,
            'name': user.name,
            'company': user.company,
            'location': user.location,
            'bio': user.bio,
            'email': user.email,
            'blog': user.blog,
            'twitter_username': user.twitter_username,
            'followers': user.followers,
            'following': user.following,
            'public_repos': user.public_repos,
            'created_at': user.created_at,
            'updated_at': user.updated_at,
            'profile_url': user.html_url,
            'type': user.type,  # "User" or "Organization"
        }
    except GithubException as e:
        logging.warning(f"Could not fetch profile for '{login}': {e.data.get('message', str(e))}")
        return {
            'login': login,
            'name': None,
            'company': None,
            'location': None,
            'bio': None,
            'email': None,
            'blog': None,
            'twitter_username': None,
            'followers': None,
            'following': None,
            'public_repos': None,
            'created_at': None,
            'updated_at': None,
            'profile_url': f"https://github.com/{login}",
            'type': None,
        }


def collect_user_profiles(github_handle, data_folder,
                          repo_data_file_name, user_profiles_file_name):
    """Collect GitHub profile data for all unique repo owners."""

    repo_data_file = Path(data_folder) / repo_data_file_name
    user_profiles_file = Path(data_folder) / user_profiles_file_name

    df_repos = pd.read_pickle(repo_data_file)
    all_users = sorted(df_repos['user'].unique())
    logging.info(f"Found {len(all_users)} unique users in {len(df_repos)} repositories")

    # Load existing profiles to allow incremental updates
    if user_profiles_file.exists():
        df_existing = pd.read_pickle(user_profiles_file)
        already_fetched = set(df_existing['login'])
        logging.info(f"Loaded {len(df_existing)} existing profiles, checking for new users")
    else:
        df_existing = pd.DataFrame()
        already_fetched = set()

    users_to_fetch = [u for u in all_users if u not in already_fetched]
    logging.info(f"Users to fetch: {len(users_to_fetch)} (skipping {len(already_fetched)} already collected)")

    if not users_to_fetch:
        logging.info("All user profiles already collected. Nothing to do.")
        return

    profiles = []
    for login in tqdm(users_to_fetch, desc="Fetching GitHub profiles"):
        profile = get_user_profile(github_handle, login)
        profiles.append(profile)

    df_new = pd.DataFrame(profiles)
    df_all = pd.concat([df_existing, df_new], ignore_index=True)

    # Save pickle and CSV
    df_all.to_pickle(user_profiles_file)
    csv_file = user_profiles_file.with_suffix('.csv')
    df_all.to_csv(csv_file, index=False)

    # Summary statistics
    has_location = df_all['location'].notna().sum()
    has_company = df_all['company'].notna().sum()
    has_bio = df_all['bio'].notna().sum()
    has_any = (df_all['location'].notna() | df_all['company'].notna() | df_all['bio'].notna()).sum()

    logging.info(f"Saved {len(df_all)} user profiles to {user_profiles_file}")
    logging.info(f"  Location: {has_location}/{len(df_all)} ({has_location*100//len(df_all)}%)")
    logging.info(f"  Company:  {has_company}/{len(df_all)} ({has_company*100//len(df_all)}%)")
    logging.info(f"  Bio:      {has_bio}/{len(df_all)} ({has_bio*100//len(df_all)}%)")
    logging.info(f"  Any info: {has_any}/{len(df_all)} ({has_any*100//len(df_all)}%)")

    print(f"\nDone! {len(df_new)} new profiles fetched, {len(df_all)} total.")
    print(f"  Location: {has_location}/{len(df_all)} ({has_location*100//len(df_all)}%)")
    print(f"  Company:  {has_company}/{len(df_all)} ({has_company*100//len(df_all)}%)")
    print(f"  Bio:      {has_bio}/{len(df_all)} ({has_bio*100//len(df_all)}%)")


class CollectGithubUserProfiles(TaskWithInputFileMonitor):
    def __init__(self, config_stage, config_global):
        super().__init__(config_stage, config_global)
        stage_param = config_stage['parameters']
        self.data_folder = Path(config_global['raw_data_folder'])

        self.repo_data_file_name = stage_param['repo_data_file_name_input']
        self.user_profiles_file_name = stage_param['user_profiles_name']

        # GitHub auth
        github_api_token = os.environ.get("GITHUB_API_KEY")
        if not github_api_token:
            raise ValueError("GITHUB_API_KEY environment variable is not set. "
                             "Make sure run_pipeline.py loads the .env file.")

        auth = Auth.Token(github_api_token)
        self.github_handle = Github(auth=auth)
        logging.getLogger("urllib3").propagate = False

        # Add custom filter for GitHub retry messages
        github_logger = logging.getLogger("github.GithubRetry")
        github_logger.addFilter(GithubRetryFilter())

    @loggedExecution
    def execute_task(self):
        collect_user_profiles(
            github_handle=self.github_handle,
            data_folder=self.data_folder,
            repo_data_file_name=self.repo_data_file_name,
            user_profiles_file_name=self.user_profiles_file_name,
        )


if __name__ == "__main__":
    """Standalone execution for testing or manual runs."""
    from dotenv import load_dotenv
    import argparse

    env_path = Path(__file__).parent.parent.parent / 'run' / '.env'
    load_dotenv(env_path)

    parser = argparse.ArgumentParser(description='Collect GitHub user profiles for LiaScript repo owners')
    parser.add_argument('--data-root', type=str,
                        default='/home/crosslab/Desktop/CL/liascript/raw',
                        help='Path to the raw data folder')
    parser.add_argument('--repos-pickle', type=str,
                        default='LiaScript_repositories.p',
                        help='Name of the repositories pickle file')
    parser.add_argument('--output', type=str,
                        default='LiaScript_user_profiles.p',
                        help='Name of the output pickle file')
    args = parser.parse_args()

    github_api_token = os.environ.get("GITHUB_API_KEY")
    if not github_api_token:
        print("Error: GITHUB_API_KEY not set. Check your .env file.")
        exit(1)

    auth = Auth.Token(github_api_token)
    github_handle = Github(auth=auth)

    # Add custom filter
    github_logger = logging.getLogger("github.GithubRetry")
    github_logger.addFilter(GithubRetryFilter())
    logging.basicConfig(level=logging.INFO)

    try:
        user = github_handle.get_user()
        print(f"Authenticated as: {user.login}")
    except Exception as e:
        print(f"Auth failed: {e}")
        exit(1)

    collect_user_profiles(
        github_handle=github_handle,
        data_folder=Path(args.data_root),
        repo_data_file_name=args.repos_pickle,
        user_profiles_file_name=args.output,
    )
