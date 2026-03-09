#!/usr/bin/env python3
"""
Script to add previously blacklisted "internal" LiaScript repos to the existing dataset.

This script:
1. Queries GitHub for all repos of the internal accounts
2. Collects repo metadata in the same format as searchForLia.py
3. Merges them into the existing LiaScript_repositories.p
4. Then aggregates LiaScript files from these repos (same logic as aggregateLia.py)
5. Merges them into the existing LiaScript_files.p

Usage:
    python add_internal_repos.py --data-root /path/to/liascript/raw

    Or with defaults from the pipeline config:
    python add_internal_repos.py
"""

import pandas as pd
import hashlib
import argparse
import os
import sys
from pathlib import Path
from tqdm import tqdm
from github import Github, GithubException
from github import Auth
from dotenv import load_dotenv

# Load .env from run directory
env_path = Path(__file__).parent / 'run' / '.env'
load_dotenv(env_path)

# Import shared functions from the pipeline stages
sys.path.insert(0, str(Path(__file__).parent))
from stages.liascript.searchForLia import get_repo_meta_data
from stages.liascript.aggregateLia import extract_liafile_meta_data


# The accounts that were previously blacklisted
INTERNAL_ACCOUNTS = [
    'SebastianZug',
    'andre-dietrich',
    'LiaScript',
    'LiaBooks',
    'LiaTemplates',
    'LiaPlayground',
    'TUBAF-IfI-LiaScript',
    'TUBAF-IUZ-LiaScript',
    'markjjacob',
    'HueblerPatricia',
]


def get_repos_for_account(github_handle, account_name):
    """Get all non-fork, non-archived repos for an account (user or org)."""
    repos = []
    try:
        # Try as organization first
        org = github_handle.get_organization(account_name)
        repo_iter = org.get_repos(type='public')
        source = 'org'
    except GithubException:
        # Fall back to user
        user = github_handle.get_user(account_name)
        repo_iter = user.get_repos(type='public')
        source = 'user'

    print(f"  [{source}] {account_name}: fetching repos...")
    for repo in tqdm(repo_iter, desc=f"  {account_name}", leave=True):
        if repo.fork or repo.archived:
            continue
        try:
            meta = get_repo_meta_data(repo)
            meta['searched_type'] = 'internal_account'
            meta['internal'] = True
            repos.append(meta)
        except Exception as e:
            print(f"    Error getting metadata for {repo.full_name}: {e}")

    print(f"  -> {len(repos)} repos (non-fork, non-archived)")
    return repos


def aggregate_files_for_repos(github_handle, df_repos, data_folder, file_folder,
                               existing_files_pickle):
    """Aggregate LiaScript files from the given repos, same logic as aggregateLia.py."""
    storage_folder = Path(file_folder)
    storage_folder.mkdir(parents=True, exist_ok=True)

    if existing_files_pickle.exists():
        df_files = pd.read_pickle(existing_files_pickle)
        existing_keys = set(zip(df_files['repo_user'], df_files['repo_name']))
    else:
        df_files = pd.DataFrame()
        existing_keys = set()

    new_count = 0
    for i, row in df_repos.iterrows():
        repo_key = (row['user'], row['name'])
        if repo_key in existing_keys:
            print(f"  {row['user']}/{row['name']} - already aggregated")
            continue

        print(f"  {row['user']}/{row['name']} - ", end='', flush=True)
        try:
            repo = github_handle.get_repo(f"{row['user']}/{row['name']}")
            contents = repo.get_contents("")
            files = []
            while contents:
                fc = contents.pop(0)
                if fc.type == "dir":
                    contents.extend(repo.get_contents(fc.path))
                else:
                    files.append(fc)
        except Exception as e:
            print(f"error: {e}")
            continue

        if not files:
            print("no files")
            continue

        print(f"{len(files)} files ", end="", flush=True)
        course_list = []
        for f in files:
            result = extract_liafile_meta_data(f)
            if result is None or result[0] is None:
                continue
            repo_data, content = result
            # Save file content
            file_name = f"{repo_data['pipe:ID']}.md"
            file_path = storage_folder / file_name
            with open(file_path, "w", encoding='utf-8') as fh:
                fh.write(content)
            course_list.append(repo_data)

        if course_list:
            df_new = pd.DataFrame(course_list)
            df_files = pd.concat([df_files, df_new], ignore_index=True)
            new_count += len(course_list)
            print(f" -> {len(course_list)} md files")
        else:
            print(" -> 0 md files")

    return df_files, new_count


def main():
    parser = argparse.ArgumentParser(description='Add internal LiaScript repos to dataset')
    parser.add_argument('--data-root', type=str,
                        default='/home/crosslab/Desktop/CL/liascript/raw',
                        help='Path to the raw data folder')
    parser.add_argument('--file-folder', type=str, default=None,
                        help='Path to store downloaded files (default: <data-root>/files)')
    parser.add_argument('--repos-pickle', type=str, default='LiaScript_repositories.p',
                        help='Name of the repositories pickle file')
    parser.add_argument('--files-pickle', type=str, default='LiaScript_files.p',
                        help='Name of the files pickle file')
    parser.add_argument('--dry-run', action='store_true',
                        help='Only collect repos, do not aggregate files')
    parser.add_argument('--accounts', nargs='+', default=None,
                        help='Override which accounts to process (default: all internal)')
    args = parser.parse_args()

    data_root = Path(args.data_root)
    file_folder = Path(args.file_folder) if args.file_folder else data_root / 'files'
    repos_pickle = data_root / args.repos_pickle
    files_pickle = data_root / args.files_pickle

    print(f"Data root:       {data_root}")
    print(f"File folder:     {file_folder}")
    print(f"Repos pickle:    {repos_pickle}")
    print(f"Files pickle:    {files_pickle}")
    print()

    # GitHub auth
    github_api_token = os.environ.get("GITHUB_API_KEY")
    if not github_api_token:
        print("Error: GITHUB_API_KEY not set. Check your .env file.")
        sys.exit(1)

    auth = Auth.Token(github_api_token)
    github_handle = Github(auth=auth)

    try:
        user = github_handle.get_user()
        print(f"Authenticated as: {user.login}")
    except Exception as e:
        print(f"Auth failed: {e}")
        sys.exit(1)

    accounts = args.accounts or INTERNAL_ACCOUNTS

    # --- Step 1: Collect repo metadata ---
    print(f"\n=== Step 1: Collecting repos from {len(accounts)} accounts ===")
    all_repos = []
    for account in accounts:
        repos = get_repos_for_account(github_handle, account)
        all_repos.extend(repos)

    df_internal = pd.DataFrame(all_repos)
    print(f"\nTotal internal repos found: {len(df_internal)}")

    if df_internal.empty:
        print("No repos found. Exiting.")
        return

    # Deduplicate
    df_internal = df_internal.drop_duplicates(
        subset=['name', 'user'], keep='first'
    ).reset_index(drop=True)
    print(f"After dedup: {len(df_internal)} unique repos")

    # --- Step 2: Merge with existing repos pickle ---
    print(f"\n=== Step 2: Merging into {repos_pickle} ===")
    if repos_pickle.exists():
        df_existing = pd.read_pickle(repos_pickle)
        existing_keys = set(zip(df_existing['user'], df_existing['name']))
        new_repos = df_internal[
            ~df_internal.apply(lambda r: (r['user'], r['name']) in existing_keys, axis=1)
        ]
        print(f"Existing repos: {len(df_existing)}, new to add: {len(new_repos)}")
        df_merged = pd.concat([df_existing, new_repos], ignore_index=True)
    else:
        df_merged = df_internal
        print(f"No existing pickle found, creating new with {len(df_merged)} repos")

    if args.dry_run:
        print(f"\n--dry-run: Would add {len(new_repos) if repos_pickle.exists() else len(df_merged)} repos. Skipping save and file aggregation.")
        return

    # Save backup and new version
    if repos_pickle.exists():
        backup = repos_pickle.with_suffix('.p.bak')
        pd.read_pickle(repos_pickle).to_pickle(backup)
        print(f"Backup saved: {backup}")

    df_merged.to_pickle(repos_pickle)
    print(f"Saved {len(df_merged)} repos to {repos_pickle}")

    # --- Step 3: Aggregate LiaScript files ---
    print(f"\n=== Step 3: Aggregating LiaScript files from internal repos ===")

    # Only process the internal repos
    df_to_process = df_internal

    if files_pickle.exists():
        backup = files_pickle.with_suffix('.p.bak')
        pd.read_pickle(files_pickle).to_pickle(backup)
        print(f"Backup saved: {backup}")

    df_all_files, new_count = aggregate_files_for_repos(
        github_handle=github_handle,
        df_repos=df_to_process,
        data_folder=data_root,
        file_folder=file_folder,
        existing_files_pickle=files_pickle,
    )

    df_all_files.reset_index(drop=True, inplace=True)
    df_all_files.to_pickle(files_pickle)
    df_all_files.to_csv(files_pickle.with_suffix('.csv'))
    print(f"\nDone! Added {new_count} new LiaScript files.")
    print(f"Total files in dataset: {len(df_all_files)}")


if __name__ == '__main__':
    main()
