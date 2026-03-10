#!/usr/bin/env python3
"""
Check if internal LiaScript repos are covered across all pipeline stages.

Usage:
    python check_internal_coverage.py --data-root /path/to/liascript/raw
"""

import pandas as pd
import argparse
from pathlib import Path
from collections import Counter
try:
    import chromadb
    HAS_CHROMADB = True
except ImportError:
    HAS_CHROMADB = False

INTERNAL_ACCOUNTS = [
    'SebastianZug', 'andre-dietrich', 'LiaScript', 'LiaBooks',
    'LiaTemplates', 'LiaPlayground', 'TUBAF-IfI-LiaScript',
    'TUBAF-IUZ-LiaScript', 'markjjacob', 'HueblerPatricia',
]

PICKLE_FILES = {
    'repositories': 'LiaScript_repositories.p',
    'files': 'LiaScript_files.p',
    'validated': 'LiaScript_files_validated.p',
    'metadata': 'LiaScript_metadata.p',
    'features': 'LiaScript_features.p',
    'commits': 'LiaScript_commits.p',
    'content': 'LiaScript_content.p',
    'embeddings': 'LiaScript_embeddings_files.p',
    'ai_meta': 'LiaScript_ai_meta.p',
    'consolidated': 'LiaScript_consolidated.p',
}


def main():
    parser = argparse.ArgumentParser(description='Check internal repo coverage across pipeline stages')
    parser.add_argument('--data-root', type=str,
                        default='/home/crosslab/Desktop/CL/liascript/raw',
                        help='Path to the raw data folder')
    args = parser.parse_args()

    data_root = Path(args.data_root)
    print(f"Data root: {data_root}\n")

    # --- Step 1: Load repositories and identify internal ones ---
    repos_path = data_root / PICKLE_FILES['repositories']
    if not repos_path.exists():
        print(f"ERROR: {repos_path} not found")
        return

    df_repos = pd.read_pickle(repos_path)
    internal_repos = df_repos[df_repos['user'].isin(INTERNAL_ACCOUNTS)]
    print(f"=== Repositories ({PICKLE_FILES['repositories']}) ===")
    print(f"  Total: {len(df_repos)}, Internal: {len(internal_repos)}")
    per_account = internal_repos.groupby('user').size().sort_values(ascending=False)
    for account, count in per_account.items():
        print(f"    {account}: {count} repos")

    # --- Step 2: Check files.p ---
    files_path = data_root / PICKLE_FILES['files']
    if not files_path.exists():
        print(f"\nERROR: {files_path} not found")
        return

    df_files = pd.read_pickle(files_path)
    int_files = df_files[df_files['repo_user'].isin(INTERNAL_ACCOUNTS)]
    int_file_ids = set(int_files['pipe:ID'])

    print(f"\n=== Files ({PICKLE_FILES['files']}) ===")
    print(f"  Total: {len(df_files)}, Internal: {len(int_files)}")

    # --- Step 3: Check validated.p ---
    val_path = data_root / PICKLE_FILES['validated']
    if val_path.exists():
        df_val = pd.read_pickle(val_path)
        int_val = df_val[df_val['pipe:ID'].isin(int_file_ids)]

        # Count valid/invalid
        if 'pipe:is_valid_liascript' in int_val.columns:
            valid = int_val[int_val['pipe:is_valid_liascript'] == True]
            invalid = int_val[int_val['pipe:is_valid_liascript'] == False]
            not_validated = int_val[int_val['pipe:is_valid_liascript'].isna()]
            valid_ids = set(valid['pipe:ID'])
        else:
            valid_ids = int_file_ids
            valid = int_val
            invalid = pd.DataFrame()
            not_validated = pd.DataFrame()

        missing = int_file_ids - set(int_val['pipe:ID'])
        print(f"\n=== Validated ({PICKLE_FILES['validated']}) ===")
        print(f"  Internal in validated: {len(int_val)} / {len(int_files)}")
        print(f"    Valid: {len(valid)}")
        print(f"    Invalid: {len(invalid)}")
        print(f"    Not yet validated: {len(not_validated)}")
        print(f"    Missing from validated: {len(missing)}")
    else:
        print(f"\n  {PICKLE_FILES['validated']}: NOT FOUND")
        valid_ids = int_file_ids

    # --- Step 4: Check downstream pickles ---
    downstream_checks = [
        ('metadata', 'pipe:ID'),
        ('features', 'pipe:ID'),
        ('content', 'pipe:ID'),
        ('ai_meta', 'pipe:ID'),
        ('consolidated', 'pipe:ID'),
    ]

    print(f"\n=== Downstream Coverage (of {len(valid_ids)} valid internal files) ===")
    print(f"  {'Stage':<20} {'Total':>8} {'Internal':>10} {'Missing':>10} {'Coverage':>10}")
    print(f"  {'-'*60}")

    for stage_name, id_col in downstream_checks:
        pickle_path = data_root / PICKLE_FILES[stage_name]
        if not pickle_path.exists():
            print(f"  {stage_name:<20} {'—':>8} {'—':>10} {'—':>10} {'NOT FOUND':>10}")
            continue

        df = pd.read_pickle(pickle_path)
        if id_col not in df.columns:
            print(f"  {stage_name:<20} {len(df):>8} {'—':>10} {'—':>10} {'no ID col':>10}")
            continue

        in_stage = set(df[id_col]) & valid_ids
        missing = valid_ids - set(df[id_col])
        coverage = f"{len(in_stage)/len(valid_ids)*100:.1f}%" if valid_ids else "N/A"
        print(f"  {stage_name:<20} {len(df):>8} {len(in_stage):>10} {len(missing):>10} {coverage:>10}")

    # --- Embeddings: check ChromaDB directly ---
    processed_folder = data_root.parent / 'processed'
    chroma_path = processed_folder / 'chroma_db'
    if HAS_CHROMADB and chroma_path.exists():
        try:
            chroma_client = chromadb.PersistentClient(path=str(chroma_path))
            collection = chroma_client.get_or_create_collection(name='liascript_courses')
            total_chunks = collection.count()

            # Get all filenames from ChromaDB (paginated)
            chroma_files = set()
            batch_size = 10000
            offset = 0
            while offset < total_chunks:
                result = collection.get(limit=batch_size, offset=offset, include=["metadatas"])
                if not result['metadatas']:
                    break
                chroma_files.update(x["filename"] for x in result['metadatas'])
                offset += len(result['metadatas'])

            # Match against valid internal file IDs
            # ChromaDB filenames are "pipe:ID.file_type"
            chroma_ids = {f.rsplit('.', 1)[0] for f in chroma_files}
            in_chroma = valid_ids & chroma_ids
            missing_chroma = valid_ids - chroma_ids
            coverage = f"{len(in_chroma)/len(valid_ids)*100:.1f}%" if valid_ids else "N/A"
            print(f"  {'embeddings (chroma)':<20} {len(chroma_files):>8} {len(in_chroma):>10} {len(missing_chroma):>10} {coverage:>10}")
        except Exception as e:
            print(f"  {'embeddings (chroma)':<20} ERROR: {e}")
    elif not HAS_CHROMADB:
        print(f"  {'embeddings (chroma)':<20} chromadb not installed")
    else:
        print(f"  {'embeddings (chroma)':<20} {chroma_path} not found")

    # --- Step 5: Check commits separately (uses different key) ---
    commits_path = data_root / PICKLE_FILES['commits']
    if commits_path.exists():
        df_commits = pd.read_pickle(commits_path)
        if 'file_download_url' in df_commits.columns and 'file_download_url' in int_files.columns:
            int_urls = set(int_files[int_files['pipe:ID'].isin(valid_ids)]['file_download_url'])
            in_commits = int_urls & set(df_commits['file_download_url'])
            missing_commits = int_urls - set(df_commits['file_download_url'])
            coverage = f"{len(in_commits)/len(int_urls)*100:.1f}%" if int_urls else "N/A"
            print(f"  {'commits':<20} {len(df_commits):>8} {len(in_commits):>10} {len(missing_commits):>10} {coverage:>10}")
        elif 'pipe:ID' in df_commits.columns:
            in_commits = set(df_commits['pipe:ID']) & valid_ids
            missing_commits = valid_ids - set(df_commits['pipe:ID'])
            coverage = f"{len(in_commits)/len(valid_ids)*100:.1f}%" if valid_ids else "N/A"
            print(f"  {'commits':<20} {len(df_commits):>8} {len(in_commits):>10} {len(missing_commits):>10} {coverage:>10}")

    # --- Step 6: AI Meta field completeness for internal files ---
    ai_meta_path = data_root / PICKLE_FILES['ai_meta']
    if ai_meta_path.exists():
        df_ai = pd.read_pickle(ai_meta_path)
        int_ai = df_ai[df_ai['pipe:ID'].isin(valid_ids)]

        if len(int_ai) > 0:
            print(f"\n=== AI Meta Field Completeness (internal files: {len(int_ai)}) ===")
            ai_fields = [c for c in int_ai.columns if c.startswith('ai:')]
            for field in sorted(ai_fields):
                filled = int_ai[field].apply(lambda x: x is not None and x != "" and not (isinstance(x, float) and pd.isna(x)) and x != []).sum()
                print(f"  {field:<30} {filled:>5} / {len(int_ai)} filled")


if __name__ == '__main__':
    main()
