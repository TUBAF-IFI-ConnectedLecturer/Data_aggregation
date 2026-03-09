#!/usr/bin/env python3
"""
Compare pickle files across pipeline stages to find missing entries.
Helps identify if new files from add_internal_repos.py have propagated
through all pipeline stages.
"""

import pandas as pd
import argparse
from pathlib import Path


def load_pickle(path):
    """Load a pickle file, return DataFrame or None."""
    if not path.exists():
        return None
    return pd.read_pickle(path)


def compare_pickles(data_root):
    data_root = Path(data_root)

    pickles = {
        "LiaScript_repositories.p": {"key_cols": ["user", "name"]},
        "LiaScript_files.p": {"key_cols": ["repo_user", "repo_name", "pipe:ID"]},
        "LiaScript_files_validated.p": {"key_cols": ["repo_user", "repo_name", "pipe:ID"]},
        "LiaScript_ai_meta.p": {"key_cols": ["repo_user", "repo_name", "pipe:ID"]},
        "LiaScript_embeddings_files.p": {"key_cols": ["repo_user", "repo_name", "pipe:ID"]},
        "LiaScript_metadata.p": {"key_cols": ["repo_user", "repo_name", "pipe:ID"]},
        "LiaScript_content.p": {"key_cols": ["repo_user", "repo_name", "pipe:ID"]},
    }

    print(f"Data root: {data_root}\n")
    print(f"{'Pickle file':<40} {'Rows':>8}  {'Exists':>6}")
    print("-" * 60)

    loaded = {}
    for name, info in pickles.items():
        path = data_root / name
        df = load_pickle(path)
        if df is not None:
            loaded[name] = (df, info["key_cols"])
            print(f"{name:<40} {len(df):>8}  {'yes':>6}")
        else:
            print(f"{name:<40} {'—':>8}  {'no':>6}")

    # Compare files.p vs validated.p
    files_key = "LiaScript_files.p"
    validated_key = "LiaScript_files_validated.p"

    if files_key in loaded and validated_key in loaded:
        df_files, cols_f = loaded[files_key]
        df_validated, cols_v = loaded[validated_key]

        # Use pipe:ID as primary key
        id_col = "pipe:ID"
        if id_col in df_files.columns and id_col in df_validated.columns:
            ids_files = set(df_files[id_col])
            ids_validated = set(df_validated[id_col])

            missing = ids_files - ids_validated
            extra = ids_validated - ids_files

            print(f"\n--- files.p vs validated.p ---")
            print(f"In files.p but NOT in validated.p: {len(missing)}")
            print(f"In validated.p but NOT in files.p: {len(extra)}")

            if missing:
                print(f"\nMissing from validated.p (first 20):")
                df_missing = df_files[df_files[id_col].isin(missing)]
                for _, row in df_missing.head(20).iterrows():
                    print(f"  {row.get('repo_user', '?')}/{row.get('repo_name', '?')} - {row.get('file_path', row.get(id_col, '?'))}")

    # Compare validated.p vs ai_meta.p
    ai_key = "LiaScript_ai_meta.p"
    if validated_key in loaded and ai_key in loaded:
        df_validated, _ = loaded[validated_key]
        df_ai, _ = loaded[ai_key]

        id_col = "pipe:ID"
        if id_col in df_validated.columns and id_col in df_ai.columns:
            ids_validated = set(df_validated[id_col])
            ids_ai = set(df_ai[id_col])

            missing = ids_validated - ids_ai
            print(f"\n--- validated.p vs ai_meta.p ---")
            print(f"In validated.p but NOT in ai_meta.p: {len(missing)}")

            if missing:
                print(f"\nMissing from ai_meta.p (first 20):")
                df_missing = df_validated[df_validated[id_col].isin(missing)]
                for _, row in df_missing.head(20).iterrows():
                    print(f"  {row.get('repo_user', '?')}/{row.get('repo_name', '?')} - {row.get('file_path', row.get(id_col, '?'))}")

    # Check AI fields completeness in ai_meta.p
    if ai_key in loaded:
        df_ai, _ = loaded[ai_key]
        ai_fields = [c for c in df_ai.columns if c.startswith("ai:")]
        if ai_fields:
            print(f"\n--- AI field completeness in ai_meta.p ---")
            for field in ai_fields:
                filled = df_ai[field].notna() & (df_ai[field] != "")
                print(f"  {field:<30} {filled.sum():>6}/{len(df_ai)} filled")

    # Check for internal repos specifically
    if files_key in loaded:
        df_files, _ = loaded[files_key]
        if "internal" in df_files.columns:
            n_internal = df_files["internal"].sum() if df_files["internal"].dtype == bool else len(df_files[df_files["internal"] == True])
            print(f"\n--- Internal repos in files.p ---")
            print(f"Files marked as internal: {n_internal}")

            if validated_key in loaded:
                df_validated, _ = loaded[validated_key]
                internal_ids = set(df_files[df_files.get("internal", False) == True]["pipe:ID"])
                validated_ids = set(df_validated["pipe:ID"]) if "pipe:ID" in df_validated.columns else set()
                missing_internal = internal_ids - validated_ids
                print(f"Internal files missing from validated.p: {len(missing_internal)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check pipeline pickle consistency")
    parser.add_argument("--data-root", type=str,
                        default="/home/crosslab/Desktop/CL/liascript/raw",
                        help="Path to the raw data folder")
    args = parser.parse_args()
    compare_pickles(args.data_root)
