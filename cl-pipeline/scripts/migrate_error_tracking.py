#!/usr/bin/env python3
"""
Migration script: Initialize ai:_errors for documents with empty fields.

This one-time script analyzes the existing ai_meta pickle and pre-populates
the ai:_errors field for documents that likely failed extraction previously.
Documents with empty conditional fields get an error count of (max_retries - 1),
giving them exactly ONE more attempt before being permanently skipped.

Usage:
    python migrate_error_tracking.py <pickle_path> [--max-retries 3] [--dry-run]
"""

import argparse
import pickle
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import pandas as pd


class SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except (ModuleNotFoundError, AttributeError):
            return type(name, (), {})


CONDITIONAL_FIELDS = [
    'ai:author', 'ai:keywords_gen', 'ai:title', 'ai:type',
    'ai:keywords_ext', 'ai:keywords_dnb', 'ai:summary',
    'ai:education_level', 'ai:target_audience',
]


def is_empty(value):
    if value is None or value == "":
        return True
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        if isinstance(value, list) and len(value) == 0:
            return True
    return False


def main():
    parser = argparse.ArgumentParser(description="Initialize ai:_errors for stuck documents")
    parser.add_argument("pickle_path", help="Path to ai_meta pickle file")
    parser.add_argument("--max-retries", type=int, default=3,
                        help="max_error_retries value from config (default: 3)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Only show what would be changed, don't write")
    args = parser.parse_args()

    pickle_path = Path(args.pickle_path)
    if not pickle_path.exists():
        print(f"ERROR: File not found: {pickle_path}")
        sys.exit(1)

    # Load pickle
    with open(pickle_path, 'rb') as f:
        df = SafeUnpickler(f).load()

    print(f"Loaded {len(df)} documents from {pickle_path}")

    # Pre-set error count: max_retries - 1 gives exactly one more attempt
    preset_count = args.max_retries - 1
    now = datetime.now().isoformat()

    # Determine which fields exist in the dataframe
    available_fields = [f for f in CONDITIONAL_FIELDS if f in df.columns]
    print(f"Conditional fields in data: {available_fields}")

    # Initialize ai:_errors column if missing
    if 'ai:_errors' not in df.columns:
        df['ai:_errors'] = [{}] * len(df)

    # Ensure all entries are dicts
    df['ai:_errors'] = df['ai:_errors'].apply(
        lambda x: x if isinstance(x, dict) else {}
    )

    modified_count = 0
    field_counts = {}

    for idx, row in df.iterrows():
        errors = deepcopy(row['ai:_errors'])
        changed = False

        for field in available_fields:
            if field in row and is_empty(row[field]):
                # Only set error if not already tracked
                if field not in errors:
                    errors[field] = {
                        'count': preset_count,
                        'last_error': 'migrated_from_empty',
                        'last_attempt': now,
                    }
                    changed = True
                    field_counts[field] = field_counts.get(field, 0) + 1

        if changed:
            df.at[idx, 'ai:_errors'] = errors
            modified_count += 1

    print(f"\n--- Migration Summary ---")
    print(f"Documents modified: {modified_count}")
    print(f"Pre-set error count: {preset_count} (one more attempt before skip)")
    for field, count in sorted(field_counts.items()):
        print(f"  {field}: {count} entries initialized")

    if args.dry_run:
        print("\n[DRY RUN] No changes written.")
    else:
        # Backup original
        backup_path = pickle_path.with_suffix('.p.bak')
        if not backup_path.exists():
            import shutil
            shutil.copy2(pickle_path, backup_path)
            print(f"\nBackup saved to: {backup_path}")

        df.to_pickle(pickle_path)
        print(f"Updated pickle saved to: {pickle_path}")


if __name__ == "__main__":
    main()
