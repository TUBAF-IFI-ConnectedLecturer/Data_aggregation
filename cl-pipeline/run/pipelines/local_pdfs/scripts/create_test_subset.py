#!/usr/bin/env python3
"""
Create a test subset of 50 documents for quick pipeline testing.

This script:
1. Loads the full LOCAL_files_base.p
2. Selects 50 documents (stratified by various criteria)
3. Saves as LOCAL_files_base_test.p
4. Reports statistics
"""

import pandas as pd
from pathlib import Path
import random

# Configuration
FULL_BASE_FILE = Path('/media/sz/Data/Veits_pdfs/data_pipeline/local_pdfs/raw/LOCAL_files_base.p')
TEST_BASE_FILE = Path('/media/sz/Data/Veits_pdfs/data_pipeline/local_pdfs/raw/LOCAL_files_base_test.p')
TEST_SIZE = 50
RANDOM_SEED = 42

def main():
    print("=== Creating Test Subset ===\n")

    # Load full dataset
    print(f"Loading full dataset: {FULL_BASE_FILE}")
    df_full = pd.read_pickle(FULL_BASE_FILE)
    print(f"  Total documents: {len(df_full)}")

    # Set random seed for reproducibility
    random.seed(RANDOM_SEED)

    # Strategy: Stratified sampling to get diverse documents
    # 1. Some with BibTeX keywords, some without
    # 2. Different years
    # 3. Different file sizes

    print("\n--- Sampling Strategy ---")

    # Get documents with and without BibTeX keywords
    has_keywords = df_full[df_full['bibtex:keywords'].notna()]
    no_keywords = df_full[df_full['bibtex:keywords'].isna()]

    print(f"Documents with BibTeX keywords: {len(has_keywords)}")
    print(f"Documents without BibTeX keywords: {len(no_keywords)}")

    # Sample proportionally
    n_with_keywords = min(30, len(has_keywords))
    n_without_keywords = TEST_SIZE - n_with_keywords

    sample_with = has_keywords.sample(n=n_with_keywords, random_state=RANDOM_SEED)
    sample_without = no_keywords.sample(n=n_without_keywords, random_state=RANDOM_SEED) if len(no_keywords) > 0 else pd.DataFrame()

    # Combine
    df_test = pd.concat([sample_with, sample_without])

    # Shuffle
    df_test = df_test.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    print(f"\n--- Test Subset Created ---")
    print(f"Total test documents: {len(df_test)}")
    print(f"  - With BibTeX keywords: {df_test['bibtex:keywords'].notna().sum()}")
    print(f"  - Without BibTeX keywords: {df_test['bibtex:keywords'].isna().sum()}")

    # Year distribution
    if 'bibtex:year' in df_test.columns:
        year_counts = df_test['bibtex:year'].value_counts().head(5)
        print(f"\nTop 5 years:")
        for year, count in year_counts.items():
            print(f"  {year}: {count} documents")

    # Show sample IDs
    print(f"\nSample document IDs:")
    sample_ids = df_test['pipe:ID'].head(10).tolist()
    print(f"  {', '.join(map(str, sample_ids))}...")

    # Save test subset
    print(f"\n--- Saving Test Subset ---")
    TEST_BASE_FILE.parent.mkdir(parents=True, exist_ok=True)
    df_test.to_pickle(TEST_BASE_FILE)
    print(f"✓ Saved to: {TEST_BASE_FILE}")

    # Also save as CSV for inspection
    csv_file = TEST_BASE_FILE.with_suffix('.csv')
    df_test.to_csv(csv_file, sep=';', index=False)
    print(f"✓ CSV saved to: {csv_file}")

    print("\n=== Test Subset Ready! ===")
    print(f"\nNext step: Run test pipeline:")
    print(f"  cd /media/sz/Data/Veits_pdfs/Data_aggregation/cl-pipeline/run")
    print(f"  pipenv run python3 run_pipeline.py -c pipelines/local_pdfs/config/test.yaml")

    print(f"\nEstimated runtime for {TEST_SIZE} documents:")
    print(f"  Stage 2 (Content): ~2-3 minutes")
    print(f"  Stage 3 (Filter): ~10 seconds")
    print(f"  Stage 4 (Embeddings): ~5-10 minutes")
    print(f"  Stage 5 (AI Metadata): ~15-25 minutes (llama3.3:70b)")
    print(f"  Stage 6 (GND Check): ~10-20 minutes (gemma3:27b + Lobid API)")
    print(f"  TOTAL: ~35-60 minutes")

if __name__ == '__main__':
    main()
