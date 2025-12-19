#!/usr/bin/env python3
"""
Script to prepare local PDF collection for pipeline processing.

This script:
1. Scans the source directory for PDFs in subdirectories
2. Reads BibTeX metadata from Artikelbasis.bib
3. Creates the expected folder structure
4. Creates symbolic links or copies files to the target structure
5. Generates a base DataFrame with file metadata (from filenames + BibTeX)
"""

import pandas as pd
from pathlib import Path
from tqdm import tqdm
import shutil
import hashlib
import os
import re

# Configuration
SOURCE_DIR = Path("/media/sz/Data/Veits_pdfs/data/raw_data/files")
BIBTEX_FILE = Path("/media/sz/Data/Veits_pdfs/data/raw_data/Artikelbasis.bib")
TARGET_BASE = Path("/media/sz/Data/Veits_pdfs/data_pipeline/local_pdfs")
TARGET_RAW = TARGET_BASE / "raw"
TARGET_FILES = TARGET_RAW / "files"
TARGET_CONTENT = TARGET_RAW / "content"
TARGET_PROCESSED = TARGET_BASE / "processed"

# Options
USE_SYMLINKS = True  # True = create symlinks, False = copy files
SKIP_MAC_METADATA = True  # Skip ._ files (macOS metadata)


def extract_bibtex_field(entry, field_name):
    """
    Extract a BibTeX field value handling nested braces correctly.

    BibTeX uses braces to protect text from case changes: title = {bavarikon - {DDB} - Europeana}
    The simple regex [^}]+ fails because it stops at the first }, missing nested braces.

    Args:
        entry: BibTeX entry text
        field_name: Field name to extract (e.g., 'title', 'author')

    Returns:
        Field value as string or None
    """
    pattern = rf'{field_name}\s*=\s*\{{'
    match = re.search(pattern, entry)
    if not match:
        return None

    start = match.end()
    brace_count = 1
    i = start

    # Find matching closing brace, handling nesting
    while i < len(entry) and brace_count > 0:
        if entry[i] == '{':
            brace_count += 1
        elif entry[i] == '}':
            brace_count -= 1
        i += 1

    if brace_count == 0:
        return entry[start:i-1].strip()
    return None


def parse_bibtex(bibtex_file):
    """
    Parse BibTeX file and extract metadata for each file ID.
    Returns a dictionary: {file_id: {author, title, keywords, ...}}
    """
    print(f"Parsing BibTeX file: {bibtex_file}...")

    if not bibtex_file.exists():
        print(f"⚠ BibTeX file not found: {bibtex_file}")
        return {}

    bibtex_data = {}

    with open(bibtex_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Split into entries (each entry starts with @)
    entries = re.split(r'@\w+\{', content)[1:]  # Skip first empty part

    for entry in entries:
        # Extract file ID from the file field
        # Pattern: file = {Full Text PDF:files/6513/Filename.pdf:application/pdf}
        file_match = re.search(r'file\s*=\s*\{[^:]*:files/(\d+)/', entry)
        if not file_match:
            continue

        file_id = file_match.group(1)

        # Extract fields using proper brace-aware parser
        metadata = {}

        # Author
        author = extract_bibtex_field(entry, 'author')
        if author:
            metadata['author'] = author

        # Title (with nested brace support)
        title = extract_bibtex_field(entry, 'title')
        if title:
            metadata['title'] = title

        # Keywords
        keywords = extract_bibtex_field(entry, 'keywords')
        if keywords:
            metadata['keywords'] = keywords

        # Year/Date
        date_value = extract_bibtex_field(entry, 'date')
        year_value = extract_bibtex_field(entry, 'year')
        if date_value:
            # Extract year from date (format: 2016-03 or 2016)
            year_part = date_value.split('-')[0]
            metadata['year'] = year_part
        elif year_value:
            metadata['year'] = year_value

        # DOI
        doi = extract_bibtex_field(entry, 'doi')
        if doi:
            metadata['doi'] = doi

        # URL
        url = extract_bibtex_field(entry, 'url')
        if url:
            metadata['url'] = url

        # Journal/Booktitle
        journal = extract_bibtex_field(entry, 'journaltitle')
        if journal:
            metadata['journal'] = journal

        # Abstract
        abstract = extract_bibtex_field(entry, 'abstract')
        if abstract:
            metadata['abstract'] = abstract

        # Language
        language = extract_bibtex_field(entry, 'langid')
        if language:
            metadata['language'] = language

        bibtex_data[file_id] = metadata

    print(f"✓ Parsed {len(bibtex_data)} BibTeX entries")
    return bibtex_data


def create_folder_structure():
    """Create the expected folder structure."""
    print("Creating folder structure...")
    TARGET_FILES.mkdir(parents=True, exist_ok=True)
    TARGET_CONTENT.mkdir(parents=True, exist_ok=True)
    TARGET_PROCESSED.mkdir(parents=True, exist_ok=True)
    print(f"✓ Created folders at {TARGET_BASE}")


def find_all_pdfs(source_dir):
    """Find all PDF files in subdirectories."""
    print(f"Scanning for PDFs in {source_dir}...")
    pdf_files = list(source_dir.rglob("*.pdf"))

    if SKIP_MAC_METADATA:
        pdf_files = [f for f in pdf_files if not f.name.startswith("._")]

    print(f"✓ Found {len(pdf_files)} PDF files")
    return pdf_files


def extract_metadata_from_filename(filename):
    """
    Extract metadata from filename.
    Expected format: "Author - Year - Title.pdf"
    """
    metadata = {
        'original_filename': filename.stem,
        'author': None,
        'year': None,
        'title': None
    }

    # Try to parse "Author - Year - Title" format
    parts = filename.stem.split(' - ', 2)
    if len(parts) >= 3:
        metadata['author'] = parts[0].strip()
        metadata['year'] = parts[1].strip()
        metadata['title'] = parts[2].strip()
    elif len(parts) == 2:
        metadata['author'] = parts[0].strip()
        metadata['title'] = parts[1].strip()
    else:
        metadata['title'] = filename.stem

    return metadata


def prepare_files(pdf_files, bibtex_data):
    """
    Prepare files by creating symlinks or copies in target structure.
    Merges metadata from filenames and BibTeX.
    Returns a list of file metadata.
    """
    print(f"\n{'Creating symlinks' if USE_SYMLINKS else 'Copying files'} to target directory...")

    file_records = []
    bibtex_matches = 0

    for source_file in tqdm(pdf_files):
        # Use the subdirectory name as ID (original folder number)
        folder_id = source_file.parent.name
        pipe_id = folder_id

        # Target file path
        target_file = TARGET_FILES / f"{pipe_id}.pdf"

        # Create symlink or copy
        if target_file.exists():
            target_file.unlink()

        try:
            if USE_SYMLINKS:
                target_file.symlink_to(source_file.resolve())
            else:
                shutil.copy2(source_file, target_file)
        except Exception as e:
            print(f"\n✗ Error processing {source_file}: {e}")
            continue

        # Extract metadata from filename
        filename_metadata = extract_metadata_from_filename(source_file)

        # Create file record with filename metadata as base
        file_record = {
            'pipe:ID': pipe_id,
            'pipe:file_type': 'pdf',
            'pipe:file_path': target_file,
            'pipe:source_path': str(source_file),
            'pipe:original_filename': filename_metadata['original_filename'],
        }

        # Check if BibTeX data exists for this file ID
        if pipe_id in bibtex_data:
            bibtex_meta = bibtex_data[pipe_id]
            bibtex_matches += 1

            # Use BibTeX data as primary source, filename as fallback
            file_record['bibtex:author'] = bibtex_meta.get('author', '')
            file_record['bibtex:title'] = bibtex_meta.get('title', '')
            file_record['bibtex:keywords'] = bibtex_meta.get('keywords', '')
            file_record['bibtex:year'] = bibtex_meta.get('year', '')
            file_record['bibtex:doi'] = bibtex_meta.get('doi', '')
            file_record['bibtex:url'] = bibtex_meta.get('url', '')
            file_record['bibtex:journal'] = bibtex_meta.get('journal', '')
            file_record['bibtex:abstract'] = bibtex_meta.get('abstract', '')
            file_record['bibtex:language'] = bibtex_meta.get('language', '')

            # Use BibTeX as primary, filename as fallback
            file_record['file:author'] = bibtex_meta.get('author') or filename_metadata['author']
            file_record['file:year'] = bibtex_meta.get('year') or filename_metadata['year']
            file_record['file:title'] = bibtex_meta.get('title') or filename_metadata['title']
        else:
            # No BibTeX data, use filename metadata only
            file_record['bibtex:author'] = ''
            file_record['bibtex:title'] = ''
            file_record['bibtex:keywords'] = ''
            file_record['bibtex:year'] = ''
            file_record['bibtex:doi'] = ''
            file_record['bibtex:url'] = ''
            file_record['bibtex:journal'] = ''
            file_record['bibtex:abstract'] = ''
            file_record['bibtex:language'] = ''

            file_record['file:author'] = filename_metadata['author']
            file_record['file:year'] = filename_metadata['year']
            file_record['file:title'] = filename_metadata['title']

        file_records.append(file_record)

    print(f"✓ Processed {len(file_records)} files ({bibtex_matches} with BibTeX data)")
    return file_records


def create_base_dataframe(file_records):
    """Create and save the base DataFrame."""
    print("\nCreating base DataFrame...")

    df = pd.DataFrame(file_records)

    # Save as pickle
    output_file = TARGET_RAW / "LOCAL_files_base.p"
    df.to_pickle(output_file)
    print(f"✓ Saved DataFrame to {output_file}")

    # Also save as CSV for inspection
    csv_file = TARGET_RAW / "LOCAL_files_base.csv"
    df.to_csv(csv_file, index=False)
    print(f"✓ Saved CSV to {csv_file}")

    # Print summary
    print(f"\nSummary:")
    print(f"  Total files: {len(df)}")
    print(f"  Files with author: {df['file:author'].notna().sum()}")
    print(f"  Files with year: {df['file:year'].notna().sum()}")
    print(f"  Files with title: {df['file:title'].notna().sum()}")
    print(f"\n  BibTeX data:")
    print(f"    With BibTeX author: {(df['bibtex:author'] != '').sum()}")
    print(f"    With BibTeX keywords: {(df['bibtex:keywords'] != '').sum()}")
    print(f"    With BibTeX DOI: {(df['bibtex:doi'] != '').sum()}")

    return df


def main():
    """Main execution function."""
    print("=" * 70)
    print("Preparing Local PDF Collection for Pipeline")
    print("=" * 70)
    print()

    # Step 1: Parse BibTeX file
    bibtex_data = parse_bibtex(BIBTEX_FILE)
    print()

    # Step 2: Create folder structure
    create_folder_structure()
    print()

    # Step 3: Find all PDFs
    pdf_files = find_all_pdfs(SOURCE_DIR)
    print()

    # Step 4: Prepare files (symlink/copy) and merge with BibTeX
    file_records = prepare_files(pdf_files, bibtex_data)
    print()

    # Step 5: Create DataFrame
    df = create_base_dataframe(file_records)
    print()

    print("=" * 70)
    print("✓ Setup complete!")
    print("=" * 70)
    print()
    print(f"Target directory: {TARGET_BASE}")
    print(f"Files ready at: {TARGET_FILES}")
    print(f"Base DataFrame: {TARGET_RAW / 'LOCAL_files_base.p'}")
    print()
    print(f"BibTeX entries matched: {len([r for r in file_records if r.get('bibtex:author')])}/{len(file_records)}")
    print()
    print("Next step: Run the pipeline with cl_local_pdfs.yaml")
    print()


if __name__ == "__main__":
    main()
