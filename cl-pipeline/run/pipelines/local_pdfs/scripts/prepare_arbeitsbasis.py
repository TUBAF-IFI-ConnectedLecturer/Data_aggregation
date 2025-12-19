#!/usr/bin/env python3
"""
Script to prepare Arbeitsbasis collection for pipeline processing.

This script handles the new structure with multiple journal folders:
1. Scans subdirectories in Arbeitsbasis (each = a journal/collection)
2. For each subdirectory, reads its .bib file
3. Scans files/<ID>/ structure within each journal folder
4. Creates unified folder structure and DataFrame
5. Generates symbolic links to target structure

Structure:
/media/sz/Data/Veits_pdfs/data/Arbeitsbasis/
  ├── Perspektive Bibliothek/
  │   ├── Perspektive Bibliothek.bib
  │   └── files/76/Apel und Hermann - 2017 - Klein, aber fein.pdf
  ├── LIBREAS/
  │   ├── LIBREAS.bib
  │   └── files/...
  └── ...
"""

import pandas as pd
from pathlib import Path
from tqdm import tqdm
import re

# Configuration
SOURCE_DIR = Path("/media/sz/Data/Veits_pdfs/data/Arbeitsbasis")
TARGET_BASE = Path("/media/sz/Data/Veits_pdfs/data_pipeline/arbeitsbasis")
TARGET_RAW = TARGET_BASE / "raw"
TARGET_FILES = TARGET_RAW / "files"
TARGET_CONTENT = TARGET_RAW / "content"
TARGET_PROCESSED = TARGET_BASE / "processed"

# Options
USE_SYMLINKS = True  # True = create symlinks, False = copy files
SKIP_MAC_METADATA = True  # Skip ._ files (macOS metadata)

# Journal Selection - Set to None to process all journals, or specify list of journal names
# Example: ["LIBREAS", "Perspektive Bibliothek", "obib"]
SELECTED_JOURNALS = [
    "LIBREAS",
    "Perspektive Bibliothek",
    "ABI-Technik",
    "obib"
]  # Set to None to process all journals


def extract_bibtex_field(entry, field_name):
    """
    Extract a BibTeX field value handling nested braces correctly.

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
    Parse a single BibTeX file and extract metadata.
    Returns a dictionary: {file_id: {author, title, keywords, ...}}
    """
    if not bibtex_file.exists():
        return {}

    bibtex_data = {}

    try:
        with open(bibtex_file, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"  ⚠ Error reading {bibtex_file}: {e}")
        return {}

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

    return bibtex_data


def find_journal_folders(source_dir):
    """Find all journal folders (subdirectories with .bib files)."""
    print(f"Scanning for journal folders in {source_dir}...")

    journal_folders = []
    for item in source_dir.iterdir():
        if item.is_dir() and not item.name.startswith('.'):
            # Look for .bib file in this folder
            bib_files = list(item.glob('*.bib'))
            if bib_files:
                journal_folders.append({
                    'folder': item,
                    'bib_file': bib_files[0],  # Take first .bib file
                    'name': item.name
                })

    print(f"✓ Found {len(journal_folders)} journal folders")
    return journal_folders


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


def process_journal_folder(journal_info, all_bibtex_data):
    """
    Process a single journal folder and collect file records.

    Args:
        journal_info: Dict with 'folder', 'bib_file', 'name'
        all_bibtex_data: Dict to accumulate all BibTeX data

    Returns:
        List of file records
    """
    folder = journal_info['folder']
    bib_file = journal_info['bib_file']
    journal_name = journal_info['name']

    print(f"\nProcessing: {journal_name}")
    print(f"  BibTeX file: {bib_file.name}")

    # Parse BibTeX for this journal
    bibtex_data = parse_bibtex(bib_file)
    print(f"  ✓ Parsed {len(bibtex_data)} BibTeX entries")

    # Merge into global BibTeX data
    all_bibtex_data.update(bibtex_data)

    # Find all PDFs in files/<ID>/ structure
    files_folder = folder / 'files'
    if not files_folder.exists():
        print(f"  ⚠ No 'files' folder found in {journal_name}")
        return []

    pdf_files = list(files_folder.rglob("*.pdf"))

    if SKIP_MAC_METADATA:
        pdf_files = [f for f in pdf_files if not f.name.startswith("._")]

    print(f"  ✓ Found {len(pdf_files)} PDF files")

    file_records = []
    bibtex_matches = 0

    for source_file in pdf_files:
        # Extract ID from path: files/<ID>/filename.pdf
        # source_file.parent = .../files/<ID>
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
                import shutil
                shutil.copy2(source_file, target_file)
        except Exception as e:
            print(f"  ✗ Error processing {source_file}: {e}")
            continue

        # Extract metadata from filename
        filename_metadata = extract_metadata_from_filename(source_file)

        # Create file record
        file_record = {
            'pipe:ID': pipe_id,
            'pipe:file_type': 'pdf',
            'pipe:file_path': target_file,
            'pipe:source_path': str(source_file),
            'pipe:original_filename': filename_metadata['original_filename'],
            'pipe:journal': journal_name,  # NEW: Track which journal this came from
        }

        # Check if BibTeX data exists for this file ID
        if pipe_id in bibtex_data:
            bibtex_meta = bibtex_data[pipe_id]
            bibtex_matches += 1

            # Add BibTeX fields
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

    print(f"  ✓ Processed {len(file_records)} files ({bibtex_matches} with BibTeX data)")
    return file_records


def create_folder_structure():
    """Create the expected folder structure."""
    print("\nCreating folder structure...")
    TARGET_FILES.mkdir(parents=True, exist_ok=True)
    TARGET_CONTENT.mkdir(parents=True, exist_ok=True)
    TARGET_PROCESSED.mkdir(parents=True, exist_ok=True)
    print(f"✓ Created folders at {TARGET_BASE}")


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

    # Journal distribution
    print(f"\n  Files per journal:")
    journal_counts = df['pipe:journal'].value_counts()
    for journal, count in journal_counts.items():
        print(f"    {journal}: {count}")

    return df


def main():
    """Main execution function."""
    print("=" * 70)
    print("Preparing Arbeitsbasis Collection for Pipeline")
    print("=" * 70)
    print()

    # Step 1: Create folder structure
    create_folder_structure()
    print()

    # Step 2: Find all journal folders
    journal_folders = find_journal_folders(SOURCE_DIR)
    print()

    # Step 3: Filter journals if SELECTED_JOURNALS is set
    if SELECTED_JOURNALS is not None:
        journal_folders = [j for j in journal_folders if j['name'] in SELECTED_JOURNALS]
        print(f"Filtering to {len(journal_folders)} selected journals: {SELECTED_JOURNALS}")
        print()

    # Step 4: Process each journal folder
    all_file_records = []
    all_bibtex_data = {}

    for journal_info in journal_folders:
        records = process_journal_folder(journal_info, all_bibtex_data)
        all_file_records.extend(records)

    print(f"\n{'='*70}")
    print(f"Total files collected: {len(all_file_records)}")
    print(f"Total BibTeX entries: {len(all_bibtex_data)}")
    print(f"{'='*70}\n")

    # Step 5: Create DataFrame
    df = create_base_dataframe(all_file_records)
    print()

    print("=" * 70)
    print("✓ Setup complete!")
    print("=" * 70)
    print()
    print(f"Target directory: {TARGET_BASE}")
    print(f"Files ready at: {TARGET_FILES}")
    print(f"Base DataFrame: {TARGET_RAW / 'LOCAL_files_base.p'}")
    print()
    print(f"Next step: Create a config file for arbeitsbasis pipeline")
    print(f"  or update pipelines/local_pdfs/config/full.yaml to point to:")
    print(f"  data_root_folder: {TARGET_BASE}")
    print()


if __name__ == "__main__":
    main()
