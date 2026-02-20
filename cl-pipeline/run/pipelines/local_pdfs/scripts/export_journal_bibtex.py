#!/usr/bin/env python3
"""
Export enriched BibTeX files per journal from Arbeitsbasis pipeline results.

This script:
1. Reads the enriched metadata (AI + Lobid data)
2. Reads the original BibTeX metadata
3. Groups documents by journal
4. Creates separate enriched BibTeX files for each journal
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add paths for checkAuthorNames import (needed for unpickling)
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / 'stages' / 'general'))
# Also add the absolute path
sys.path.insert(0, '/media/sz/Data/Veits_pdfs/Data_aggregation/cl-pipeline/stages/general')

# Configuration
DATA_DIR = Path("/media/sz/Data/Veits_pdfs/data_pipeline/arbeitsbasis")
RAW_DIR = DATA_DIR / "raw"
OUTPUT_DIR = DATA_DIR / "enriched_bibtex"

# Input files
BASE_FILE = RAW_DIR / "LOCAL_files_base.p"
AI_META_FILE = RAW_DIR / "LOCAL_ai_meta.p"
CHECKED_KEYWORDS_FILE = RAW_DIR / "LOCAL_checked_keywords.p"


def format_author_list(authors):
    """Format author list for BibTeX."""
    if not authors or (isinstance(authors, float) and np.isnan(authors)):
        return ""

    if isinstance(authors, list):
        # Filter out institutional names and format
        valid_authors = []
        for author in authors:
            if hasattr(author, 'Vorname') and hasattr(author, 'Familienname'):
                if author.Vorname:
                    valid_authors.append(f"{author.Familienname}, {author.Vorname}")
                else:
                    valid_authors.append(author.Familienname)
            elif isinstance(author, str):
                valid_authors.append(author)

        return " and ".join(valid_authors) if valid_authors else ""

    return str(authors)


def format_keywords(keywords, max_count=None):
    """Format keywords for BibTeX, optionally limiting the number."""
    if not keywords or (isinstance(keywords, float) and np.isnan(keywords)):
        return ""

    if isinstance(keywords, list):
        items = [str(k) for k in keywords if k]
    else:
        items = [k.strip() for k in str(keywords).split(",") if k.strip()]

    if max_count is not None:
        items = items[:max_count]

    return ", ".join(items)


def format_dewey(dewey):
    """Format Dewey classification for BibTeX."""
    if not dewey or (isinstance(dewey, float) and np.isnan(dewey)):
        return ""

    if isinstance(dewey, list):
        # Extract notations from list of dicts
        notations = []
        for item in dewey:
            if isinstance(item, dict) and 'notation' in item:
                notations.append(item['notation'])
            elif isinstance(item, str):
                notations.append(item)
        return ", ".join(notations) if notations else ""

    return str(dewey)


def is_valid_value(value):
    """Check if a value is valid (not None, not NaN, not empty)."""
    if value is None:
        return False
    if isinstance(value, float) and np.isnan(value):
        return False
    if isinstance(value, str) and value == "":
        return False
    return True


def escape_bibtex_special_chars(text):
    """Escape special characters for BibTeX."""
    if not text or (isinstance(text, float) and np.isnan(text)):
        return ""

    text = str(text)
    # Replace special characters
    replacements = {
        '&': '\\&',
        '%': '\\%',
        '$': '\\$',
        '#': '\\#',
        '_': '\\_',
        '{': '\\{',
        '}': '\\}',
        '~': '\\textasciitilde{}',
        '^': '\\textasciicircum{}',
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    return text


def create_bibtex_entry(doc_id, base_data, ai_data, keyword_data):
    """
    Create a BibTeX entry from merged data.

    Args:
        doc_id: Document ID
        base_data: Original metadata from BibTeX
        ai_data: AI-extracted metadata
        keyword_data: Lobid-checked keywords

    Returns:
        BibTeX entry as string
    """
    # Start with article type (could be enhanced based on ai:type)
    entry_type = "article"
    if ai_data is not None and 'ai:type' in ai_data and is_valid_value(ai_data['ai:type']):
        doc_type = str(ai_data['ai:type']).lower()
        if 'buch' in doc_type or 'book' in doc_type:
            entry_type = "book"
        elif 'konferenz' in doc_type or 'conference' in doc_type:
            entry_type = "inproceedings"

    lines = [f"@{entry_type}{{{doc_id},"]

    # === Standard BibTeX fields (preserve original values) ===

    # Title (use original BibTeX value; AI title goes to title_ai)
    if 'bibtex:title' in base_data and is_valid_value(base_data['bibtex:title']):
        lines.append(f"  title = {{{escape_bibtex_special_chars(base_data['bibtex:title'])}}},")

    # Author (use original BibTeX value; AI author goes to author_ai)
    if 'bibtex:author' in base_data and is_valid_value(base_data['bibtex:author']):
        lines.append(f"  author = {{{escape_bibtex_special_chars(base_data['bibtex:author'])}}},")

    # Year
    year = None
    if 'bibtex:year' in base_data and is_valid_value(base_data['bibtex:year']):
        year = base_data['bibtex:year']
    elif 'file:year' in base_data and is_valid_value(base_data['file:year']):
        year = base_data['file:year']

    if year:
        lines.append(f"  year = {{{year}}},")

    # Journal
    journal = None
    if 'bibtex:journal' in base_data and is_valid_value(base_data['bibtex:journal']):
        journal = base_data['bibtex:journal']
    elif 'pipe:journal' in base_data and is_valid_value(base_data['pipe:journal']):
        journal = base_data['pipe:journal']

    if journal:
        lines.append(f"  journal = {{{escape_bibtex_special_chars(journal)}}},")

    # DOI
    if 'bibtex:doi' in base_data and is_valid_value(base_data['bibtex:doi']):
        lines.append(f"  doi = {{{base_data['bibtex:doi']}}},")

    # URL
    if 'bibtex:url' in base_data and is_valid_value(base_data['bibtex:url']):
        lines.append(f"  url = {{{base_data['bibtex:url']}}},")

    # Original Keywords (from BibTeX)
    if 'bibtex:keywords' in base_data and is_valid_value(base_data['bibtex:keywords']):
        orig_kw = format_keywords(base_data['bibtex:keywords'])
        if orig_kw:
            lines.append(f"  keywords = {{{escape_bibtex_special_chars(orig_kw)}}},")

    # Abstract (prefer original, fallback to AI summary)
    abstract = None
    if 'bibtex:abstract' in base_data and is_valid_value(base_data['bibtex:abstract']):
        abstract = base_data['bibtex:abstract']
    elif ai_data is not None and 'ai:summary' in ai_data and is_valid_value(ai_data['ai:summary']):
        abstract = ai_data['ai:summary']

    if abstract:
        lines.append(f"  abstract = {{{escape_bibtex_special_chars(abstract)}}},")

    # Language
    if 'bibtex:language' in base_data and is_valid_value(base_data['bibtex:language']):
        lines.append(f"  language = {{{base_data['bibtex:language']}}},")

    # Source file
    if 'pipe:original_filename' in base_data and is_valid_value(base_data['pipe:original_filename']):
        lines.append(f"  file = {{{base_data['pipe:original_filename']}.pdf}},")

    # === AI-extracted metadata section ===

    # AI Author
    if ai_data is not None:
        if 'ai:revisedAuthor' in ai_data and is_valid_value(ai_data['ai:revisedAuthor']):
            ai_author = format_author_list(ai_data['ai:revisedAuthor'])
        elif 'ai:author' in ai_data and is_valid_value(ai_data['ai:author']):
            ai_author = format_author_list(ai_data['ai:author'])
        else:
            ai_author = None

        if ai_author:
            lines.append(f"  author_ai = {{{escape_bibtex_special_chars(ai_author)}}},")

    # AI Title
    if ai_data is not None and 'ai:title' in ai_data and is_valid_value(ai_data['ai:title']):
        lines.append(f"  title_ai = {{{escape_bibtex_special_chars(ai_data['ai:title'])}}},")

    # AI Keywords (combined: extracted + generated + DNB/GND, max 7)
    if ai_data is not None:
        all_ai_kw = []
        for kw_field in ['ai:keywords_ext', 'ai:keywords_gen', 'ai:keywords_dnb']:
            if kw_field in ai_data and is_valid_value(ai_data[kw_field]):
                kw_val = ai_data[kw_field]
                if isinstance(kw_val, list):
                    all_ai_kw.extend([str(k) for k in kw_val if k])
                elif isinstance(kw_val, str):
                    all_ai_kw.extend([k.strip() for k in kw_val.split(",") if k.strip()])

        # Deduplicate while preserving order, then limit to 7
        seen = set()
        unique_ai_kw = []
        for kw in all_ai_kw:
            if kw.lower() not in seen:
                seen.add(kw.lower())
                unique_ai_kw.append(kw)
        unique_ai_kw = unique_ai_kw[:7]

        if unique_ai_kw:
            combined_ai_kw = ", ".join(unique_ai_kw)
            lines.append(f"  keywords_ai = {{{escape_bibtex_special_chars(combined_ai_kw)}}},")

    # AI Summary
    if ai_data is not None and 'ai:summary' in ai_data and is_valid_value(ai_data['ai:summary']):
        lines.append(f"  summary_ai = {{{escape_bibtex_special_chars(ai_data['ai:summary'])}}},")

    # AI Dewey
    if ai_data is not None and 'ai:dewey' in ai_data and is_valid_value(ai_data['ai:dewey']):
        dewey = format_dewey(ai_data['ai:dewey'])
        if dewey:
            lines.append(f"  dewey_ai = {{{dewey}}},")

    # AI Type
    if ai_data is not None and 'ai:type' in ai_data and is_valid_value(ai_data['ai:type']):
        lines.append(f"  type_ai = {{{escape_bibtex_special_chars(ai_data['ai:type'])}}},")

    # AI Affiliation
    if ai_data is not None and 'ai:affiliation' in ai_data and is_valid_value(ai_data['ai:affiliation']):
        affiliation = format_keywords(ai_data['ai:affiliation'])
        if affiliation:
            lines.append(f"  affiliation_ai = {{{escape_bibtex_special_chars(affiliation)}}},")

    # === PDF/File metadata section ===

    # File Author
    if 'file:author' in base_data and is_valid_value(base_data['file:author']):
        lines.append(f"  author_file = {{{escape_bibtex_special_chars(base_data['file:author'])}}},")

    # File Title
    if 'file:title' in base_data and is_valid_value(base_data['file:title']):
        lines.append(f"  title_file = {{{escape_bibtex_special_chars(base_data['file:title'])}}},")

    # File Year
    if 'file:year' in base_data and is_valid_value(base_data['file:year']):
        lines.append(f"  year_file = {{{base_data['file:year']}}},")

    # === File ID ===
    lines.append(f"  file_id = {{{doc_id}}}")

    # Close entry (remove trailing comma from last field)
    if lines[-1].endswith(','):
        lines[-1] = lines[-1][:-1]

    lines.append("}\n")

    return "\n".join(lines)


def export_journal_bibtex():
    """Main export function."""
    print("=" * 70)
    print("Exporting Enriched BibTeX Files per Journal")
    print("=" * 70)
    print()

    # Load data
    print("Loading data files...")
    df_base = pd.read_pickle(BASE_FILE)
    print(f"  ✓ Loaded {len(df_base)} base records")

    df_ai = pd.read_pickle(AI_META_FILE) if AI_META_FILE.exists() else pd.DataFrame()
    print(f"  ✓ Loaded {len(df_ai)} AI metadata records")

    df_keywords = pd.read_pickle(CHECKED_KEYWORDS_FILE) if CHECKED_KEYWORDS_FILE.exists() else pd.DataFrame()
    print(f"  ✓ Loaded {len(df_keywords)} keyword-checked records")
    print()

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")
    print()

    # Group by journal
    journals = df_base['pipe:journal'].unique()
    print(f"Found {len(journals)} journals:")
    for journal in journals:
        count = len(df_base[df_base['pipe:journal'] == journal])
        print(f"  - {journal}: {count} documents")
    print()

    # Export each journal
    total_entries = 0
    for journal in journals:
        print(f"Processing journal: {journal}")

        # Filter documents for this journal
        journal_docs = df_base[df_base['pipe:journal'] == journal]

        # Create BibTeX file
        journal_filename = journal.replace(' ', '_').replace('/', '_')
        output_file = OUTPUT_DIR / f"{journal_filename}_enriched.bib"

        entries = []
        for _, doc in journal_docs.iterrows():
            doc_id = doc['pipe:ID']

            # Find corresponding AI and keyword data
            ai_data = df_ai[df_ai['pipe:ID'] == doc_id].iloc[0] if len(df_ai[df_ai['pipe:ID'] == doc_id]) > 0 else None
            keyword_data = df_keywords[df_keywords['pipe:ID'] == doc_id].iloc[0] if len(df_keywords[df_keywords['pipe:ID'] == doc_id]) > 0 else None

            # Create BibTeX entry
            entry = create_bibtex_entry(doc_id, doc, ai_data, keyword_data)
            entries.append(entry)

        # Write to file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("\n".join(entries))

        print(f"  ✓ Exported {len(entries)} entries to {output_file.name}")
        total_entries += len(entries)

    print()
    print("=" * 70)
    print(f"✓ Export complete!")
    print(f"  Total journals: {len(journals)}")
    print(f"  Total entries: {total_entries}")
    print(f"  Output directory: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    export_journal_bibtex()
