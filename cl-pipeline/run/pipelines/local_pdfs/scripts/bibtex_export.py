#!/usr/bin/env python3
"""
BibTeX Export Script for Local PDFs Pipeline

Exports AI-extracted metadata to BibTeX format with custom fields:

AI-extracted fields:
- author_ai: AI-extracted author names
- title_ai: AI-extracted title (hybrid: layout/PDF metadata/LLM)
- keywords_ai: AI-generated keywords (combined from all keyword fields)
- summary_ai: AI-generated summary
- dewey_ai: Dewey Decimal Classification
- type_ai: Document type

PDF embedded metadata fields:
- author_file: Author extracted from PDF metadata
- title_file: Title extracted from PDF metadata
- keywords_file: Keywords extracted from PDF metadata

Usage:
    python bibtex_export.py --input LOCAL_ai_meta_improved.p --files-meta LOCAL_files_meta.p --bibtex-source Artikelbasis.bib --output enriched_bibliography.bib
"""

import pandas as pd
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional
import re

# Add stages path to sys.path to allow unpickling of custom modules
# The pickle file may contain objects that reference modules like checkAuthorNames
# Script is in: cl-pipeline/run/pipelines/local_pdfs/scripts/bibtex_export.py
# Stages is in: cl-pipeline/stages/general
_stages_general_path = Path(__file__).parent.parent.parent.parent.parent / 'stages' / 'general'
if str(_stages_general_path) not in sys.path:
    sys.path.insert(0, str(_stages_general_path))

# Import checkAuthorNames to make it available for unpickling
# The pickle file references this module directly
try:
    import checkAuthorNames
except ImportError:
    pass  # Module not available, but we'll try to continue


class BibTeXExporter:
    """Export AI metadata to BibTeX format"""

    def __init__(self, input_file: Path, output_file: Path, data_root: Optional[Path] = None, bibtex_source: Optional[Path] = None, files_meta_source: Optional[Path] = None):
        self.input_file = input_file
        self.output_file = output_file
        self.data_root = data_root
        self.bibtex_source = bibtex_source
        self.files_meta_source = files_meta_source
        self.original_bibtex_data = {}

    def extract_braced_value(self, text: str, field_name: str) -> Optional[str]:
        """
        Extract field value with proper handling of nested braces.

        Args:
            text: The BibTeX entry text
            field_name: The field name to extract (e.g., 'title', 'author')

        Returns:
            The field value with outer braces removed, or None if not found
        """
        # Find the field assignment
        pattern = rf'{field_name}\s*=\s*\{{'
        match = re.search(pattern, text, re.IGNORECASE)
        if not match:
            return None

        # Start after the opening brace
        start_pos = match.end() - 1  # Position of the opening brace
        brace_count = 0
        pos = start_pos

        # Count braces to find the matching closing brace
        while pos < len(text):
            if text[pos] == '{':
                brace_count += 1
            elif text[pos] == '}':
                brace_count -= 1
                if brace_count == 0:
                    # Found the matching closing brace
                    value = text[start_pos + 1:pos]
                    return value.strip()
            pos += 1

        return None

    def parse_original_bibtex(self, bibtex_file: Path) -> Dict:
        """
        Parse original BibTeX file and extract metadata for each file ID.

        Args:
            bibtex_file: Path to BibTeX file

        Returns:
            Dictionary: {file_id: {author, title, keywords, ...}}
        """
        print(f"Parsing original BibTeX file: {bibtex_file}...")

        if not bibtex_file.exists():
            print(f"⚠ BibTeX file not found: {bibtex_file}")
            return {}

        bibtex_data = {}

        with open(bibtex_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # Split into entries (each entry starts with @)
        entries = re.split(r'@(\w+)\{', content)[1:]  # Split and capture entry type

        for i in range(0, len(entries), 2):
            if i + 1 >= len(entries):
                break

            entry_type = entries[i].lower()
            entry = entries[i + 1]

            # Extract file ID from the file field using the brace-aware parser
            file_value = self.extract_braced_value(entry, 'file')
            if not file_value:
                continue

            # Pattern: Full Text PDF:files/6513/Filename.pdf:application/pdf
            file_match = re.search(r'files/(\d+)/', file_value)
            if not file_match:
                continue

            file_id = file_match.group(1)

            # Extract fields
            metadata = {'entry_type': entry_type}

            # Extract all standard BibTeX fields using brace-aware parsing
            field_names = [
                'author', 'title', 'keywords', 'abstract', 'year', 'date',
                'doi', 'url', 'journaltitle', 'booktitle', 'publisher',
                'language', 'langid', 'pages', 'volume', 'number',
                'issn', 'isbn', 'file'
            ]

            for field_name in field_names:
                value = self.extract_braced_value(entry, field_name)
                if value:
                    # Map journaltitle to journal for consistency
                    if field_name == 'journaltitle':
                        metadata['journal'] = value
                    elif field_name == 'langid':
                        metadata['language'] = value
                    else:
                        metadata[field_name] = value

            # Extract year from date if not present
            if 'year' not in metadata and 'date' in metadata:
                date_str = metadata['date']
                year_part = date_str.split('-')[0]
                metadata['year'] = year_part

            bibtex_data[file_id] = metadata

        print(f"✓ Parsed {len(bibtex_data)} original BibTeX entries")
        return bibtex_data

    def load_metadata(self) -> pd.DataFrame:
        """Load AI metadata from pickle file"""
        print(f"Loading AI metadata from: {self.input_file}")
        df = pd.read_pickle(self.input_file)
        print(f"Loaded {len(df)} AI metadata entries")

        # Load original BibTeX data if available
        if self.bibtex_source and self.bibtex_source.exists():
            self.original_bibtex_data = self.parse_original_bibtex(self.bibtex_source)

            # Join original BibTeX data with AI metadata
            print("Joining original BibTeX data with AI metadata...")
            for idx, row in df.iterrows():
                file_id = str(row['pipe:ID'])
                if file_id in self.original_bibtex_data:
                    orig_data = self.original_bibtex_data[file_id]
                    # Add original fields with 'orig:' prefix
                    for key, value in orig_data.items():
                        df.at[idx, f'orig:{key}'] = value

            print(f"✓ Joined with {len(self.original_bibtex_data)} original entries")

        # Load PDF metadata from files_meta if available
        if self.files_meta_source and self.files_meta_source.exists():
            print(f"Loading PDF metadata from: {self.files_meta_source}")
            df_files = pd.read_pickle(self.files_meta_source)
            print(f"Loaded {len(df_files)} file metadata entries")

            # Join PDF metadata with AI metadata
            print("Joining PDF metadata with AI metadata...")
            for idx, row in df.iterrows():
                file_id = row['pipe:ID']
                # Find matching row in files_meta
                file_row = df_files[df_files['pipe:ID'] == file_id]
                if not file_row.empty:
                    file_row = file_row.iloc[0]
                    # Add file metadata fields
                    if 'file:author' in file_row and pd.notna(file_row['file:author']):
                        df.at[idx, 'file:author'] = file_row['file:author']
                    if 'file:title' in file_row and pd.notna(file_row['file:title']):
                        df.at[idx, 'file:title'] = file_row['file:title']
                    if 'file:keywords' in file_row and pd.notna(file_row['file:keywords']):
                        df.at[idx, 'file:keywords'] = file_row['file:keywords']

            print(f"✓ Joined PDF metadata for matching entries")

        return df

    def sanitize_bibtex_key(self, file_id: str) -> str:
        """
        Generate a valid BibTeX citation key from file ID.

        Args:
            file_id: File identifier (e.g., "7972")

        Returns:
            Valid BibTeX key (e.g., "pdf_7972")
        """
        # Remove any special characters and ensure it starts with a letter
        key = re.sub(r'[^a-zA-Z0-9_]', '_', str(file_id))
        if not key[0].isalpha():
            key = f"pdf_{key}"
        return key

    def sanitize_bibtex_value(self, value: str) -> str:
        """
        Sanitize a value for BibTeX format.

        Args:
            value: Raw string value

        Returns:
            Sanitized value safe for BibTeX
        """
        if not value or pd.isna(value):
            return ""

        # Convert to string
        value = str(value)

        # Replace curly braces with their escaped versions
        value = value.replace("{", "\\{").replace("}", "\\}")

        # Remove or escape other problematic characters
        value = value.replace("\\", "\\textbackslash ")
        value = value.replace("%", "\\%")
        value = value.replace("&", "\\&")
        value = value.replace("#", "\\#")

        return value.strip()

    def format_keywords(self, row: pd.Series) -> str:
        """
        Combine all keyword fields into a single comma-separated string.

        Args:
            row: DataFrame row with keyword fields

        Returns:
            Combined keywords string
        """
        keywords = []

        # Collect keywords from different fields
        for field in ['ai:keywords_ext', 'ai:keywords_gen', 'ai:keywords_dnb']:
            if field in row and row[field] and not pd.isna(row[field]):
                kw = str(row[field]).strip()
                if kw:
                    # Split if comma-separated and add individually
                    keywords.extend([k.strip() for k in kw.split(',') if k.strip()])

        # Remove duplicates while preserving order
        seen = set()
        unique_keywords = []
        for kw in keywords:
            kw_lower = kw.lower()
            if kw_lower not in seen:
                seen.add(kw_lower)
                unique_keywords.append(kw)

        return ", ".join(unique_keywords)

    def format_dewey(self, dewey_list: List) -> str:
        """
        Format Dewey classification list as string.

        Args:
            dewey_list: List of Dewey classification dicts

        Returns:
            Formatted Dewey string (e.g., "004 (Informatik), 020 (Bibliotheks- und Informationswissenschaften)")
        """
        # Handle None/NaN
        if dewey_list is None:
            return ""

        # For scalar values, use pd.isna
        if not isinstance(dewey_list, (list, tuple)):
            try:
                if pd.isna(dewey_list):
                    return ""
            except (TypeError, ValueError):
                pass

        # Handle empty lists
        if isinstance(dewey_list, (list, tuple)) and len(dewey_list) == 0:
            return ""

        if isinstance(dewey_list, str):
            return dewey_list

        if not isinstance(dewey_list, list):
            return str(dewey_list)

        dewey_strings = []
        for item in dewey_list:
            if isinstance(item, dict):
                notation = item.get('notation', '')
                label = item.get('label', '')
                if notation and label:
                    dewey_strings.append(f"{notation} ({label})")
                elif notation:
                    dewey_strings.append(notation)
            else:
                dewey_strings.append(str(item))

        return ", ".join(dewey_strings)

    def create_bibtex_entry(self, row: pd.Series) -> str:
        """
        Create a BibTeX entry from a metadata row.

        Args:
            row: DataFrame row with AI metadata

        Returns:
            BibTeX entry as string
        """
        # Generate citation key
        cite_key = self.sanitize_bibtex_key(row['pipe:ID'])

        # Determine entry type - prefer original, fallback to AI
        entry_type = "misc"  # Default
        if 'orig:entry_type' in row and row['orig:entry_type'] and not pd.isna(row['orig:entry_type']):
            entry_type = str(row['orig:entry_type']).lower()
        elif 'ai:type' in row and row['ai:type'] and not pd.isna(row['ai:type']):
            doc_type = str(row['ai:type']).lower()
            type_mapping = {
                'zeitschriftenartikel': 'article',
                'konferenzbeitrag': 'inproceedings',
                'forschungsbericht': 'techreport',
                'dissertation': 'phdthesis',
                'buchkapitel': 'inbook',
                'whitepaper': 'techreport',
                'technischer bericht': 'techreport',
            }
            entry_type = type_mapping.get(doc_type, 'misc')

        # Build entry
        lines = [f"@{entry_type}{{{cite_key},"]

        # Add ALL original BibTeX fields (preserve order: standard fields first)
        # Standard fields in typical BibTeX order
        standard_field_order = [
            'author', 'title', 'year', 'date', 'journal', 'booktitle', 'publisher',
            'volume', 'number', 'pages', 'doi', 'url', 'issn', 'isbn',
            'keywords', 'abstract', 'language', 'file'
        ]

        # First add standard fields in order
        for field in standard_field_order:
            orig_key = f'orig:{field}'
            if orig_key in row and row[orig_key] and not pd.isna(row[orig_key]):
                value = self.sanitize_bibtex_value(row[orig_key])
                if value:
                    lines.append(f"  {field} = {{{value}}},")

        # Then add any remaining orig: fields that aren't in standard order
        for col in row.index:
            if col.startswith('orig:') and col != 'orig:entry_type':
                field_name = col.replace('orig:', '')
                if field_name not in standard_field_order:
                    if row[col] and not pd.isna(row[col]):
                        value = self.sanitize_bibtex_value(row[col])
                        if value:
                            lines.append(f"  {field_name} = {{{value}}},")

        # Add separator comment
        lines.append("  % --- AI-extracted metadata below ---")

        # Add AI-extracted fields
        if 'ai:author' in row and row['ai:author'] and not pd.isna(row['ai:author']):
            author = self.sanitize_bibtex_value(row['ai:author'])
            if author:
                lines.append(f"  author_ai = {{{author}}},")

        if 'ai:title' in row and row['ai:title'] and not pd.isna(row['ai:title']):
            title = self.sanitize_bibtex_value(row['ai:title'])
            if title:
                lines.append(f"  title_ai = {{{title}}},")

        # Keywords (combined from all keyword fields)
        keywords = self.format_keywords(row)
        if keywords:
            keywords = self.sanitize_bibtex_value(keywords)
            lines.append(f"  keywords_ai = {{{keywords}}},")

        # Summary
        if 'ai:summary' in row and row['ai:summary'] and not pd.isna(row['ai:summary']):
            summary = self.sanitize_bibtex_value(row['ai:summary'])
            if summary:
                lines.append(f"  summary_ai = {{{summary}}},")

        # Dewey classification
        if 'ai:dewey' in row and row['ai:dewey']:
            dewey = self.format_dewey(row['ai:dewey'])
            if dewey:
                dewey = self.sanitize_bibtex_value(dewey)
                lines.append(f"  dewey_ai = {{{dewey}}},")

        # Document type
        if 'ai:type' in row and row['ai:type'] and not pd.isna(row['ai:type']):
            doc_type = self.sanitize_bibtex_value(row['ai:type'])
            lines.append(f"  type_ai = {{{doc_type}}},")

        # Add separator comment for PDF metadata (embedded in PDF files)
        lines.append("  % --- PDF embedded metadata below ---")

        # Add PDF metadata (extracted from PDF files via MetaDataExtraction)
        if 'file:author' in row and row['file:author'] and not pd.isna(row['file:author']):
            author = self.sanitize_bibtex_value(row['file:author'])
            if author:
                lines.append(f"  author_file = {{{author}}},")

        if 'file:title' in row and row['file:title'] and not pd.isna(row['file:title']):
            title = self.sanitize_bibtex_value(row['file:title'])
            if title:
                lines.append(f"  title_file = {{{title}}},")

        if 'file:keywords' in row and row['file:keywords'] and not pd.isna(row['file:keywords']):
            keywords = self.sanitize_bibtex_value(row['file:keywords'])
            if keywords:
                lines.append(f"  keywords_file = {{{keywords}}},")

        # Add separator comment for file ID
        lines.append("  % --- File ID ---")

        # Add file ID (internal reference)
        lines.append(f"  file_id = {{{row['pipe:ID']}}}")

        # Close entry
        lines.append("}")

        return "\n".join(lines)

    def export(self) -> int:
        """
        Export metadata to BibTeX file.

        Returns:
            Number of entries exported
        """
        # Load metadata
        df = self.load_metadata()

        # Filter out entries without any AI metadata
        df_with_ai = df[
            df['ai:title'].notna() |
            df['ai:author'].notna() |
            df['ai:summary'].notna()
        ].copy()

        print(f"Found {len(df_with_ai)} entries with AI metadata")

        if len(df_with_ai) == 0:
            print("Warning: No entries with AI metadata found!")
            return 0

        # Calculate missing entries (if files_meta was loaded)
        missing_ids = []
        total_files = len(df_with_ai)
        if self.files_meta_source and self.files_meta_source.exists():
            df_files = pd.read_pickle(self.files_meta_source)
            all_ids = set(df_files['pipe:ID'].values)
            exported_ids = set(df_with_ai['pipe:ID'].values)
            missing_ids = sorted(all_ids - exported_ids)
            total_files = len(df_files)

        # Create output directory if needed
        self.output_file.parent.mkdir(parents=True, exist_ok=True)

        # Write BibTeX file
        print(f"Writing BibTeX to: {self.output_file}")

        with open(self.output_file, 'w', encoding='utf-8') as f:
            # Write header comment
            f.write("% BibTeX export from Local PDFs AI Metadata Pipeline\n")
            f.write(f"% Generated from: {self.input_file}\n")
            f.write(f"% Total entries: {len(df_with_ai)}\n")
            if missing_ids:
                f.write(f"% Total files in collection: {total_files}\n")
                f.write(f"% Missing entries (no AI metadata): {len(missing_ids)}\n")
                f.write("%\n")
                f.write("% Missing file IDs (PDFs without AI metadata):\n")
                f.write(f"%   {', '.join(map(str, missing_ids))}\n")
            f.write("%\n")
            f.write("% Custom AI-extracted fields:\n")
            f.write("%   - author_ai: AI-extracted author names\n")
            f.write("%   - title_ai: AI-extracted title (hybrid: layout/PDF metadata/LLM)\n")
            f.write("%   - keywords_ai: Combined AI-generated keywords\n")
            f.write("%   - summary_ai: AI-generated summary\n")
            f.write("%   - dewey_ai: Dewey Decimal Classification\n")
            f.write("%   - type_ai: Document type\n")
            f.write("%\n")
            f.write("% PDF embedded metadata fields:\n")
            f.write("%   - author_file: Author extracted from PDF metadata\n")
            f.write("%   - title_file: Title extracted from PDF metadata\n")
            f.write("%   - keywords_file: Keywords extracted from PDF metadata\n")
            f.write("%\n\n")

            # Write entries
            for idx, row in df_with_ai.iterrows():
                entry = self.create_bibtex_entry(row)
                f.write(entry)
                f.write("\n\n")

        print(f"Successfully exported {len(df_with_ai)} entries to {self.output_file}")
        return len(df_with_ai)


def load_config(config_path: Path) -> Dict:
    """Load pipeline configuration to get data paths"""
    import yaml

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config


def main():
    parser = argparse.ArgumentParser(
        description="Export AI-extracted metadata to BibTeX format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export AI metadata only
  python bibtex_export.py --input ../../raw/LOCAL_ai_meta.p --output output.bib

  # Export with original BibTeX data merged
  python bibtex_export.py \\
    --input ../../raw/LOCAL_ai_meta.p \\
    --bibtex-source /path/to/Artikelbasis.bib \\
    --output output.bib

  # Export with config file (auto-detect paths)
  python bibtex_export.py --config ../config/full.yaml --output output.bib

  # Export test data
  python bibtex_export.py \\
    --input ../../raw/LOCAL_ai_meta_test.p \\
    --output output_test.bib
        """
    )

    parser.add_argument(
        '--input', '-i',
        type=Path,
        help='Input pickle file with AI metadata (e.g., LOCAL_ai_meta.p)'
    )

    parser.add_argument(
        '--output', '-o',
        type=Path,
        required=True,
        help='Output BibTeX file path'
    )

    parser.add_argument(
        '--config', '-c',
        type=Path,
        help='Pipeline config file (optional, for auto-detecting paths)'
    )

    parser.add_argument(
        '--bibtex-source', '-b',
        type=Path,
        help='Original BibTeX source file (e.g., Artikelbasis.bib) to merge with AI metadata'
    )

    parser.add_argument(
        '--files-meta', '-f',
        type=Path,
        help='Files metadata pickle file (e.g., LOCAL_files_meta.p) with PDF embedded metadata'
    )

    args = parser.parse_args()

    # Determine input file
    input_file = args.input
    data_root = None
    bibtex_source = args.bibtex_source
    files_meta_source = args.files_meta

    if not input_file and args.config:
        # Try to load from config
        try:
            config = load_config(args.config)
            data_root = Path(config['folder_structure']['data_root_folder'])
            raw_folder = Path(config['folder_structure']['raw_data_folder'])

            # Try to find the AI metadata file
            potential_files = [
                raw_folder / 'LOCAL_ai_meta.p',
                raw_folder / 'LOCAL_ai_meta_test.p',
            ]

            for pf in potential_files:
                if pf.exists():
                    input_file = pf
                    print(f"Auto-detected input file: {input_file}")
                    break
        except Exception as e:
            print(f"Error loading config: {e}", file=sys.stderr)

    if not input_file:
        print("Error: --input or --config must be provided", file=sys.stderr)
        parser.print_help()
        sys.exit(1)

    if not input_file.exists():
        print(f"Error: Input file not found: {input_file}", file=sys.stderr)
        sys.exit(1)

    # Export
    try:
        exporter = BibTeXExporter(input_file, args.output, data_root, bibtex_source, files_meta_source)
        count = exporter.export()

        if count > 0:
            print(f"\n✓ Successfully exported {count} entries")
            print(f"  Output: {args.output.absolute()}")
            sys.exit(0)
        else:
            print("\n⚠ No entries exported", file=sys.stderr)
            sys.exit(1)

    except Exception as e:
        print(f"\n✗ Export failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
