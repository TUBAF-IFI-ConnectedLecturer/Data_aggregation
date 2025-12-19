#!/usr/bin/env python3
"""
Generate HTML overview of processed PDFs with metadata comparison.

This script creates an HTML table comparing:
- BibTeX metadata (author, title)
- LLM-extracted metadata (author, title)
- GND-validated keywords
- Links to original PDFs

Output HTML is designed to be placed in /media/sz/Data/Veits_pdfs/data/raw_data
with relative links to ../data_pipeline/local_pdfs/raw/files/*.pdf
"""

import pandas as pd
from pathlib import Path
import sys
from typing import List, Dict
import html
import os

# Change to pipeline root directory for imports
script_path = Path(__file__).resolve()
pipeline_root = script_path.parent.parent.parent.parent.parent
os.chdir(pipeline_root)

# Add stage paths for unpickling
sys.path.insert(0, str(pipeline_root / 'stages/general'))
sys.path.insert(0, str(pipeline_root / 'stages/opal'))


def load_data(base_path: Path) -> tuple:
    """Load all required data files."""
    df_base = pd.read_pickle(base_path / 'LOCAL_files_base.p')

    # Try to load improved AI metadata first, fallback to original
    improved_file = base_path / 'LOCAL_ai_meta_improved.p'
    if improved_file.exists():
        df_ai = pd.read_pickle(improved_file)
        print(f"✓ Using improved AI metadata: {improved_file}")
    else:
        df_ai = pd.read_pickle(base_path / 'LOCAL_ai_meta.p')
        print(f"✓ Using original AI metadata (improved version not found)")

    df_keywords = pd.read_pickle(base_path / 'LOCAL_checked_keywords.p')

    return df_base, df_ai, df_keywords


def get_gnd_validated_keywords(keywords_list: List[Dict]) -> List[str]:
    """Extract only GND-validated keywords from the keywords list."""
    if not keywords_list or not isinstance(keywords_list, list):
        return []

    validated = []
    for kw in keywords_list:
        if isinstance(kw, dict) and kw.get('is_gnd', False):
            preferred_name = kw.get('gnd_preferred_name', '')
            if preferred_name:
                validated.append(preferred_name)

    return validated


def format_author_list(author_str: str) -> str:
    """Format author string for display."""
    if pd.isna(author_str) or not author_str:
        return '<em>N/A</em>'

    # Clean up and limit length
    author_str = str(author_str).strip()

    # Remove BibTeX curly braces
    author_str = author_str.replace('{', '').replace('}', '')

    if len(author_str) > 100:
        return html.escape(author_str[:100] + '...')
    return html.escape(author_str)


def format_title(title_str: str) -> str:
    """Format title for display."""
    if pd.isna(title_str) or not title_str:
        return '<em>N/A</em>'

    title_str = str(title_str).strip()

    # Remove BibTeX curly braces (used to protect capitalization)
    # {DDB} -> DDB, {FID} -> FID, etc.
    title_str = title_str.replace('{', '').replace('}', '')

    if len(title_str) > 150:
        return html.escape(title_str[:150] + '...')
    return html.escape(title_str)


def format_title_with_source(title_str: str, source: str) -> str:
    """
    Format title with source indicator badge.

    Args:
        title_str: The title text
        source: Source indicator ('layout', 'pdf_metadata', 'llm', 'none')

    Returns:
        HTML string with title and colored badge
    """
    formatted_title = format_title(title_str)

    # Choose badge based on source
    if source == 'layout':
        badge = '<span style="background-color: #9c27b0; color: white; padding: 2px 6px; border-radius: 3px; font-size: 0.75em; margin-left: 5px;" title="Titel aus PDF-Layout-Analyse (Schriftgröße)">LAYOUT</span>'
    elif source == 'pdf_metadata':
        badge = '<span style="background-color: #28a745; color: white; padding: 2px 6px; border-radius: 3px; font-size: 0.75em; margin-left: 5px;" title="Titel aus PDF-Metadaten">PDF</span>'
    elif source == 'llm':
        badge = '<span style="background-color: #007bff; color: white; padding: 2px 6px; border-radius: 3px; font-size: 0.75em; margin-left: 5px;" title="Titel von LLM extrahiert">AI</span>'
    else:
        badge = '<span style="background-color: #6c757d; color: white; padding: 2px 6px; border-radius: 3px; font-size: 0.75em; margin-left: 5px;" title="Kein Titel verfügbar">-</span>'

    return formatted_title + ' ' + badge


def format_keywords(keywords: List[str], max_display: int = 10) -> str:
    """Format keywords list for display."""
    if not keywords:
        return '<em>Keine GND-Keywords</em>'

    # Limit to max_display
    if len(keywords) > max_display:
        displayed = keywords[:max_display]
        html_kw = ', '.join(f'<span class="keyword">{html.escape(kw)}</span>' for kw in displayed)
        html_kw += f' <em>(+{len(keywords) - max_display} weitere)</em>'
    else:
        html_kw = ', '.join(f'<span class="keyword">{html.escape(kw)}</span>' for kw in keywords)

    return html_kw


def get_relative_pdf_path(file_path: str) -> str:
    """
    Convert absolute PDF path to relative path from /media/sz/Data/Veits_pdfs/data/raw_data.

    Example:
        Input:  /media/sz/Data/Veits_pdfs/data_pipeline/local_pdfs/raw/files/7972.pdf
        Output: ../../data_pipeline/local_pdfs/raw/files/7972.pdf
    """
    if pd.isna(file_path):
        return '#'

    # Extract filename from absolute path
    path = Path(file_path)
    filename = path.name

    # Construct relative path from /data/raw_data to /data_pipeline/local_pdfs/raw/files/
    return f"../../data_pipeline/local_pdfs/raw/files/{filename}"


def generate_html(df_base: pd.DataFrame, df_ai: pd.DataFrame, df_keywords: pd.DataFrame) -> str:
    """Generate HTML report."""

    # Merge dataframes
    df = df_base.merge(df_ai, on='pipe:ID', how='left')
    df = df.merge(df_keywords, on='pipe:ID', how='left')

    # Sort by ID for consistency
    df = df.sort_values('pipe:ID')

    html_parts = ['''<!DOCTYPE html>
<html lang="de">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PDF Metadata Analysis - Local PDFs Collection</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        h1 {
            color: #333;
            border-bottom: 3px solid #007bff;
            padding-bottom: 10px;
        }
        .stats {
            background-color: #e9ecef;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
        }
        .stats strong {
            color: #007bff;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
            font-size: 14px;
        }
        th {
            background-color: #007bff;
            color: white;
            padding: 12px 8px;
            text-align: left;
            position: sticky;
            top: 0;
            z-index: 10;
        }
        td {
            padding: 10px 8px;
            border-bottom: 1px solid #dee2e6;
            vertical-align: top;
        }
        tr:hover {
            background-color: #f8f9fa;
        }
        .bibtex-col {
            background-color: #fff3cd;
        }
        .ai-col {
            background-color: #d1ecf1;
        }
        .keyword-col {
            background-color: #d4edda;
        }
        .keyword {
            display: inline-block;
            background-color: #28a745;
            color: white;
            padding: 2px 6px;
            border-radius: 3px;
            margin: 2px;
            font-size: 12px;
        }
        .match {
            color: #28a745;
            font-weight: bold;
        }
        .mismatch {
            color: #dc3545;
            font-weight: bold;
        }
        .pdf-link {
            display: inline-block;
            background-color: #007bff;
            color: white;
            padding: 5px 10px;
            text-decoration: none;
            border-radius: 4px;
            font-size: 12px;
        }
        .pdf-link:hover {
            background-color: #0056b3;
        }
        em {
            color: #6c757d;
        }
        .id-col {
            font-weight: bold;
            color: #495057;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>📚 PDF Metadata Analysis - Local PDFs Collection</h1>

        <div class="stats">
            <strong>Gesamt:</strong> ''' + str(len(df)) + ''' Dokumente<br>
            <strong>Generiert:</strong> ''' + pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S') + '''<br>
            <strong>Quelle:</strong> /media/sz/Data/Veits_pdfs/data_pipeline/local_pdfs/
        </div>

        <div style="margin-top: 20px; padding: 15px; background-color: #fff3cd; border-left: 4px solid #ffc107; border-radius: 4px;">
            <strong style="color: #856404;">⚠️ WICHTIG: PDF-Links</strong>
            <p style="margin: 10px 0 0 0; color: #856404;">
                Die PDF-Links funktionieren nur, wenn sich diese HTML-Datei im gleichen Ordner wie die BibTeX-Dateien befindet.
            </p>
        </div>

        <table>
            <thead>
                <tr>
                    <th style="width: 60px;">ID</th>
                    <th style="width: 150px;">BibTeX Autor</th>
                    <th style="width: 200px;">BibTeX Titel</th>
                    <th style="width: 150px;">LLM Autor</th>
                    <th style="width: 200px;">Titel (AI/PDF)</th>
                    <th style="width: 300px;">GND-Keywords</th>
                    <th style="width: 80px;">PDF</th>
                </tr>
            </thead>
            <tbody>
''']

    # Generate table rows
    for idx, row in df.iterrows():
        doc_id = row['pipe:ID']

        # BibTeX data
        bibtex_author = format_author_list(row.get('bibtex:author', ''))
        bibtex_title = format_title(row.get('bibtex:title', ''))

        # AI data
        ai_author = format_author_list(row.get('ai:author', ''))

        # Title with source indicator
        ai_title_text = row.get('ai:title', '')
        title_source = row.get('ai:title_source', 'llm')  # Default to 'llm' if not present
        ai_title = format_title_with_source(ai_title_text, title_source)

        # Keywords
        keywords_list = row.get('keywords', [])
        gnd_keywords = get_gnd_validated_keywords(keywords_list)
        keywords_html = format_keywords(gnd_keywords)

        # PDF link
        pdf_path = get_relative_pdf_path(row.get('pipe:file_path', ''))

        html_parts.append(f'''                <tr>
                    <td class="id-col">{doc_id}</td>
                    <td class="bibtex-col">{bibtex_author}</td>
                    <td class="bibtex-col">{bibtex_title}</td>
                    <td class="ai-col">{ai_author}</td>
                    <td class="ai-col">{ai_title}</td>
                    <td class="keyword-col">{keywords_html}</td>
                    <td><a href="{pdf_path}" class="pdf-link" target="_blank">📄 PDF</a></td>
                </tr>
''')

    html_parts.append('''            </tbody>
        </table>

        <div style="margin-top: 20px; padding: 15px; background-color: #f8f9fa; border-left: 4px solid #007bff;">
            <strong>Hinweise:</strong>
            <ul>
                <li><span style="background-color: #fff3cd; padding: 2px 5px;">Gelb</span>: BibTeX-Metadaten (aus .bib-Dateien)</li>
                <li><span style="background-color: #d1ecf1; padding: 2px 5px;">Blau</span>: LLM-extrahierte Metadaten (aus PDF-Inhalten)</li>
                <li><span style="background-color: #d4edda; padding: 2px 5px;">Grün</span>: GND-validierte Keywords (durch Lobid bestätigt)</li>
                <li><strong>Titel-Quelle (Hybrid-Strategie):</strong>
                    <ul style="margin-top: 5px;">
                        <li><span style="background-color: #9c27b0; color: white; padding: 2px 6px; border-radius: 3px; font-size: 0.75em;">LAYOUT</span> = Layout-Analyse (größte Schriftgröße auf Seite 1) - <strong>94.5%</strong></li>
                        <li><span style="background-color: #28a745; color: white; padding: 2px 6px; border-radius: 3px; font-size: 0.75em;">PDF</span> = PDF-Metadaten (eingebettete Informationen) - <strong>0.1%</strong></li>
                        <li><span style="background-color: #007bff; color: white; padding: 2px 6px; border-radius: 3px; font-size: 0.75em;">AI</span> = LLM-Extraktion aus PDF-Text - <strong>5.4%</strong></li>
                        <li><span style="background-color: #6c757d; color: white; padding: 2px 6px; border-radius: 3px; font-size: 0.75em;">-</span> = Kein Titel verfügbar - <strong>0.0%</strong></li>
                    </ul>
                </li>
            </ul>
        </div>
    </div>
</body>
</html>''')

    return ''.join(html_parts)


def main():
    """Main execution."""
    # Paths
    data_path = Path('/media/sz/Data/Veits_pdfs/data_pipeline/local_pdfs/raw')
    output_path = Path('/media/sz/Data/Veits_pdfs/data/raw_data/metadata_overview.html')

    print("Loading data...")
    df_base, df_ai, df_keywords = load_data(data_path)

    print(f"  - Base: {len(df_base)} documents")
    print(f"  - AI metadata: {len(df_ai)} documents")
    print(f"  - Keywords: {len(df_keywords)} documents")

    print("\nGenerating HTML report...")
    html_content = generate_html(df_base, df_ai, df_keywords)

    print(f"Writing to: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_content, encoding='utf-8')

    print(f"\n✅ Success! HTML report generated:")
    print(f"   {output_path}")
    print(f"\nÖffnen Sie die Datei im Browser:")
    print(f"   file://{output_path}")


if __name__ == '__main__':
    main()
