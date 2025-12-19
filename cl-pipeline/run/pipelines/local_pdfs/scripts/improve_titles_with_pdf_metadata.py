#!/usr/bin/env python3
"""
Post-processing script to improve AI-extracted titles using PDF metadata fallback.

This script:
1. Loads AI-extracted metadata (ai:title) and PDF metadata (file:title)
2. Validates LLM-extracted titles for suspicious patterns
3. Falls back to PDF metadata when LLM title is unreliable
4. Saves improved titles with source tracking

Usage:
    cd /media/sz/Data/Veits_pdfs/Data_aggregation/cl-pipeline
    pipenv run python3 run/pipelines/local_pdfs/scripts/improve_titles_with_pdf_metadata.py

Author: Claude Code
Date: 2025-12-07
"""

import pandas as pd
from pathlib import Path
import sys
import os
import logging

# Add stage paths for unpickling
pipeline_root = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(pipeline_root / 'stages/general'))
sys.path.insert(0, str(pipeline_root / 'stages/opal'))
os.chdir(pipeline_root)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TitleImprover:
    """Improves AI-extracted titles using PDF metadata fallback"""

    def __init__(self, base_path: Path):
        self.base_path = base_path

    def load_data(self):
        """Load AI metadata and PDF metadata"""
        logger.info("Loading data files...")

        # Load AI metadata
        ai_meta_file = self.base_path / 'LOCAL_ai_meta.p'
        if not ai_meta_file.exists():
            raise FileNotFoundError(f"AI metadata file not found: {ai_meta_file}")
        self.df_ai = pd.read_pickle(ai_meta_file)
        logger.info(f"  Loaded AI metadata: {len(self.df_ai)} documents")

        # Load PDF metadata (file:title)
        pdf_meta_file = self.base_path / 'LOCAL_files_meta.p'
        if not pdf_meta_file.exists():
            raise FileNotFoundError(f"PDF metadata file not found: {pdf_meta_file}")
        self.df_pdf = pd.read_pickle(pdf_meta_file)
        logger.info(f"  Loaded PDF metadata: {len(self.df_pdf)} documents")

        # Merge on pipe:ID
        self.df = self.df_ai.merge(
            self.df_pdf[['pipe:ID', 'file:title', 'file:author']],
            on='pipe:ID',
            how='left'
        )
        logger.info(f"  Merged dataset: {len(self.df)} documents")

    def extract_pdf_title(self, pdf_title: str) -> str:
        """
        Extract and validate PDF embedded title metadata.

        Returns empty string if:
        - No title
        - Title is placeholder (e.g., "Untitled", "Document1")
        - Title is suspiciously short (<5 chars)
        """
        if pd.isna(pdf_title):
            return ''

        title = str(pdf_title).strip()

        if not title or len(title) < 5:
            return ''

        # Reject common placeholders
        placeholders = ['untitled', 'document', 'doc1', 'new document', 'ohne titel']
        if title.lower() in placeholders:
            return ''

        return title

    def is_llm_title_suspicious(self, title: str) -> bool:
        """
        Check if LLM-extracted title looks suspicious.

        Returns True if title appears to be:
        - Generic section headers
        - Error messages
        - Lists/multiple titles
        - Empty/very short
        """
        if pd.isna(title) or not title or len(str(title)) < 5:
            return True

        title_lower = str(title).lower().strip()

        # Suspicious patterns
        suspicious_patterns = [
            'kein titel',
            'no title',
            'untitled',
            'publikationen',
            'editorial',
            'abstract',
            'einleitung',
            'introduction',
            'bericht',  # Too generic
            'document title:',  # Prefix that should've been removed
            'personalia',  # Section header
            'vorwort',
            'inhaltsverzeichnis',
        ]

        for pattern in suspicious_patterns:
            if title_lower == pattern or title_lower.startswith(pattern + ':'):
                return True

        # Check for list patterns (multiple titles)
        if title_lower.startswith('1.') or title_lower.startswith('document titles'):
            return True

        # Check for multiple numbered items
        if '1.' in title and '2.' in title:
            return True

        return False

    def select_best_title(self, row) -> tuple:
        """
        Select the best title using hybrid strategy.

        Returns: (title, source) where source is 'pdf_metadata', 'llm', or 'none'
        """
        llm_title = row.get('ai:title', '')
        pdf_title = self.extract_pdf_title(row.get('file:title', ''))
        doc_id = row.get('pipe:ID', 'unknown')

        # Case 1: Both available - validate LLM title
        if pdf_title and llm_title:
            if self.is_llm_title_suspicious(llm_title):
                logger.debug(f"ID {doc_id}: LLM title suspicious, using PDF metadata")
                return (pdf_title, 'pdf_metadata')
            else:
                # LLM title looks good
                return (llm_title, 'llm')

        # Case 2: Only PDF title available
        if pdf_title and not llm_title:
            logger.debug(f"ID {doc_id}: No LLM title, using PDF metadata")
            return (pdf_title, 'pdf_metadata')

        # Case 3: Only LLM title available
        if llm_title and not pdf_title:
            if self.is_llm_title_suspicious(llm_title):
                logger.warning(f"ID {doc_id}: LLM title suspicious and no PDF fallback: {str(llm_title)[:60]}")
            return (llm_title, 'llm')

        # Case 4: Neither available
        logger.warning(f"ID {doc_id}: No title from PDF metadata or LLM")
        return ('', 'none')

    def improve_titles(self):
        """Apply hybrid title selection to all documents"""
        logger.info("Improving titles using hybrid strategy...")

        improved_count = 0
        llm_kept = 0
        pdf_used = 0

        improved_titles = []
        title_sources = []

        for idx, row in self.df.iterrows():
            best_title, source = self.select_best_title(row)

            improved_titles.append(best_title)
            title_sources.append(source)

            # Track statistics
            if source == 'pdf_metadata' and row.get('ai:title') != best_title:
                improved_count += 1
                pdf_used += 1
            elif source == 'llm':
                llm_kept += 1

        # Update DataFrame
        self.df['ai:title_improved'] = improved_titles
        self.df['ai:title_source'] = title_sources

        logger.info(f"\n=== Title Improvement Statistics ===")
        logger.info(f"  LLM titles kept:        {llm_kept} ({llm_kept/len(self.df)*100:.1f}%)")
        logger.info(f"  PDF metadata used:      {pdf_used} ({pdf_used/len(self.df)*100:.1f}%)")
        logger.info(f"  Actually improved:      {improved_count} ({improved_count/len(self.df)*100:.1f}%)")
        logger.info(f"  No title available:     {len(self.df) - llm_kept - pdf_used}")

    def save_results(self, output_file: str = None):
        """Save improved AI metadata"""
        if output_file is None:
            output_file = self.base_path / 'LOCAL_ai_meta_improved.p'
        else:
            output_file = self.base_path / output_file

        # Create output dataframe (only AI columns + improved title)
        ai_columns = [col for col in self.df.columns if col.startswith('ai:') or col.startswith('pipe:')]
        df_output = self.df[ai_columns].copy()

        # Replace ai:title with improved version
        df_output['ai:title'] = self.df['ai:title_improved']

        # Save
        df_output.to_pickle(output_file)
        logger.info(f"\n✓ Saved improved metadata to: {output_file}")

        # Also save comparison CSV for inspection
        csv_file = output_file.with_suffix('.comparison.csv')
        comparison_df = self.df[['pipe:ID', 'ai:title', 'file:title', 'ai:title_improved', 'ai:title_source']].copy()
        comparison_df.to_csv(csv_file, sep=';', index=False)
        logger.info(f"✓ Saved comparison CSV to: {csv_file}")


def main():
    """Main execution"""
    base_path = Path('/media/sz/Data/Veits_pdfs/data_pipeline/local_pdfs/raw')

    logger.info("=== Title Improvement with PDF Metadata Fallback ===\n")

    improver = TitleImprover(base_path)

    try:
        improver.load_data()
        improver.improve_titles()
        improver.save_results()

        logger.info("\n=== Done! ===")
        logger.info("Next steps:")
        logger.info("  1. Review: LOCAL_ai_meta_improved.comparison.csv")
        logger.info("  2. If satisfied: mv LOCAL_ai_meta.p LOCAL_ai_meta_backup.p")
        logger.info("  3. Then: mv LOCAL_ai_meta_improved.p LOCAL_ai_meta.p")

    except Exception as e:
        logger.error(f"Error: {e}")
        raise


if __name__ == '__main__':
    main()
