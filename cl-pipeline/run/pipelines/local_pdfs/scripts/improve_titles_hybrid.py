#!/usr/bin/env python3
"""
Hybrid title improvement using three sources: layout analysis, PDF metadata, and LLM.

This script:
1. Loads layout-based titles (font size analysis from first page)
2. Loads PDF embedded metadata (file:title)
3. Loads LLM-extracted titles (ai:title)
4. Selects best title using priority: layout > PDF metadata > LLM
5. Saves improved titles with source tracking

Priority strategy:
- Layout-based: Most reliable (actual visual title on first page)
- PDF metadata: Reliable if present and valid
- LLM extraction: Fallback when others unavailable

Usage:
    cd /media/sz/Data/Veits_pdfs/Data_aggregation/cl-pipeline
    pipenv run python3 run/pipelines/local_pdfs/scripts/improve_titles_hybrid.py

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


class HybridTitleImprover:
    """Improves titles using hybrid strategy: layout > PDF metadata > LLM"""

    def __init__(self, base_path: Path):
        self.base_path = base_path

    def load_data(self):
        """Load all three data sources"""
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

        # Load layout-based titles
        layout_file = self.base_path / 'LOCAL_layout_titles.p'
        if not layout_file.exists():
            raise FileNotFoundError(f"Layout titles file not found: {layout_file}")
        self.df_layout = pd.read_pickle(layout_file)
        logger.info(f"  Loaded layout titles: {len(self.df_layout)} documents")

        # Merge all sources on pipe:ID
        self.df = self.df_ai.merge(
            self.df_pdf[['pipe:ID', 'file:title', 'file:author']],
            on='pipe:ID',
            how='left'
        )
        self.df = self.df.merge(
            self.df_layout[['pipe:ID', 'layout:title', 'layout:status']],
            on='pipe:ID',
            how='left'
        )
        logger.info(f"  Merged dataset: {len(self.df)} documents")

    def extract_layout_title(self, row) -> str:
        """
        Extract and validate layout-based title.

        Returns empty string if:
        - Status is not 'extracted'
        - Title is empty or too short
        """
        if row.get('layout:status') != 'extracted':
            return ''

        title = row.get('layout:title', '')
        if pd.isna(title):
            return ''

        title = str(title).strip()

        # Minimum length check
        if len(title) < 10:
            return ''

        return title

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

        Priority: layout > PDF metadata > LLM

        Returns: (title, source) where source is:
        - 'layout': Layout-based extraction (font size analysis)
        - 'pdf_metadata': PDF embedded metadata
        - 'llm': LLM extraction
        - 'none': No valid title found
        """
        layout_title = self.extract_layout_title(row)
        pdf_title = self.extract_pdf_title(row.get('file:title', ''))
        llm_title = row.get('ai:title', '')
        doc_id = row.get('pipe:ID', 'unknown')

        # Priority 1: Layout-based title (most reliable)
        if layout_title:
            return (layout_title, 'layout')

        # Priority 2: PDF metadata (if LLM is suspicious or empty)
        if pdf_title:
            if not llm_title or self.is_llm_title_suspicious(llm_title):
                return (pdf_title, 'pdf_metadata')
            # If both PDF and LLM available and LLM looks good, prefer LLM
            # (LLM might have better formatting/cleanup)
            else:
                return (llm_title, 'llm')

        # Priority 3: LLM title (fallback)
        if llm_title:
            if self.is_llm_title_suspicious(llm_title):
                logger.warning(f"ID {doc_id}: Only suspicious LLM title: {str(llm_title)[:60]}")
            return (llm_title, 'llm')

        # No valid title found
        logger.warning(f"ID {doc_id}: No valid title from any source")
        return ('', 'none')

    def improve_titles(self):
        """Apply hybrid title selection to all documents"""
        logger.info("Improving titles using hybrid strategy (layout > PDF > LLM)...")

        layout_used = 0
        pdf_used = 0
        llm_kept = 0
        none_count = 0

        improved_titles = []
        title_sources = []

        for idx, row in self.df.iterrows():
            best_title, source = self.select_best_title(row)

            improved_titles.append(best_title)
            title_sources.append(source)

            # Track statistics
            if source == 'layout':
                layout_used += 1
            elif source == 'pdf_metadata':
                pdf_used += 1
            elif source == 'llm':
                llm_kept += 1
            else:
                none_count += 1

        # Update DataFrame
        self.df['ai:title_improved'] = improved_titles
        self.df['ai:title_source'] = title_sources

        total = len(self.df)
        logger.info(f"\n=== Hybrid Title Selection Statistics ===")
        logger.info(f"  Layout-based:           {layout_used} ({layout_used/total*100:.1f}%)")
        logger.info(f"  PDF metadata:           {pdf_used} ({pdf_used/total*100:.1f}%)")
        logger.info(f"  LLM extraction:         {llm_kept} ({llm_kept/total*100:.1f}%)")
        logger.info(f"  No valid title:         {none_count} ({none_count/total*100:.1f}%)")
        logger.info(f"  Total:                  {total}")

        # Calculate improvement over original LLM-only
        original_empty = self.df['ai:title'].isna().sum()
        new_empty = none_count
        logger.info(f"\n  Originally no title:    {original_empty}")
        logger.info(f"  Now no title:           {new_empty}")
        logger.info(f"  Newly filled:           {original_empty - new_empty}")

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

        # Add source tracking
        df_output['ai:title_source'] = self.df['ai:title_source']

        # Save
        df_output.to_pickle(output_file)
        logger.info(f"\n✓ Saved improved metadata to: {output_file}")

        # Also save comparison CSV for inspection
        csv_file = output_file.with_suffix('.comparison.csv')
        comparison_df = self.df[[
            'pipe:ID',
            'layout:title',
            'file:title',
            'ai:title',
            'ai:title_improved',
            'ai:title_source'
        ]].copy()
        comparison_df.to_csv(csv_file, sep=';', index=False)
        logger.info(f"✓ Saved comparison CSV to: {csv_file}")


def main():
    """Main execution"""
    base_path = Path('/media/sz/Data/Veits_pdfs/data_pipeline/local_pdfs/raw')

    logger.info("=== Hybrid Title Improvement (Layout + PDF + LLM) ===\n")

    improver = HybridTitleImprover(base_path)

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
