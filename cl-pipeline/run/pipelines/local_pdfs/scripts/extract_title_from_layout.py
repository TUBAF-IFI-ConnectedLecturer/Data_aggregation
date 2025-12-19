#!/usr/bin/env python3
"""
Extract document title from PDF layout analysis.

This script uses PyMuPDF to analyze the first page of PDFs and extract
the title based on visual properties:
- Font size (largest text is likely the title)
- Position (top of page)
- Text characteristics

This provides a more reliable title extraction than pure text-based LLM extraction.

Usage:
    cd /media/sz/Data/Veits_pdfs/Data_aggregation/cl-pipeline
    pipenv run python3 run/pipelines/local_pdfs/scripts/extract_title_from_layout.py

Author: Claude Code
Date: 2025-12-07
"""

import fitz  # PyMuPDF
import pandas as pd
from pathlib import Path
import sys
import os
import logging
from typing import Optional, Tuple, List, Dict
from collections import defaultdict

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


class LayoutTitleExtractor:
    """Extract title from PDF using layout/font analysis"""

    def __init__(self):
        self.base_path = Path('/media/sz/Data/Veits_pdfs/data_pipeline/local_pdfs/raw')

    def extract_text_blocks_with_font_info(self, pdf_path: Path) -> List[Dict]:
        """
        Extract text blocks from first page with font size information.

        Returns list of dicts with:
        - text: The text content
        - size: Font size
        - x0: X-coordinate (left position)
        - y0: Y-coordinate (top position)
        - bbox: Bounding box (x0, y0, x1, y1)
        """
        try:
            doc = fitz.open(pdf_path)
            if len(doc) == 0:
                return []

            page = doc[0]  # First page only
            blocks = []

            # Get text with detailed font information
            text_dict = page.get_text("dict")

            for block in text_dict.get("blocks", []):
                if block.get("type") != 0:  # Not a text block
                    continue

                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        text = span.get("text", "").strip()
                        if not text:
                            continue

                        bbox = span.get("bbox", [0, 0, 0, 0])
                        blocks.append({
                            "text": text,
                            "size": span.get("size", 0),
                            "font": span.get("font", ""),
                            "x0": bbox[0],  # Left X coordinate
                            "y0": bbox[1],  # Top Y coordinate
                            "bbox": bbox
                        })

            doc.close()
            return blocks

        except Exception as e:
            logger.error(f"Error extracting layout from {pdf_path}: {e}")
            return []

    def find_title_candidate(self, blocks: List[Dict]) -> Optional[str]:
        """
        Find the most likely title from text blocks.

        Strategy:
        1. Get all large text blocks in upper third of page (excluding headers/footers)
        2. Group by Y-position to form lines (handles drop-caps)
        3. Sort each line by X-position (left to right)
        4. Combine multi-line titles
        5. Filter out common non-title patterns
        """
        if not blocks:
            return None

        # Find maximum font size
        max_size = max(b["size"] for b in blocks)

        page_height = max(b["bbox"][3] for b in blocks) if blocks else 1000
        upper_third = page_height / 3

        # Define header/footer zones (typically very top/bottom of page)
        header_zone = page_height * 0.08  # Top 8% is likely header
        footer_zone = page_height * 0.92  # Bottom 8% is likely footer

        # Get all large font blocks in upper third
        # Use top 2-3 largest sizes to handle drop-caps, but require minimum 14pt
        large_sizes = sorted(set(b["size"] for b in blocks), reverse=True)[:3]
        # Filter out sizes that are too small (< 14pt = normal text size)
        large_sizes = [s for s in large_sizes if s >= 14.0]

        if not large_sizes:
            # Fallback: if no font >= 14pt, use maximum size
            min_title_size = max_size
        else:
            min_title_size = min(large_sizes)

        title_candidates = [
            b for b in blocks
            if b["size"] >= min_title_size * 0.95  # Include 2-3 largest sizes
            and b["y0"] < upper_third  # Upper third of page
            and b["y0"] > header_zone  # Exclude header zone
            and b["y0"] < footer_zone  # Exclude footer zone (not needed for upper third, but good practice)
        ]

        if not title_candidates:
            return None

        # Group blocks by Y-position into lines
        # Sort by Y first to process top-to-bottom
        title_candidates.sort(key=lambda x: (x["y0"], x["x0"]))

        lines = []
        current_line = []
        current_y = -1
        # Use larger threshold - blocks on same line can vary ~5pt in Y
        y_same_line_threshold = 10.0

        for block in title_candidates:
            if current_y < 0 or abs(block["y0"] - current_y) < y_same_line_threshold:
                # Same line - add to current line
                current_line.append(block)
                if current_y < 0:
                    current_y = block["y0"]
            else:
                # New line - save previous line
                if current_line:
                    # Sort line by X position (left to right)
                    current_line.sort(key=lambda x: x["x0"])
                    # Join text - add space between words
                    line_text = ""
                    prev_x1 = -1
                    for b in current_line:
                        x0 = b["bbox"][0]
                        x1 = b["bbox"][2]
                        # Add space if there's a significant gap between blocks
                        if prev_x1 > 0:
                            gap = x0 - prev_x1
                            # Add space only if gap is significant (>3pt)
                            # Small gaps (<=3pt) are drop-caps or kerning, not word boundaries
                            if gap > 3:
                                line_text += " "
                        line_text += b["text"]
                        prev_x1 = x1

                    line_text = line_text.strip()
                    if line_text:
                        lines.append({
                            "text": line_text,
                            "y0": current_y
                        })

                current_line = [block]
                current_y = block["y0"]

        # Don't forget last line
        if current_line:
            current_line.sort(key=lambda x: x["x0"])
            line_text = ""
            prev_x1 = -1
            for b in current_line:
                x0 = b["bbox"][0]
                x1 = b["bbox"][2]
                if prev_x1 > 0:
                    gap = x0 - prev_x1
                    if gap > 3:
                        line_text += " "
                line_text += b["text"]
                prev_x1 = x1

            line_text = line_text.strip()
            if line_text:
                lines.append({
                    "text": line_text,
                    "y0": current_y
                })

        # Now combine multi-line titles
        title_parts = []

        for line in lines:
            text = line["text"].strip()

            # Skip very short fragments
            if len(text) < 4:
                continue

            # Filter out common non-title patterns
            if self._is_non_title_text(text):
                continue

            title_parts.append(text)

            # Stop after we have enough for a title
            # Most titles are < 100 chars; stop earlier to avoid body text
            if len(" ".join(title_parts)) > 120:
                break

        if not title_parts:
            return None

        # Join parts with space
        title = " ".join(title_parts)

        # Clean up
        title = self._clean_title(title)

        return title if len(title) > 10 else None

    def _is_non_title_text(self, text: str) -> bool:
        """Check if text is likely NOT a title (header, page number, etc.)"""
        text_lower = text.lower().strip()

        # Common non-title patterns
        non_title_patterns = [
            # Page numbers, dates
            lambda t: t.isdigit(),
            lambda t: len(t) < 4,

            # Common headers/footers
            lambda t: t in ['abstract', 'introduction', 'einleitung', 'zusammenfassung'],

            # Journal/conference info (usually smaller or different position)
            lambda t: any(x in t for x in ['proceedings', 'conference', 'journal', 'volume', 'issue']),

            # Copyright/license
            lambda t: any(x in t for x in ['copyright', '©', 'cc by', 'license']),

            # Author indicators (these should be below title)
            lambda t: any(x in t for x in ['et al.', 'university', 'institut', 'department']),
        ]

        return any(pattern(text_lower) for pattern in non_title_patterns)

    def _clean_title(self, title: str) -> str:
        """Clean extracted title"""
        # Remove multiple spaces
        title = " ".join(title.split())

        # Remove trailing punctuation that shouldn't be there
        title = title.rstrip('.,;:')

        return title.strip()

    def process_all_pdfs(self) -> pd.DataFrame:
        """Process all PDFs and extract layout-based titles"""
        logger.info("Loading file metadata...")

        df_base = pd.read_pickle(self.base_path / 'LOCAL_files_base.p')
        logger.info(f"  {len(df_base)} PDFs to process")

        results = []
        errors = 0

        for idx, row in df_base.iterrows():
            doc_id = row['pipe:ID']
            pdf_path = Path(row['pipe:file_path'])

            if not pdf_path.exists():
                logger.warning(f"ID {doc_id}: PDF not found: {pdf_path}")
                results.append({
                    'pipe:ID': doc_id,
                    'layout:title': '',
                    'layout:status': 'file_not_found'
                })
                continue

            # Extract layout-based title
            blocks = self.extract_text_blocks_with_font_info(pdf_path)
            title = self.find_title_candidate(blocks)

            if title:
                status = 'extracted'
                logger.debug(f"ID {doc_id}: {title[:60]}")
            else:
                status = 'no_title_found'
                title = ''
                errors += 1

            results.append({
                'pipe:ID': doc_id,
                'layout:title': title,
                'layout:status': status
            })

            # Progress update every 100 PDFs
            if (idx + 1) % 100 == 0:
                logger.info(f"  Processed {idx + 1}/{len(df_base)} PDFs...")

        logger.info(f"\n=== Extraction Complete ===")
        logger.info(f"  Total: {len(results)}")
        logger.info(f"  Extracted: {len(results) - errors}")
        logger.info(f"  Failed: {errors}")

        return pd.DataFrame(results)

    def save_results(self, df_results: pd.DataFrame):
        """Save layout-based titles"""
        output_file = self.base_path / 'LOCAL_layout_titles.p'
        df_results.to_pickle(output_file)
        logger.info(f"\n✓ Saved to: {output_file}")

        # Also save CSV for inspection
        csv_file = output_file.with_suffix('.csv')
        df_results.to_csv(csv_file, sep=';', index=False)
        logger.info(f"✓ CSV: {csv_file}")


def test_single_pdf(pdf_path: str):
    """Test extraction on a single PDF"""
    extractor = LayoutTitleExtractor()
    blocks = extractor.extract_text_blocks_with_font_info(Path(pdf_path))

    print(f"\n=== Text blocks from first page ===")
    print(f"Total blocks: {len(blocks)}\n")

    # Show top 10 largest text blocks
    sorted_blocks = sorted(blocks, key=lambda x: x["size"], reverse=True)[:10]
    for b in sorted_blocks:
        print(f"Size {b['size']:.1f}: {b['text'][:80]}")

    print(f"\n=== Extracted Title ===")
    title = extractor.find_title_candidate(blocks)
    print(f"Title: {title}")


def main():
    """Main execution"""
    logger.info("=== Layout-based Title Extraction ===\n")

    # Test mode: process single PDF
    if len(sys.argv) > 1:
        test_pdf = sys.argv[1]
        test_single_pdf(test_pdf)
        return

    # Process all PDFs
    extractor = LayoutTitleExtractor()
    df_results = extractor.process_all_pdfs()
    extractor.save_results(df_results)

    logger.info("\n=== Done! ===")
    logger.info("Next step: Compare with LLM and PDF metadata titles")


if __name__ == '__main__':
    main()
