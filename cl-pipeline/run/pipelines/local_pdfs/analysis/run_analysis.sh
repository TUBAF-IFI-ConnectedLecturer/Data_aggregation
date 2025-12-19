#!/bin/bash
# Wrapper script to generate HTML overview for local PDFs pipeline

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_ROOT="$(cd "$SCRIPT_DIR/../../../../.." && pwd)"

echo "=== PDF Metadata Analysis ==="
echo "Pipeline root: $PIPELINE_ROOT"
echo ""

cd "$PIPELINE_ROOT"

echo "Running analysis script..."
pipenv run python3 run/pipelines/local_pdfs/analysis/generate_overview.py

echo ""
echo "✅ Done!"
echo ""
echo "Open the HTML file in your browser:"
echo "  file:///media/sz/Data/Veits_pdfs/data/raw_data/metadata_overview.html"
