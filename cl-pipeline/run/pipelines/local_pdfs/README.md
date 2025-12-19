# Local PDFs Pipeline

This pipeline processes a local collection of scientific research papers (PDFs) and extracts comprehensive metadata using AI-based techniques.

## Overview

The Local PDFs pipeline is optimized for academic research papers with structured metadata. It processes 1,010 scientific papers from various domains.

### Document Characteristics
- **Language**: 97.2% German, 2.8% English
- **Type**: Scientific papers (Zeitschriftenartikel, Konferenzbeiträge, Forschungsberichte, etc.)
- **Structure**: Standardized academic format with title page, abstract, author affiliations

## Pipeline Stages

### 1. Extract PDF Metadata
Extracts embedded metadata from PDF files (title, author, creation date, etc.)

### 2. Extract Content
Extracts full text content from PDFs using PyMuPDF

### 3. Filter Files by Content
Filters out files with insufficient or poor quality content

### 4. Generate Embeddings
Creates vector embeddings using Jina embeddings model for RAG (Retrieval-Augmented Generation)
- **Chunk size**: 800 characters
- **Chunk overlap**: 150 characters
- **Embedding model**: jina/jina-embeddings-v2-base-de

### 5. AI Metadata Extraction
Extracts structured metadata using LLM (llama3:70b):
- **Author names**: Extracted and validated with specialized LLM-based name parser
- **Title**: Document title
- **Document type**: Classification (article, conference paper, thesis, etc.)
- **Affiliations**: Institutional affiliations from title page
- **Keywords**:
  - `ai:keywords_ext`: Keywords from document (author-provided or extracted)
  - `ai:keywords_gen`: Generated descriptive keywords for cataloging
  - `ai:keywords_dnb`: Controlled vocabulary keywords (GND, DDC)
- **Dewey Classification**: Up to 3 DDC classifications
- **Summary**: 3-sentence summary in German

#### Retrieval Configuration
- **max_retrieval_chunks**: 10 (limits context to reduce hallucinations)
- **retrieval_strategy**: "all" (can be changed to "first_and_last_pages" or "first_pages_only")

### 6. GND Keyword Check
Validates and enriches keywords against GND (German National Library)
- LLM-based semantic matching with Wikidata
- Uses document context for disambiguation

### 7. Document Similarity
Calculates document similarity using vector embeddings

## Configuration Files

### `config/full.yaml`
Complete pipeline configuration for all 1,010 PDFs
- Collection: `local_pdfs_collection`
- ChromaDB: `chroma_db_local`

### `config/test.yaml`
Test configuration for first 100 PDFs
- Collection: `local_pdfs_test_collection`
- ChromaDB: `chroma_db_local_test`

## Prompts

### `prompts/prompts_scientific_papers.yaml`
Specialized prompts optimized for scientific papers:
- **Structured metadata**: Prompts tuned for academic paper format
- **Author extraction**: Handles multiple authors, affiliations with markers
- **Keywords**: Distinguishes between author-provided and technical keywords
- **Summary**: Focus on research question, methods, findings

## Scripts

### `scripts/prepare_local_pdfs.py`
Utility script for preparing the local PDF collection

## Data Paths

**Input**: `/media/sz/Data/Veits_pdfs/data_pipeline/local_pdfs/raw/files/`
**Output**: `/media/sz/Data/Veits_pdfs/data_pipeline/local_pdfs/processed/`

## Running the Pipeline

### Test Run (100 PDFs)
```bash
cd /media/sz/Data/Veits_pdfs/Data_aggregation/cl-pipeline/run
python -m pipeline.run pipelines/local_pdfs/config/test.yaml
```

### Full Run (1,010 PDFs)
```bash
cd /media/sz/Data/Veits_pdfs/Data_aggregation/cl-pipeline/run
python -m pipeline.run pipelines/local_pdfs/config/full.yaml
```

## Key Features

### Intelligent Retrieval Strategies
Three strategies for selecting document chunks:
1. **all**: First N chunks (default, backwards compatible)
2. **first_and_last_pages**: Chunks from first and last pages (solves "author on last page" problem)
3. **first_pages_only**: Only first 1-2 pages (optimal for metadata extraction)

### Response Filtering
Automatically filters unwanted LLM output:
- Removes verbose phrases ("Here is", "Based on", etc.)
- Filters short unhelpful responses ("No", "Yes", "Ja")
- Cleans up formatting artifacts

### Name Parsing
Specialized LLM-based name parser (gemma3:27b):
- Handles single-word names (treats as last name)
- Processes compound names (von Goethe, van der Meer)
- Validates against blacklist
- Extracts titles (Dr., Prof.)

## Processing Mode

The pipeline uses a hybrid processing approach:

**Force Processing** (always overwrite):
- ai:affiliation
- ai:dewey
- ai:author

**Conditional Processing** (only if empty):
- ai:keywords_gen
- ai:title
- ai:type
- ai:keywords_ext
- ai:keywords_dnb
- ai:summary

## Common Issues

### Low Quality Metadata
**Solution**: Ensure prompts are loaded correctly from `prompts/prompts_scientific_papers.yaml`

### Author Hallucinations
**Solution**:
1. Increase `max_retrieval_chunks` if author info is spread across pages
2. Use `retrieval_strategy: "first_and_last_pages"` if authors appear at end

### Language Detection Errors
**Check**: `pipe:language` field is set during metadata extraction stage

## Logging

### Log File Location

Pipeline logs are automatically saved to:
- **Test pipeline**: `pipelines/local_pdfs/logs/test_<timestamp>.log`
- **Full pipeline**: `pipelines/local_pdfs/logs/full_<timestamp>.log`

Each pipeline run creates a new timestamped log file for easy tracking.

### Log Rotation

- **Max file size**: 10 MB
- **Backup files**: 5 (keeps last 5 rotations)
- When a log reaches 10 MB, it's rotated to `<filename>.log.1`, `<filename>.log.2`, etc.

### Viewing Logs

```bash
# View latest test run
ls -t pipelines/local_pdfs/logs/test_*.log | head -1 | xargs tail -f

# View all logs
cat pipelines/local_pdfs/logs/test_*.log

# Filter for errors
grep ERROR pipelines/local_pdfs/logs/test_*.log

# Filter for specific stage
grep "AIMetaDataExtraction" pipelines/local_pdfs/logs/test_*.log
```

### Log Format

Each log entry contains:
- Timestamp
- Logger name (module/class)
- Line number
- Log level (DEBUG/INFO/WARNING/ERROR)
- Thread name
- Message

Example:
```
2025-12-03 12:32:26,409 - root (105) - INFO - MainThread - Starting pipeline
```

## BibTeX Export

After the pipeline completes, you can export the AI-extracted metadata to BibTeX format for use in bibliography management tools.

### Usage

```bash
cd pipelines/local_pdfs/scripts

# Export AI metadata only
python bibtex_export.py --input ../../../raw/LOCAL_ai_meta.p --output output.bib

# Export with original BibTeX data merged (recommended)
python bibtex_export.py \
  --input /media/sz/Data/Veits_pdfs/data_pipeline/local_pdfs/raw/LOCAL_ai_meta.p \
  --bibtex-source /media/sz/Data/Veits_pdfs/data/raw_data/Artikelbasis.bib \
  --output output_with_original.bib

# Or use config to auto-detect paths
python bibtex_export.py --config ../config/full.yaml --output output.bib
```

### BibTeX Fields

When exporting with `--bibtex-source`, the output includes:

**Original fields** (from source BibTeX):
- **author**: Original author field
- **title**: Original title
- **year**: Publication year
- **journal/booktitle**: Publication venue
- **volume, number, pages**: Citation details
- **doi, url**: Digital identifiers
- **keywords**: Original keywords
- **abstract**: Original abstract
- **language**: Document language

**AI-extracted fields** (with `_ai` suffix):
- **author_ai**: AI-extracted author names
- **title_ai**: AI-extracted title
- **keywords_ai**: Combined AI-generated keywords (from all keyword fields)
- **summary_ai**: AI-generated 3-sentence summary
- **dewey_ai**: Dewey Decimal Classification with labels
- **type_ai**: Document type (Zeitschriftenartikel, Konferenzbeitrag, etc.)
- **affiliation_ai**: Institutional affiliation

### Example BibTeX Entry (with original data)

```bibtex
@article{pdf_7972,
  author = {Müller, Hans and Schmidt, Anna},
  title = {Machine Learning in Library Science: A Survey},
  year = {2023},
  journal = {Journal of Library and Information Science},
  volume = {45},
  number = {3},
  pages = {123--145},
  doi = {10.1234/jlis.2023.7972},
  keywords = {Machine Learning, Libraries},
  abstract = {This paper surveys machine learning applications in library science...},
  language = {de},
  % --- AI-extracted metadata below ---
  author_ai = {Müller, Hans and Schmidt, Anna},
  title_ai = {Machine Learning in der Bibliothekswissenschaft},
  keywords_ai = {Machine Learning, Bibliothekswissenschaft, Künstliche Intelligenz, Informationsextraktion, Metadatenextraktion},
  summary_ai = {Diese Arbeit untersucht den Einsatz von Machine Learning in Bibliotheken. Es werden verschiedene Ansätze zur automatischen Metadatenextraktion vorgestellt. Die Ergebnisse zeigen vielversprechende Anwendungsmöglichkeiten.},
  dewey_ai = {020 (Bibliotheks- und Informationswissenschaften), 006.3 (Künstliche Intelligenz)},
  type_ai = {Zeitschriftenartikel},
  affiliation_ai = {Universität Leipzig},
  file = {7972.pdf},
  file_id = {7972}
}
```

### Integration with Reference Managers

The exported BibTeX file can be imported into:
- **JabRef**: Open source reference manager
- **Zotero**: Import via BibTeX
- **Mendeley**: Import via BibTeX
- **BibDesk** (macOS): Native BibTeX support

### Script Features

- ✅ Sanitizes special characters for BibTeX compatibility
- ✅ Combines keywords from multiple fields (ext, gen, dnb)
- ✅ Formats Dewey classifications with labels
- ✅ Auto-detects entry type based on document type
- ✅ Removes duplicates while preserving order
- ✅ UTF-8 encoding support

## Dependencies

- LangChain
- Ollama (llama3:70b, gemma3:27b)
- ChromaDB
- PyMuPDF
- Jina Embeddings (jina-embeddings-v2-base-de)
