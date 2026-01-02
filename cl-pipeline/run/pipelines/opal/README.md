# OPAL Pipeline

This pipeline processes Open Educational Resources (OER) documents from the OPAL platform (Bildungsportal Sachsen).

## Overview

The OPAL pipeline collects, processes, and enriches educational materials from the Saxon education portal. It handles multiple file formats and extracts comprehensive metadata for cataloging and discovery.

### Document Characteristics
- **Source**: https://bildungsportal.sachsen.de/opal/oer/
- **File Types**: PDF, PPTX, DOCX, XLSX, Markdown
- **Content**: Educational materials, lectures, course materials
- **Language**: Primarily German

## Metadata Extraction

Metadata extraction for OPAL files is performed on three levels:

| Level | Source                                                    | Prefix in Dataset |
| ----- | --------------------------------------------------------- | ----------------- |
| 1     | Original metadata from UB Freiberg JSON dataset           | `opal`            |
| 2     | Metadata/Properties embedded in the files                 | `file`            |
| 3     | AI-extracted metadata from file contents                  | `ai`              |

The final dataset combines metadata from all three levels and maps them to the comprehensive [OER Metadata Schema](https://dini-ag-kim.github.io/hs-oer-lom-profil/latest/).

### Pipeline Metadata for Each Dataset

| Meaning         | CL-Naming             |
| --------------- | --------------------- |
| ID              | `pipe:ID`             |
| Folder          | `pipe:file_path`      |
| File Type       | `pipe:file_type`      |
| Language        | `pipe:language`       |
| Error           | `pipe:error_download` |
| Download Date   | `pipe:download_date`  |

### Original OPAL Metadata

| OPAL Label         | OPAL Meaning        | CL-Naming               |
| ------------------ | ------------------- | ----------------------- |
| `filename`         | Filename            | `opal:filename`         |
| `license`          | License             | `opal:license`          |
| `oer_permalink`    | Permalink           | `opal:oer_permalink`    |
| `title`            | Title               | `opal:title`            |
| `comment`          | Description         | `opal:comment`          |
| `creator`          | Author              | `opal:creator`          |
| `publisher`        | Publisher           |                         |
| `source`           | Source              |                         |
| `city`             | City                |                         |
| `publicationMonth` | Publication Month   | `opal:publicationMonth` |
| `publicationYear`  | Publication Year    | `opal:publicationYear`  |
| `pages`            | Pages               |                         |
| `language`         | Language            | `opal:language`         |
| `url`              | Link / URL          |                         |
| `act`              | Work                |                         |
| `appId`            | Project             |                         |
| `category`         | Category            |                         |
| `chapter`          | Chapter             |                         |
| `duration`         | Duration            |                         |
| `mediaType`        | Media Type          |                         |
| `nav1`             | ?                   |                         |
| `nav2`             | ?                   |                         |
| `nav3`             | ?                   |                         |
| `series`           | Series              |                         |

### File Metadata

Metadata is extracted from `docx`, `pptx`, `xlsx` and `pdf` files.

| Office Files     | PDF Files      | CL-Naming       |
| ---------------- | -------------- | --------------- |
| `creator`        | `author`       | `file:author`   |
| `title`          | `title`        | `file:title`    |
| `description`    |                |                 |
| `subject`        | `subject`      | `file:subject`  |
| `identifier`     |                |                 |
| `language`       | `language`     | `file:language` |
| `created`        | `creationDate` | `file:created`  |
| `modified`       | `modDate`      | `file:modified` |
| `lastModifiedBy` |                |                 |
| `category`       |                |                 |
| `contentStatus`  |                |                 |
| `version`        |                |                 |
| `revision`       |                |                 |
| `keywords`       |                | `file:keywords` |
| `lastPrinted`    |                |                 |
|                  | `creator`      |                 |
|                  | `producer`     |                 |
|                  | `format`       |                 |

### AI-Extracted Metadata

| CL-Naming     | Prompt                                                                                                        |
| ------------- | ------------------------------------------------------------------------------------------------------------- |
| `ai:title`    | `f"Give me a title of the document {file}. Just answer by the title. Please answer in German."`              |
| `ai:author`   | `f"Who is the author of the document {file}. Avoid all additional information, just answer by authors name."` |
| `ai:keywords` | `f"Please extract 5 Keywords from {file}? Just answer by a list separated by commas. Please answer in German.` |

### OER Metadata Schema Mapping

The structure from [LOM for Higher Education OER Repositories](https://dini-ag-kim.github.io/hs-oer-lom-profil/latest/) has been flattened here. The final schema will be determined during the project runtime and a transformation script will be integrated.

| Field Name           | Notes                                          | `pipe:`          | `opal:`              | `file:`         | `ai:`         |
| -------------------- | ---------------------------------------------- | ---------------- | -------------------- | --------------- | ------------- |
| `<title>`            |                                                |                  | `opal:title`         | `file:title`    | `ai:title`    |
| `<language>`         |                                                | `pipe:language`  | `opal:language`      | `file:language` |               |
| `<description>`      |                                                |                  | `opal:comment`       | `file:subject`  |               |
| `<keyword>`          |                                                |                  |                      | `file:keywords` | `ai:keywords` |
| `<aggregationlevel>` | For individual, atomic materials (1)           | 1                |                      |                 |               |
| `<format>`           | e.g. application/pdf or image/png              | `pipe:file_type` |                      |                 |               |
| `<location>`         | Usually a Uniform Resource Locator (URL)       |                  | `opal:oer_permalink` |                 |               |
| `<rights>`           | License parameters                             |                  | `opal:license`       |                 |               |
| `<author>`           |                                                |                  | `opal:creator`       | `file:author`   | `ai:author`   |
| `<date>`             |                                                |                  |                      | `file:modified` |               |

## Pipeline Stages

### 1. Generate Data Folder Structure
Creates necessary directory structure for data processing

### 2. Collect Raw Data from OPAL
Fetches OER documents from OPAL JSON endpoint
- Downloads metadata from `content.json`
- Stores repository information

### 3. Aggregate Basic Features
Preprocesses OPAL metadata and extracts basic features

### 4. Download Files
Downloads OER files from OPAL platform
- Supported formats: PPTX, PDF, DOCX, XLSX, MD
- **Stratified sampling**: Optional `max_downloads_per_type` parameter
- **Total limit**: Optional `max_total_downloads` parameter
- Enables balanced file type distribution for testing

### 5. Extract Metadata
Extracts embedded metadata from files (varies by file type)

### 6. Extract Content
Extracts text content from various file formats:
- PDF: PyMuPDF
- PPTX/DOCX: python-pptx, python-docx
- XLSX: openpyxl
- MD: Plain text

### 7. Filter Files by Content
Filters out files with insufficient content quality

### 8. Generate Embeddings
Creates vector embeddings for RAG
- **Embedding model**: jina/jina-embeddings-v2-base-de
- **Collection**: oer_connected_lecturer

### 9. AI Metadata Extraction
Extracts structured metadata using LLM (llama3.3:70b):
- Author names
- Title
- Document type
- Affiliations
- Keywords (extracted, generated, controlled vocabulary)
- Dewey Classification
- Summary

### 10. GND Keyword Check
Validates keywords against GND (German National Library)

### 11. Document Similarity
Calculates similarity between documents using embeddings

## Configuration Files

### `config/full.yaml`
Complete pipeline configuration for OPAL collection
- Collection: `oer_connected_lecturer`
- ChromaDB: `chroma_db`
- Data path: `/media/sz/Data/Connected_Lecturers/Opal`
- Processes all available OPAL files

### `config/test.yaml`
Test configuration for development and testing
- Collection: `oer_connected_lecturer_test`
- ChromaDB: `chroma_db_test`
- Data path: `/media/sz/Data/Connected_Lecturers/Opal_test`
- **Limited dataset**: Max 20 downloads total
- **Stratified sampling**: 5 files per type (PDF, PPTX, DOCX, XLSX, MD)
- **Mixed file types**: Balanced mix for representative testing
- **Force run**: All stages set to force_run for clean testing

## Prompts

### `prompts/prompts.yaml`
Standard prompts for educational materials:
- Optimized for diverse document types (presentations, documents, spreadsheets)
- Handles mixed content (educational materials vs. administrative documents)
- Flexible keyword extraction for varied content

## Data Paths

**Base**: `/media/sz/Data/Connected_Lecturers/Opal/`
- **Raw data**: `raw/`
- **Files**: `raw/files/`
- **Content**: `raw/content/`
- **Processed**: `processed/`

## Running the Pipeline

### Test Run (Recommended for first time)
```bash
cd /media/sz/Data/Veits_pdfs/Data_aggregation/cl-pipeline/run
pipenv run python3 run_pipeline.py -c pipelines/opal/config/test.yaml
```

**Test run features**:
- Downloads max 20 files with stratified sampling (5 per file type)
- Mixed file types: PDF, PPTX, DOCX, XLSX, MD
- Separate test database and output files
- Logs to: `pipelines/opal/logs/test_<timestamp>.log`
- Fast execution for testing changes

### Full Pipeline
```bash
cd /media/sz/Data/Veits_pdfs/Data_aggregation/cl-pipeline/run
pipenv run python3 run_pipeline.py -c pipelines/opal/config/full.yaml
```

**Full run features**:
- Processes all available OPAL files
- Multiple file types (PDF, PPTX, DOCX, XLSX, MD)
- Logs to: `pipelines/opal/logs/full_<timestamp>.log`
- Production database and output files

## Key Differences from Local PDFs Pipeline

### Multi-Format Support
Unlike the local PDFs pipeline which processes only PDFs, OPAL handles:
- PPTX (PowerPoint presentations)
- DOCX (Word documents)
- XLSX (Excel spreadsheets)
- MD (Markdown files)
- PDF (documents)

### Content Extraction Strategy
Different extractors for each file type:
- Text-based for DOCX, MD
- Slide-based for PPTX
- Cell-based for XLSX
- Page-based for PDF

### Prompts
OPAL uses `prompts.yaml` (generic educational materials) vs. Local PDFs uses `prompts_scientific_papers.yaml` (academic papers)

### Data Source
- **OPAL**: Downloads from web API
- **Local PDFs**: Processes existing local files

## Processing Mode

**Force Processing**: (always overwrite)
- Configurable per stage

**Conditional Processing**: (only if empty)
- Most metadata fields

## Logging

### Log File Location

Pipeline logs are automatically saved to:
- **Test pipeline**: `pipelines/opal/logs/test_<timestamp>.log`
- **Full pipeline**: `pipelines/opal/logs/full_<timestamp>.log`

Each pipeline run creates a new timestamped log file for easy tracking.

### Log Rotation

- **Max file size**: 10 MB
- **Backup files**: 5 (keeps last 5 rotations)
- When a log reaches 10 MB, it's rotated to `<filename>.log.1`, `<filename>.log.2`, etc.

### Viewing Logs

```bash
# View latest test run
ls -t pipelines/opal/logs/test_*.log | head -1 | xargs tail -f

# View latest full run
ls -t pipelines/opal/logs/full_*.log | head -1 | xargs tail -f

# Filter for errors
grep ERROR pipelines/opal/logs/*.log

# Filter for specific stage
grep "AIMetaDataExtraction" pipelines/opal/logs/*.log
```

## Dependencies

- LangChain
- Ollama (llama3.3:70b, gemma3:27b)
- ChromaDB
- PyMuPDF (PDF)
- python-pptx (PowerPoint)
- python-docx (Word)
- openpyxl (Excel)
- Jina Embeddings

## Common Issues

### Download Failures
**Check**: Network connectivity to bildungsportal.sachsen.de
**Solution**: Pipeline includes retry logic for failed downloads

### Format-Specific Extraction Errors
**PDF**: Check PyMuPDF installation
**PPTX/DOCX**: Check python-pptx/python-docx versions
**XLSX**: Verify openpyxl compatibility

### Mixed Content Types
**Issue**: Educational materials vary widely in structure
**Solution**: Prompts are designed to be flexible and handle diverse content
