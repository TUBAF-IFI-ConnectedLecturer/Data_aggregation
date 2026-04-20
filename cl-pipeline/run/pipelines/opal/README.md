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
| `creator`          | Author              | `opal:author_raw`       |
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

**Note**: Since Session 5 refactoring, all raw extracted metadata uses the `*_raw` suffix for consistency.

| Office Files     | PDF Files      | CL-Naming            | Notes                                |
| ---------------- | -------------- | -------------------- | ------------------------------------ |
| `creator`        | `author`       | `file:author_raw`    | Raw author from file metadata        |
| `title`          | `title`        | `file:title`         | Document title                      |
| `description`    |                |                      |                                    |
| `subject`        | `subject`      | `file:subject`       | Document subject/description       |
| `identifier`     |                |                      |                                    |
| `language`       | `language`     | `file:language`      | Document language code             |
| `created`        | `creationDate` | `file:created`       | Creation date                      |
| `modified`       | `modDate`      | `file:modified`      | Last modification date             |
| `lastModifiedBy` |                |                      |                                    |
| `category`       |                |                      |                                    |
| `contentStatus`  |                |                      |                                    |
| `version`        |                |                      |                                    |
| `revision`       |                |                      |                                    |
| `keywords`       |                | `file:keywords`      | File embedded keywords             |
| `lastPrinted`    |                |                      |                                    |
|                  | `creator`      |                      |                                    |
|                  | `producer`     |                      |                                    |
|                  | `format`       |                      |                                    |

### AI-Extracted Metadata

**Note**: Since Session 5 refactoring, raw LLM-extracted data is stored separately from validated data.

| CL-Naming (Raw)       | CL-Naming (Validated) | Content                          | Prompt                                                                                                        |
| --------------------- | -------------------- | -------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| `ai:author_raw`       | `ai:author_final`    | Extracted author names           | `f"Who is the author of the document {file}. Avoid all additional information, just answer by authors name."` |
| `ai:affiliation_raw`  | `ai:affiliation_final`| Extracted affiliation           | (Custom affiliation extraction prompt)                                                                       |
| `ai:title`            |                      | Extracted title                  | `f"Give me a title of the document {file}. Just answer by the title. Please answer in German."`              |
| `ai:keywords`         |                      | Extracted 5 keywords             | `f"Please extract 5 Keywords from {file}? Just answer by a list separated by commas. Please answer in German."` |

**Raw vs. Validated Data**:
- **Raw columns** (`*_raw`): Unprocessed LLM output, stored by AI Metadata Extraction stage
- **Validated columns** (`*_final`): Processed and validated by AIVerificationStage
- Example: `ai:author_raw` (raw string) → `ai:author_final` (validated Name objects)

### OER Metadata Schema Mapping

The structure from [LOM for Higher Education OER Repositories](https://dini-ag-kim.github.io/hs-oer-lom-profil/latest/) has been flattened here. The final schema will be determined during the project runtime and a transformation script will be integrated.

| Field Name           | Notes                                          | `pipe:`          | `opal:`              | `file:`         | `ai:`                    |
| -------------------- | ---------------------------------------------- | ---------------- | -------------------- | --------------- | ------------------------ |
| `<title>`            |                                                |                  | `opal:title`         | `file:title`    | `ai:title`               |
| `<language>`         |                                                | `pipe:language`  | `opal:language`      | `file:language` |                          |
| `<description>`      |                                                |                  | `opal:comment`       | `file:subject`  |                          |
| `<keyword>`          |                                                |                  |                      | `file:keywords` | `ai:keywords`            |
| `<aggregationlevel>` | For individual, atomic materials (1)           | 1                |                      |                 |                          |
| `<format>`           | e.g. application/pdf or image/png              | `pipe:file_type` |                      |                 |                          |
| `<location>`         | Usually a Uniform Resource Locator (URL)       |                  | `opal:oer_permalink` |                 |                          |
| `<rights>`           | License parameters                             |                  | `opal:license`       |                 |                          |
| `<author>`           | Raw → Validated conversion in AIVerificationStage | |                  | `opal:author_raw` (raw) | `file:author_raw` (raw) | `ai:author_raw` (raw) → `ai:author_final` (validated) |
| `<date>`             |                                                |                  |                      | `file:modified` |                          |

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
- Author names (raw, to `ai:author_raw`)
- Title
- Document type
- Affiliations (raw, to `ai:affiliation_raw`)
- Keywords (extracted, generated, controlled vocabulary)
- Dewey Classification
- Summary

**Raw Data Storage**: Since Session 5, extraction outputs are stored in `*_raw` columns (unprocessed LLM responses).

### 10. AI Verification & Merge (NEW - Session 5)
Consolidates and validates names/affiliations from multiple sources with priority-based conflict resolution:

**Priority order**: OPAL > AI > File

**Input sources**:
- `opal:author_raw` - From OPAL metadata (user-entered, highest quality)
- `ai:author_raw`, `ai:affiliation_raw` - From LLM extraction
- `file:author_raw` - From document metadata (lowest priority)

**Output**:
- `ai:author_final` - Validated author names (List of Name objects)
- `ai:author_source` - Which source was selected ('opal', 'ai', 'file', or None)
- `ai:affiliation_final` - Normalized affiliation string
- `ai:affiliation_source` - Which source was selected ('ai' or None)
- `ai:_errors` - Validation errors if any

This stage separates data extraction from validation, improving maintainability and allowing reprocessing without recomputation.

### 11. GND Keyword Check
Validates keywords against GND (German National Library)

### 12. Document Similarity
Calculates similarity between documents using embeddings

## Multi-Source Name & Affiliation Handling

Since **Session 5** (2026-04-08), the pipeline implements a unified multi-source data fusion architecture for handling names and affiliations:

### Architecture Overview

```
Three Data Sources (Raw Collection)
├── OPAL:known_creator (user-entered, highest quality)
├── File metadata author (PDF/DOCX properties)
└── LLM extraction (AI-generated)
         │
         ↓
All sources stored with *_raw suffix
├── opal:author_raw
├── file:author_raw
└── ai:author_raw, ai:affiliation_raw
         │
         ↓
AIVerificationStage (Priority-based merge)
├── Check OPAL → Use if available
├── Else check AI → Use if available
└── Else check File → Use if available
         │
         ↓
Validated Output
├── ai:author_final (validated names)
├── ai:author_source (which source)
├── ai:affiliation_final (normalized)
└── ai:affiliation_source (which source)
```

### Why This Design?

**Separation of Concerns**:
- Extraction (LLM steps) separate from validation (verification step)
- No data loss - raw data preserved for reprocessing
- Validation can be updated without recomputing extraction

**Priority-Based Resolution**:
1. **OPAL** (highest): Manually entered by course creators, most accurate
2. **AI**: LLM-extracted, good coverage but variable quality
3. **File** (lowest): Document metadata, often incomplete but available

**Quality Control**:
- Each source labeled with provenance (`ai:author_source`, `ai:affiliation_source`)
- Validation errors tracked in `ai:_errors` column
- Can detect and resolve conflicts between sources

### Backward Compatibility

✅ **100% Backward Compatible**
- Old column names preserved (`opal:known_creator`, `file:author`)
- All other stages unchanged
- Can skip AIVerificationStage if not needed
- No breaking changes to existing pipelines

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

The AI metadata extraction stage supports three processing modes, configured in the YAML config under `processing_mode`:

**Force Processing** (`force_processing`): Fields that are always reprocessed, overwriting existing data.

**Conditional Processing** (`conditional_processing`): Fields that are only processed if empty/missing. Filled fields are skipped.

**Error Tracking** (`max_error_retries`): Maximum number of LLM errors (timeouts, connection failures) before a field is permanently skipped for a document. Set to `0` for unlimited retries (old behavior), recommended default is `3`.

The pipeline distinguishes between:
- **LLM errors** (Ollama timeout, connection failure) → tracked in `ai:_errors`, retried up to `max_error_retries` times
- **Legitimate empty results** (LLM returns no data) → accepted as final, field marked as processed

```yaml
processing_mode:
  force_processing:
    - ai:affiliation
    - ai:dewey
  conditional_processing:
    - ai:author
    - ai:keywords_gen
    - ai:title
    - ai:type
  allow_skip_when_all_conditional_filled: false
  max_error_retries: 3
```

For existing datasets with stuck documents, use the migration script to initialize error tracking:
```bash
python cl-pipeline/scripts/migrate_error_tracking.py <pickle_path> --max-retries 3 --dry-run
```

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
