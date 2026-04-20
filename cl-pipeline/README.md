# Connected Lecturers Pipeline

A modular data aggregation and metadata extraction pipeline for processing educational resources and scientific publications. The pipeline uses AI-based techniques to extract, enrich, and standardize metadata across multiple document sources.

## Overview

This project provides a flexible framework for processing different types of educational and scientific content:

- **OPAL Pipeline**: Processes Open Educational Resources (OER) from the Saxon education portal
- **LiaScript Pipeline**: Discovers and analyzes interactive educational content on GitHub
- **Local PDFs Pipeline**: Extracts metadata from scientific research papers

Each pipeline is independently configurable and can be run separately or combined for comprehensive metadata aggregation.

## Project Structure

```
cl-pipeline/
├── run/                        # Pipeline execution and configurations
│   └── pipelines/
│       ├── opal/              # OPAL pipeline (OER from Bildungsportal Sachsen)
│       ├── liascript/         # LiaScript pipeline (GitHub repositories)
│       ├── local_pdfs/        # Local PDFs pipeline (scientific papers)
│       └── shared/            # Shared resources (Dewey classification, etc.)
├── scripts/                    # Utility scripts
│   ├── migrate_error_tracking.py  # Initialize error tracking for existing datasets
│   └── monitor_dgx.sh            # Live GPU monitoring for DGX
├── src/                        # Core pipeline framework and utilities
│   ├── ai_metadata_core/      # AI metadata extraction framework
│   └── pipeline_logging/      # Logging infrastructure
└── stages/                     # Pipeline stage implementations
    ├── general/               # General-purpose stages
    ├── opal/                  # OPAL-specific stages
    └── liascript/             # LiaScript-specific stages
```

## Available Pipelines

### 1. OPAL Pipeline

**Purpose**: Process Open Educational Resources from OPAL (Bildungsportal Sachsen)

**Key Features**:
- Multi-format support (PDF, PPTX, DOCX, XLSX, Markdown)
- Three-level metadata extraction (original, file-embedded, AI-extracted)
- OER metadata schema mapping
- Stratified sampling for testing

**Documentation**: [run/pipelines/opal/README.md](run/pipelines/opal/README.md)

**Quick Start**:
```bash
cd run
pipenv run python3 run_pipeline.py -c pipelines/opal/config/test.yaml
```

### 2. LiaScript Pipeline

**Purpose**: Discover and analyze LiaScript-based educational content on GitHub

**Key Features**:
- GitHub API integration for repository discovery
- Commit history tracking
- LiaScript metadata extraction from markdown headers
- Feature usage analysis (quizzes, videos, animations, etc.)
- License tracking (both repository and content licenses)

**Documentation**: [run/pipelines/liascript/README.md](run/pipelines/liascript/README.md)

**Quick Start**:
```bash
cd run
pipenv run python3 run_pipeline.py -c pipelines/liascript/config/full.yaml
```

### 3. Local PDFs Pipeline

**Purpose**: Extract comprehensive metadata from scientific research papers

**Key Features**:
- Specialized for academic papers (97% German, 3% English)
- AI-based author name parsing with validation
- Dewey Decimal Classification
- GND keyword validation and enrichment
- Multiple retrieval strategies for metadata extraction
- BibTeX export functionality

**Documentation**: [run/pipelines/local_pdfs/README.md](run/pipelines/local_pdfs/README.md)

**Quick Start**:
```bash
cd run
python -m pipeline.run pipelines/local_pdfs/config/test.yaml
```

## Core Features

### AI-Powered Metadata Extraction

All pipelines use Large Language Models (LLMs) for intelligent metadata extraction:
- **Primary model**: llama3.3:70b (metadata extraction)
- **Name parsing**: gemma3:27b (author name validation)
- **Embeddings**: jina-embeddings-v2-base-de (German-optimized)

**Session 5 Enhancement**: Raw extracted data is now stored separately (`*_raw` columns) and validated in a dedicated verification stage, separating extraction from validation concerns.

### Metadata Schema Standardization

Extracted metadata is mapped to standardized schemas:
- **OER materials**: [LOM for Higher Education OER Repositories](https://dini-ag-kim.github.io/hs-oer-lom-profil/latest/)
- **Scientific papers**: Custom schema with BibTeX export
- **LiaScript content**: LiaScript metadata specification

### Multi-Level Processing

Each pipeline implements a staged processing architecture:
1. **Data Collection**: Download or identify source materials
2. **Content Extraction**: Extract text and embedded metadata
3. **Quality Filtering**: Remove low-quality or incomplete documents
4. **Embedding Generation**: Create vector embeddings for RAG
5. **AI Extraction**: Extract structured metadata using LLMs (stores raw data with `*_raw` suffix)
6. **AI Verification**: Consolidate and validate names/affiliations from multiple sources (Session 5+)
7. **Validation & Enrichment**: Validate keywords against GND, calculate similarity

**Note**: Since Session 5, the pipeline separates data extraction (step 5) from validation (step 6), improving maintainability and allowing reprocessing without recomputation.

### Flexible Configuration

- **YAML-based configuration**: Each pipeline has `test.yaml` and `full.yaml` configs
- **Force/conditional processing**: Control which stages always run vs. skip if data exists
- **Error tracking**: Per-field error counting with configurable `max_error_retries` to avoid endless retries on failing documents
- **Stratified sampling**: Balanced file type distribution for testing
- **Logging**: Timestamped log files with rotation

## Recent Enhancements (Session 5 - 2026-04-08)

### Multi-Source Name & Affiliation Handling
A new unified architecture consolidates author and affiliation data from multiple sources with priority-based conflict resolution:

- **Raw Data Storage**: All extracted data stored in `*_raw` columns (unprocessed)
- **Multi-Source Priority**: OPAL (user-entered) > AI (LLM-extracted) > File (metadata)
- **Unified Validation**: Dedicated `AIVerificationStage` validates and normalizes all sources
- **Source Tracking**: Output columns record which source was used for each field
- **No Data Loss**: Existing data preserved, reprocessing without recomputation

**Example Data Flow**:
```
OPAL:known_creator → opal:author_raw ────┐
File:author        → file:author_raw      ├─→ AIVerificationStage → ai:author_final
LLM extraction     → ai:author_raw  ─────┘                          ai:author_source
```

### Error Tracking & Resilience
- Per-field error tracking with configurable `max_error_retries`
- Persistent LLM errors logged to `ai:_errors` column
- Failed documents don't block entire pipeline
- Graceful degradation to empty values for permanently failed fields

---

### Testing a Pipeline

All pipelines provide test configurations for quick validation:

```bash
cd run
pipenv run python3 run_pipeline.py -c pipelines/<pipeline-name>/config/test.yaml
```

Test configurations use:
- Limited datasets (100 PDFs, 20 OPAL files)
- Separate test databases
- Force run on all stages
- Faster execution

### Running Production Pipelines

```bash
cd run
pipenv run python3 run_pipeline.py -c pipelines/<pipeline-name>/config/full.yaml
```

Production configurations:
- Process complete datasets
- Production databases
- Conditional processing (skip completed stages)
- Comprehensive logging

### Viewing Logs

```bash
# View latest log for a pipeline
ls -t pipelines/<pipeline-name>/logs/*.log | head -1 | xargs tail -f

# Search for errors
grep ERROR pipelines/<pipeline-name>/logs/*.log

# Filter by stage
grep "AIMetaDataExtraction" pipelines/<pipeline-name>/logs/*.log
```

## Dependencies

### Core Dependencies
- **Python**: 3.8+
- **Pipenv**: Virtual environment and dependency management
- **LangChain**: LLM integration framework
- **Ollama**: Local LLM inference (llama3.3:70b, gemma3:27b)
- **ChromaDB**: Vector database for embeddings

### Format-Specific Dependencies
- **PyMuPDF**: PDF processing
- **python-pptx**: PowerPoint processing
- **python-docx**: Word document processing
- **openpyxl**: Excel spreadsheet processing
- **Jina Embeddings**: German-optimized embeddings

### External APIs
- **GitHub API**: LiaScript repository discovery
- **GND/Wikidata**: Keyword validation and enrichment

## Installation

```bash
# Install dependencies
cd cl-pipeline
pipenv install

# Verify Ollama models are available
ollama list

# Required models:
# - llama3.3:70b
# - gemma3:27b
```

## Environment Setup

The pipeline requires certain paths to be configured in the YAML config files:

- **OPAL**: `/media/sz/Data/Connected_Lecturers/Opal/`
- **LiaScript**: `/media/sz/Data/Connected_Lecturers/LiaScript/`
- **Local PDFs**: `/media/sz/Data/Veits_pdfs/data_pipeline/local_pdfs/`

Adjust these paths in the respective `config/*.yaml` files to match your environment.

## Development

### Adding a New Pipeline

1. Create pipeline directory: `run/pipelines/<new-pipeline>/`
2. Add configuration files: `config/test.yaml`, `config/full.yaml`
3. Create pipeline-specific stages in `stages/<new-pipeline>/`
4. Add README documenting the pipeline: `run/pipelines/<new-pipeline>/README.md`
5. Update this README with the new pipeline information

### Creating Custom Stages

Stages are modular Python classes that implement specific processing steps. See existing stages in `stages/` for examples.

### Shared Resources

Place resources used by multiple pipelines in `run/pipelines/shared/`:
- Dewey Decimal Classification
- Controlled vocabularies
- Common prompts

See [run/pipelines/shared/README.md](run/pipelines/shared/README.md) for details.

## Troubleshooting

### Common Issues

**LLM Connection Errors**:
- Verify Ollama is running: `ollama list`
- Check model availability: `ollama pull llama3.3:70b`
- Persistent LLM errors are tracked per field in `ai:_errors` and skipped after `max_error_retries` attempts

**ChromaDB Errors**:
- Clear test database: Delete ChromaDB directory and rerun
- Check disk space for embeddings storage

**Download Failures** (OPAL):
- Verify network connectivity to bildungsportal.sachsen.de
- Pipeline includes retry logic for transient failures

**GitHub API Rate Limits** (LiaScript):
- Authenticate with GitHub token for higher limits
- Pipeline includes rate limit handling

### Debug Mode

Enable debug logging in config files:
```yaml
logging:
  level: DEBUG
```

## Contributing

When contributing:
1. Test changes with both test and full configurations
2. Update relevant README files
3. Ensure shared resources remain backwards compatible
4. Add logging for new stages

## License

[Specify project license here]

## Contact

[Add contact information or links to project documentation]
