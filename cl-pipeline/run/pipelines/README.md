# Pipelines Directory

This directory contains organized configurations for different data processing pipelines. Each pipeline is self-contained with its own configuration files, prompts, and scripts.

## Directory Structure

```
pipelines/
├── local_pdfs/              # Scientific papers pipeline
│   ├── config/
│   │   ├── full.yaml        # Complete pipeline (1,010 PDFs)
│   │   └── test.yaml        # Test pipeline (100 PDFs)
│   ├── prompts/
│   │   └── prompts_scientific_papers.yaml
│   ├── scripts/
│   │   └── prepare_local_pdfs.py
│   └── README.md
│
├── opal/                    # OPAL OER materials pipeline
│   ├── config/
│   │   └── full.yaml        # Complete OPAL pipeline
│   ├── prompts/
│   │   └── prompts.yaml
│   ├── scripts/
│   └── README.md
│
├── shared/                  # Shared resources
│   ├── dewey_classification.txt
│   └── README.md
│
└── README.md               # This file
```

## Available Pipelines

### 1. Local PDFs (`local_pdfs/`)
Processes scientific research papers from a local collection.

**Key Features**:
- 1,010 scientific papers (97.2% German, 2.8% English)
- Optimized prompts for academic paper structure
- Intelligent retrieval strategies for metadata extraction
- Specialized name parsing for author extraction

**Use Cases**:
- Academic paper cataloging
- Research metadata extraction
- Institutional repositories

[→ See local_pdfs/README.md for details](local_pdfs/README.md)

### 2. OPAL (`opal/`)
Processes Open Educational Resources from Bildungsportal Sachsen.

**Key Features**:
- Multiple file formats (PDF, PPTX, DOCX, XLSX, MD)
- Downloads from OPAL web API
- Educational materials metadata
- Course and lecture content processing

**Use Cases**:
- OER cataloging
- Educational content discovery
- Course material management

[→ See opal/README.md for details](opal/README.md)

## Running a Pipeline

### General Command
```bash
cd /media/sz/Data/Veits_pdfs/Data_aggregation/cl-pipeline/run
python -m pipeline.run pipelines/<pipeline_name>/config/<config_file>.yaml
```

### Examples
```bash
# Local PDFs test run
python -m pipeline.run pipelines/local_pdfs/config/test.yaml

# Local PDFs full run
python -m pipeline.run pipelines/local_pdfs/config/full.yaml

# OPAL pipeline
python -m pipeline.run pipelines/opal/config/full.yaml
```

## Common Pipeline Stages

Most pipelines share these core stages:

1. **Metadata Extraction** - Extract embedded file metadata
2. **Content Extraction** - Extract text from files
3. **Content Filtering** - Filter low-quality content
4. **Embeddings Generation** - Create vector embeddings for RAG
5. **AI Metadata Extraction** - LLM-based metadata extraction
6. **GND Keyword Check** - Validate against German National Library
7. **Document Similarity** - Calculate similarity scores

## Shared Components

### Code (`../../stages/`)
- `stages/general/` - General-purpose stages
- `stages/opal/` - OPAL-specific stages
- `src/ai_metadata_core/` - Modular AI metadata processors

### Resources (`shared/`)
- `dewey_classification.txt` - DDC classification reference

### Configuration Patterns
All pipelines use consistent YAML configuration structure:
```yaml
folder_structure:
  data_root_folder: /path/to/data
  raw_data_folder: ...
  processed_data_folder: ...

stages_module_path:
  - ../../../stages/general/
  - ../../../stages/opal/

stages:
  - name: Stage Name
    class: StageClass
    parameters: ...
```

## Creating a New Pipeline

1. **Create directory structure**:
   ```bash
   mkdir -p pipelines/new_pipeline/{config,prompts,scripts}
   ```

2. **Add configuration**:
   - Copy a similar pipeline's config as template
   - Update paths to point to:
     - `../prompts/` for prompt files
     - `../../shared/` for shared resources
     - `../../../stages/` for stage classes

3. **Create prompts**:
   - Add pipeline-specific prompts in `prompts/`
   - Or reuse existing prompts from `shared/`

4. **Add README.md**:
   - Document pipeline purpose
   - List stages
   - Explain configuration options
   - Provide running instructions

5. **Update this README**:
   - Add to "Available Pipelines" section

## Path Conventions

### Stage Modules
From config file perspective:
```yaml
stages_module_path:
  - ../../../stages/general/    # Three levels up, then stages/
  - ../../../stages/opal/
```

### Prompts
From config file perspective:
```yaml
prompts_file_name: ../prompts/prompts.yaml  # One level up, then prompts/
```

### Shared Resources
From config file perspective:
```yaml
resource_file: ../../shared/resource.txt  # Two levels up, then shared/
```

## Dependencies

All pipelines require:
- Python 3.8+
- LangChain
- Ollama (with required models)
- ChromaDB
- Various file processors (PyMuPDF, python-pptx, etc.)

See individual pipeline READMEs for specific requirements.

## Best Practices

### 1. Isolation
- Each pipeline is self-contained
- No cross-pipeline dependencies (except shared resources)
- Independent configuration and prompts

### 2. Naming Conventions
- Config files: `full.yaml`, `test.yaml`, `<variant>.yaml`
- Prompts: `prompts.yaml` or `prompts_<specialization>.yaml`
- Scripts: Descriptive names (e.g., `prepare_local_pdfs.py`)

### 3. Documentation
- Every pipeline has a README.md
- Document all configuration parameters
- Explain pipeline-specific features

### 4. Testing
- Provide test configurations with small datasets
- Test before running full pipelines
- Validate output quality

## Troubleshooting

### Path Resolution Errors
**Issue**: Stage modules or resources not found
**Solution**:
- Check relative paths in config
- Verify `stages_module_path` has correct depth (`../../../stages/`)
- Ensure prompt/resource paths match directory structure

### Configuration Conflicts
**Issue**: Multiple pipelines running simultaneously
**Solution**:
- Use different ChromaDB collections
- Separate output files
- Independent data folders

### Prompt Loading Errors
**Issue**: Prompts not applied correctly
**Solution**:
- Verify prompt file path in config
- Check YAML syntax in prompt file
- Ensure PromptManager loads correct file

## Migration Notes

This structure was introduced to organize previously flat configuration files:

**Old Structure** (deprecated):
```
run/
├── cl_local_pdfs.yaml
├── cl_local_pdfs_test.yaml
├── cl_opal.yaml
├── prompts.yaml
├── prompts_scientific_papers.yaml
└── dewey_classification.txt
```

**New Structure**:
```
run/pipelines/
├── local_pdfs/config/
├── opal/config/
└── shared/
```

Old configuration files remain in `run/` for backwards compatibility but should be considered deprecated.
