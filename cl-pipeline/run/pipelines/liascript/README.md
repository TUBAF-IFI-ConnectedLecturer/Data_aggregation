# LiaScript Pipeline

This pipeline identifies and analyzes LiaScript repositories and educational content on GitHub.

## Overview

The LiaScript pipeline crawls GitHub to discover LiaScript-based educational materials, aggregates files and commits, and extracts metadata for analysis and cataloging.

### Document Characteristics
- **Source**: GitHub repositories
- **File Types**: Markdown files with LiaScript syntax
- **Content**: Interactive educational materials, courses, tutorials
- **Language**: Multi-language (primarily English and German)

## Pipeline Stages

### 1. Generate Data Folder Structure
Creates necessary directory structure for data processing

### 2. Identify LiaScript Repositories
Crawls GitHub to find repositories containing LiaScript content
- Uses GitHub API to search for LiaScript files
- Collects repository metadata
- Stores results in `LiaScript_repositories.p`

### 3. Aggregate LiaScript Files
Collects all LiaScript files from identified repositories
- Parses repository structure
- Identifies Markdown files with LiaScript syntax
- Stores file metadata in `LiaScript_files.p`

### 4. Aggregate LiaScript Commits
Gathers commit history for LiaScript files
- Tracks content evolution
- Identifies contributors
- Stores commit data in `LiaScript_commits.p`

### 5. Extract LiaScript Metadata
Extracts structured metadata from LiaScript files:
- Course titles
- Authors
- Topics
- Learning objectives
- Interactive elements

## Configuration Files

### `config/full.yaml`
Complete pipeline configuration for LiaScript collection
- Data path: `/media/sz/Data/Connected_Lecturers/LiaScript`
- Processes all discovered repositories

## Data Paths

**Base**: `/media/sz/Data/Connected_Lecturers/LiaScript/`
- **Raw data**: `raw/`
- **Files**: `raw/files/`
- **Processed**: `processed/`

## Running the Pipeline

```bash
cd /media/sz/Data/Veits_pdfs/Data_aggregation/cl-pipeline/run
pipenv run python3 run_pipeline.py -c pipelines/liascript/config/full.yaml
```

## Logging

Pipeline logs are automatically saved to:
- **Full pipeline**: `pipelines/liascript/logs/full_<timestamp>.log`

Each pipeline run creates a new timestamped log file for easy tracking.

### Viewing Logs

```bash
# View latest run
ls -t pipelines/liascript/logs/full_*.log | head -1 | xargs tail -f

# Filter for errors
grep ERROR pipelines/liascript/logs/*.log
```

## Dependencies

- GitHub API access (GITHUB_API_KEY required)
- Python libraries for GitHub interaction
- Markdown parsers

## Common Issues

### GitHub API Rate Limits
**Issue**: API rate limits exceeded during large crawls
**Solution**: Configure GitHub API token in `.env` file for higher rate limits

### Repository Access
**Issue**: Cannot access private repositories
**Solution**: Ensure GitHub token has appropriate permissions
