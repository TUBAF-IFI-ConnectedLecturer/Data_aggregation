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
- Collects repository metadata including:
  - Repository name, owner, URL
  - Creation/update dates
  - Stars, forks, watchers
  - **License information** (SPDX ID and full name)
- Stores results in `LiaScript_repositories.p`

### 3. Aggregate LiaScript Files
Collects all LiaScript files from identified repositories
- Parses repository structure
- Identifies Markdown files with LiaScript syntax
- Stores file metadata including:
  - File name, download URL, HTML URL
  - Repository information
  - **Repository license** (SPDX ID and full name)
  - LiaScript-specific indicators
- Stores file metadata in `LiaScript_files.p`

### 4. Aggregate LiaScript Commits
Gathers commit history for LiaScript files
- Tracks content evolution
- Identifies contributors
- Stores commit data in `LiaScript_commits.p`

### 5. Extract LiaScript Metadata
Extracts structured metadata from LiaScript markdown headers:
- **Author information**: author, email
- **Content metadata**: version, language, narrator, comment (multi-line support)
- **Visual elements**: icon, logo URLs
- **External resources**: import, link, script (multi-line support)
- **Translation**: translation references
- **License**: content license from markdown (CC-BY, MIT, etc.) and license URLs
- Stores metadata in `LiaScript_metadata.p`

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

## Data Structure

### Repository Data (`LiaScript_repositories.p`)
| Field | Type | Description |
|-------|------|-------------|
| `user` | str | Repository owner/organization |
| `name` | str | Repository name |
| `repo_url` | str | GitHub repository URL |
| `created_at` | datetime | Repository creation date |
| `updated_at` | datetime | Last update date |
| `stars` | int | Number of stars |
| `forks` | int | Number of forks |
| `watchers` | int | Number of watchers |
| `contributors_per_repo` | int | Total number of contributors |
| `license_spdx` | str | SPDX license identifier (e.g., "MIT", "Apache-2.0", "CC-BY-4.0") |
| `license_name` | str | Full license name (e.g., "MIT License") |

### File Data (`LiaScript_files.p`)
| Field | Type | Description |
|-------|------|-------------|
| `pipe:ID` | str | Unique file identifier (hash) |
| `pipe:file_type` | str | File type (always "md") |
| `repo_name` | str | Repository name |
| `repo_user` | str | Repository owner |
| `repo_url` | str | Repository URL |
| `repo_license_spdx` | str | Repository's SPDX license ID |
| `repo_license_name` | str | Repository's full license name |
| `file_name` | str | Markdown file name |
| `file_download_url` | str | Direct download URL |
| `file_html_url` | str | GitHub web view URL |
| `liaIndi_*` | bool | LiaScript indicator flags |

### Metadata (`LiaScript_metadata.p`)
| Field | Type | Description |
|-------|------|-------------|
| `pipe:ID` | str | Unique file identifier (matches LiaScript_files.p) |
| `lia:author` | str | Course author name |
| `lia:email` | str | Author email address |
| `lia:version` | str | Course version (e.g., "1.0.0") |
| `lia:language` | str | Content language code (e.g., "de", "en", "PT-BR") |
| `lia:narrator` | str | Text-to-speech narrator voice |
| `lia:comment` | list[str] | Course description/comments (multi-line support) |
| `lia:icon` | str | Icon/logo file path or URL |
| `lia:logo` | str | Logo file path or URL |
| `lia:import` | list[str] | Imported LiaScript templates/macros |
| `lia:link` | list[str] | External CSS stylesheets |
| `lia:script` | list[str] | External JavaScript libraries |
| `lia:translation` | str | Translation file reference |
| `lia:content_license` | str | License from content (e.g., "CC-BY", "MIT") |
| `lia:content_license_url` | str | URL to license terms |

### License Information
The pipeline automatically captures license information for each repository:
- **Open licenses**: MIT, Apache-2.0, GPL-*, BSD-*, CC-BY-*, ISC, LGPL-*
- **No license**: `None` if repository has no license file
- **SPDX identifiers**: Standardized license identifiers for easy filtering

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
