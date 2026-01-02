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

### 6. Analyze LiaScript Feature Usage
Statistically analyzes which LiaScript features are used across all ~2700 documents:
- **Imported templates**: Counts and lists all imported LiaScript templates (from `import:` in header)
- **Videos**: Detects video embeds using `!?[` syntax (YouTube, Vimeo, etc.)
- **Audio**: Detects audio embeds using `?[` syntax
- **Webapps**: Identifies interactive webapps using `??[` syntax (SketchFab, CircuitJS, etc.)
- **Tables**: Counts all markdown tables and specifically LiaScript visualization tables (with `data-type` annotations like heatmap, barchart, etc.)
- **Quizzes**: Detects different quiz types:
  - Text input quizzes: `[[solution]]`
  - Single choice: `[( )]` and `[(X)]`
  - Multiple choice: `[[ ]]` and `[[X]]`
  - Quiz hints: `[[?]]`
- **Code blocks**: Counts fenced code blocks with ` ``` ` syntax
- **Script tags**: Finds `<script>` tags for JavaScript execution
- **TTS narrator**: Checks for narrator configuration and `--{{number}}--` comment syntax
- **Animation fragments**: Detects `{{number}}` fragment indicators
- **Animate.css**: Identifies animate.css animation classes
- **Images**: Counts standard markdown images `![]()`
- **External resources**: Scripts and CSS stylesheets from header
- **QR codes**: Detects `[qr-code]()` syntax
- **Preview links**: Finds `[preview-lia]()` syntax
- **Math formulas**: Counts inline `$...$` and display `$$...$$` math
- Stores per-document features in `LiaScript_features.p`
- Generates aggregate statistics in `LiaScript_feature_statistics.p` and `.txt` format

### 7. Extract Content from Markdown Files
Extracts full-text content from LiaScript markdown files for AI processing

### 8. Provide Embeddings
Generates vector embeddings for RAG (Retrieval-Augmented Generation):
- **Embedding model**: jina/jina-embeddings-v2-base-de
- **Vector store**: ChromaDB
- Enables content-based retrieval for AI metadata extraction

### 9. Extract AI Metadata from Content
Uses LLM (llama3.3:70b) to extract structured metadata through RAG:
- **Title**: AI-extracted course title
- **Summary**: 3-sentence course summary in German
- **Keywords**:
  - `ai:keywords_ext`: 15 extracted keywords from course content
  - `ai:keywords_gen`: 15 generated descriptive keywords
- **Dewey Classification**: Up to 3 DDC classifications with confidence scores
- **Educational Level**: Classification of target education level
  - Categories: Frühkindliche Bildung, Primarstufe, Sekundarstufe I, Sekundarstufe II, Berufsbildung, Hochschulbildung, Wissenschaftliche Forschung, Weiterbildung, Nicht spezifisch
  - Based on content complexity, language, exercises, and prerequisites
- **Target Audience**: Detailed 1-2 sentence description of intended learners
  - Includes expected knowledge level, age group, educational context
  - Example: "Informatik-Studierende im Bachelor-Studium mit Grundkenntnissen in Programmierung"
- Stores results in `LiaScript_ai_meta.p`

#### Educational Level Classification

The AI analyzes each course to determine the appropriate educational level by examining:
- **Content complexity and difficulty**: Technical depth and conceptual sophistication
- **Language and terminology**: Academic vs. simplified language
- **Exercise types**: Complexity of tasks and quizzes
- **Prerequisites**: Required prior knowledge
- **Explicit target group mentions**: Direct statements about intended audience

**Educational Level Categories**:
1. **Frühkindliche Bildung** - Kindergarten, preschool (ages 3-6)
2. **Primarstufe** - Primary school, grades 1-4 (ages 6-10)
3. **Sekundarstufe I** - Secondary school, grades 5-10 (ages 10-16)
4. **Sekundarstufe II** - Upper secondary, grades 11-13, Abitur level (ages 16-19)
5. **Berufsbildung** - Vocational training, professional qualification
6. **Hochschulbildung** - Higher education (Bachelor, Master)
7. **Wissenschaftliche Forschung** - Academic research, doctoral level
8. **Weiterbildung** - Adult education, professional development
9. **Nicht spezifisch** - General education, suitable for multiple audiences

This classification enables targeted filtering and recommendation of educational materials based on learner level and context.

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

### AI-Generated Metadata (`LiaScript_ai_meta.p`)
| Field | Type | Description |
|-------|------|-------------|
| `pipe:ID` | str | Unique file identifier (matches LiaScript_files.p) |
| `ai:title` | str | AI-extracted course title |
| `ai:summary` | str | 3-sentence course summary |
| `ai:dewey` | list[dict] | Dewey Decimal Classifications with notation, label, and score |
| `ai:keywords_ext` | str | 15 extracted keywords from course content |
| `ai:keywords_gen` | str | 15 generated descriptive keywords |
| `ai:education_level` | str | Primary educational level (e.g., "Hochschulbildung", "Sekundarstufe I") |
| `ai:target_audience` | str | Detailed target audience description (1-2 sentences) |

### Feature Analysis (`LiaScript_features.p`)
| Field | Type | Description |
|-------|------|-------------|
| `pipe:ID` | str | Unique file identifier (matches LiaScript_files.p) |
| `repo_name` | str | Repository name |
| `repo_user` | str | Repository owner |
| `file_name` | str | File name |
| `feature:video_count` | int | Number of video embeds (`!?[` syntax) |
| `feature:has_video` | bool | Document contains videos |
| `feature:audio_count` | int | Number of audio embeds (`?[` syntax) |
| `feature:has_audio` | bool | Document contains audio |
| `feature:webapp_count` | int | Number of webapp embeds (`??[` syntax) |
| `feature:has_webapp` | bool | Document contains webapps |
| `feature:table_count` | int | Number of markdown tables (table headers) |
| `feature:has_tables` | bool | Document contains tables |
| `feature:lia_viz_table_count` | int | Number of LiaScript visualization tables (with `data-type` annotations) |
| `feature:has_lia_viz_tables` | bool | Document contains LiaScript visualization tables |
| `feature:text_quiz_count` | int | Number of text input quizzes (`[[solution]]`) |
| `feature:has_text_quiz` | bool | Document contains text input quizzes |
| `feature:single_choice_count` | int | Number of single choice questions (`[( )]`/`[(X)]`) |
| `feature:has_single_choice` | bool | Document contains single choice quizzes |
| `feature:multiple_choice_count` | int | Number of multiple choice questions (`[[ ]]`/`[[X]]`) |
| `feature:has_multiple_choice` | bool | Document contains multiple choice quizzes |
| `feature:quiz_hint_count` | int | Number of quiz hints (`[[?]]`) |
| `feature:has_quiz_hints` | bool | Document contains quiz hints |
| `feature:total_quiz_count` | int | Total number of all quiz elements |
| `feature:has_quiz` | bool | Document contains any type of quiz |
| `feature:code_block_count` | int | Number of fenced code blocks |
| `feature:has_code_blocks` | bool | Document has code blocks |
| `feature:script_tag_count` | int | Number of `<script>` tags |
| `feature:has_script_tags` | bool | Document has script tags |
| `feature:import_count` | int | Number of imported templates |
| `feature:has_imports` | bool | Document imports templates |
| `feature:imported_templates` | list[str] | List of template URLs |
| `feature:has_narrator` | bool | Text-to-speech narrator configured |
| `feature:tts_comment_count` | int | Number of TTS narrator comments (`--{{n}}--`) |
| `feature:has_tts_comments` | bool | TTS comments present |
| `feature:animation_fragment_count` | int | Number of animation fragments (`{{n}}`) |
| `feature:has_animation_fragments` | bool | Animation fragments used |
| `feature:animated_css_count` | int | Number of animate.css animations |
| `feature:has_animated_css` | bool | Animate.css used |
| `feature:image_count` | int | Number of images (`![]()`syntax) |
| `feature:has_images` | bool | Document contains images |
| `feature:external_script_count` | int | Number of external scripts in header |
| `feature:has_external_scripts` | bool | External scripts loaded |
| `feature:external_css_count` | int | Number of external CSS links in header |
| `feature:has_external_css` | bool | External CSS loaded |
| `feature:qr_code_count` | int | Number of QR codes |
| `feature:has_qr_codes` | bool | QR codes present |
| `feature:preview_lia_count` | int | Number of preview-lia links |
| `feature:has_preview_lia` | bool | Preview-lia links present |
| `feature:inline_math_count` | int | Number of inline math formulas (`$...$`) |
| `feature:display_math_count` | int | Number of display math formulas (`$$...$$`) |
| `feature:has_math` | bool | Math formulas present |

### Feature Statistics (`LiaScript_feature_statistics.p` / `.txt`)
Aggregated statistics across all documents:
- **Total documents analyzed**
- **Feature usage counts and percentages** for each feature type
- **Template usage ranking** showing most popular imported templates
- **Summary statistics** including most common features

The `.txt` file provides a human-readable format for quick review.

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

## Visualization Tools

### Commit Statistics Visualization

The pipeline includes a powerful visualization tool for analyzing commit patterns across LiaScript documents:

**Script**: [scripts/visualize_commits.py](scripts/visualize_commits.py)

This script creates an interactive scatter plot showing:
- **X-axis**: Duration in days (logarithmic scale) - how long the document has been maintained
- **Y-axis**: Number of commits (logarithmic scale) - how actively the document is updated
- **Color**: Number of authors contributing to each document
- **Size**: Indicates grouped documents at the same position (same commit count and duration)
- **Interactive**: Click on single documents to open their GitHub URL, hover for detailed information

**Features**:
- Aggregates overlapping points to avoid clutter
- Shows author lists and repository information on hover
- Generates both online (CDN-based, smaller) and offline (embedded, larger) HTML versions
- Provides statistics summary of all documents

**Usage**:
```bash
cd /media/sz/Data/Veits_pdfs/Data_aggregation/cl-pipeline/run/pipelines/liascript/scripts
python visualize_commits.py
```

**Output**:
- `liascript_commits_visualization.html` - Lightweight version (requires internet)
- `liascript_commits_visualization_offline.html` - Standalone version (works offline)

The visualization helps identify:
- Long-term maintained educational materials
- Actively updated courses
- Single-author vs. collaborative projects
- Document lifecycle patterns

## Common Issues

### GitHub API Rate Limits
**Issue**: API rate limits exceeded during large crawls
**Solution**: Configure GitHub API token in `.env` file for higher rate limits

### Repository Access
**Issue**: Cannot access private repositories
**Solution**: Ensure GitHub token has appropriate permissions
