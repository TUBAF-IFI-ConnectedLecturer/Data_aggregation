# LiaScript Community Analysis - Data Documentation

**Generated:** 2026-01-03
**Last updated:** 2026-03-15
**Location:** `/media/sz/Data/Connected_Lecturers/liascript_march_2026/raw/`
**Pipeline:** LiaScript Community Analysis Pipeline
**Purpose:** Research data for analyzing LiaScript usage in the international educational community

---

## Overview

This folder contains data from a systematic analysis of the LiaScript community on GitHub. The pipeline identifies, validates, and analyzes LiaScript courses to understand:
- Thematic focus areas and content characteristics
- Collaboration patterns and authorship
- Usage patterns (custom JavaScript, templates, interactive features)
- Geographic and linguistic distribution

### Data Collection Process

The pipeline consists of twelve stages:

1. **Repository Discovery** ([searchForLia.py](../../stages/liascript/searchForLia.py))
   - GitHub API search for repositories containing LiaScript content
   - Identifies 1,076 repositories with potential LiaScript files

2. **File Aggregation** ([aggregateLia.py](../../stages/liascript/aggregateLia.py))
   - Extracts all markdown files from identified repositories
   - Applies heuristic indicators to detect LiaScript-specific features
   - Generates unique identifiers (pipe:ID) for each file

3. **AI Validation** ([validateLiaScript.py](../../stages/liascript/validateLiaScript.py))
   - Uses LLM (llama3.3:70b) to validate whether files are genuine LiaScript courses
   - Filters out false positives from heuristic detection
   - Result: 3,672 validated LiaScript courses from 57,096 markdown files

4. **Commit History Analysis** ([aggregateLiaCommits.py](../../stages/liascript/aggregateLiaCommits.py))
   - Extracts Git commit history for each validated file via GitHub API
   - Identifies actual committers (`contributors_list`), commit counts, and temporal activity
   - Uses URL-decoded file paths and generic branch stripping for reliable API queries
   - Committer data is used for the author-based cluster analysis (replacing `repo_user`)

5. **Metadata Extraction** ([extractLiaScriptMetadata.py](../../stages/liascript/extractLiaScriptMetadata.py))
   - Parses LiaScript header block (HTML comment at file beginning)
   - Extracts author, title, language, narrator settings, imports, etc.

6. **Feature Analysis** ([analyzeLiaScriptFeatures.py](../../stages/liascript/analyzeLiaScriptFeatures.py))
   - Analyzes usage of LiaScript-specific features (quizzes, videos, code blocks, etc.)
   - Tracks template imports and usage patterns
   - Generates feature statistics for the entire corpus

7. **Feature Cluster Analysis** ([analyzeFeatureClusters.py](../../stages/liascript/analyzeFeatureClusters.py))
   - Assigns behavioral clusters to documents (template_power_user, minimalist, etc.)
   - Committer-based author analysis using `contributors_list` from commit data
   - K-Means clustering of committers by feature + template usage vectors
   - Output: `LiaScript_clusters.p`

8. **Content Extraction** (General stage)
   - Full text extraction and language detection
   - Word counts and content hashing for deduplication

9. **Embeddings Generation** (AIEmbeddingsGeneration)
   - Creates vector embeddings for semantic search
   - Stored in ChromaDB for similarity queries

10. **AI Metadata Extraction** (AIMetaDataExtraction)
    - LLM-based analysis for keywords, classification, summaries
    - Dewey Decimal Classification assignment
    - Education level and target audience classification

11. **GitHub User Profile Collection** ([collectGithubUserProfiles.py](../../stages/liascript/collectGithubUserProfiles.py))
    - Fetches GitHub profile metadata for all repository owners
    - Enables geographic and institutional affiliation analysis
    - Output: `LiaScript_user_profiles.p`

12. **Data Consolidation** ([mergeLiaScriptData.py](../../stages/liascript/mergeLiaScriptData.py))
    - Joins all per-file datasets (validated, metadata, features, commits) into one consolidated DataFrame
    - Simplifies downstream analysis with a single wide-format table
    - Output: `LiaScript_consolidated.p`

### Internal Accounts

During initial dataset construction, repositories from known LiaScript core developers and institutions were **excluded** to avoid bias from the creators themselves. In a later extension step ([add_internal_repos.py](../../add_internal_repos.py)), these repositories were added back with an `internal = True` flag, enabling comparative analysis between the core community and external adopters.

**Internal accounts (8):**
| Account | Type | Role |
|---------|------|------|
| `SebastianZug` | User | LiaScript co-creator (TU Bergakademie Freiberg) |
| `andre-dietrich` | User | LiaScript co-creator |
| `LiaScript` | Organization | Official LiaScript organization |
| `LiaBooks` | Organization | Official LiaScript book templates |
| `LiaTemplates` | Organization | Official LiaScript template library |
| `LiaPlayground` | Organization | LiaScript playground examples |
| `TUBAF-IfI-LiaScript` | Organization | TU Bergakademie Freiberg – Computer Science |
| `TUBAF-IUZ-LiaScript` | Organization | TU Bergakademie Freiberg – IUZ |

**Analytical relevance:** The `internal` boolean field (available in `LiaScript_repositories.p` and propagated through `searched_type = "internal_account"`) allows filtering for:
- `internal == True`: official / expert usage patterns
- `internal == False`: community / external adoption patterns

### Key Identifiers

All datasets can be linked using these identifiers:

- **`pipe:ID`** (PRIMARY KEY): SHA256 hash of file's download URL (first 16 characters)
  - Unique across all files
  - Stable across pipeline runs
  - Use this for joining datasets

- **`file_download_url`**: GitHub raw content URL (unique per file)

- **`id`**: Legacy hash identifier (kept for backwards compatibility)

---

## Datasets

### 1. LiaScript_repositories.p

**Purpose:** Repository-level metadata for all GitHub repositories containing LiaScript content
**Rows:** 1,076 | **Columns:** 21

#### Columns

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `user` | string | GitHub username or organization | "LiaScript", "TUBAF-IfI-LiaScript" |
| `name` | string | Repository name | "docs", "VL_ProzeduraleProgrammierung" |
| `repo_url` | string | Full repository URL | "https://github.com/LiaScript/docs" |
| `description` | string | Repository description text | "Interactive LiaScript course on..." |
| `created_at` | datetime | Repository creation timestamp | 2018-03-15 |
| `updated_at` | datetime | Last repository update timestamp | 2024-12-20 |
| `stars` | integer | GitHub star count | 42 |
| `forks` | integer | Number of repository forks | 5 |
| `is_fork` | boolean | Whether repository is itself a fork | True/False |
| `watchers` | integer | Number of repository watchers | 12 |
| `subscribers` | integer | Number of repository subscribers (notification watchers) | 3 |
| `contributors_per_repo` | integer | Total contributors to repository | 3 |
| `license_spdx` | string | Repository license SPDX identifier | "CC-BY-4.0" |
| `license_name` | string | Repository license name | "Creative Commons Attribution 4.0" |
| `default_branch` | string | Default branch name | "main", "master" |
| `size_kb` | integer | Repository size in kilobytes | 1024 |
| `open_issues_count` | integer | Number of open issues | 3 |
| `topics` | list[str] | GitHub repository topics/tags | ["liascript", "oer", "education"] |
| `languages` | dict | Programming languages used (bytes) | {"JavaScript": 1234, "HTML": 567} |
| `searched_type` | string | How repository was discovered | "code", "repositories" |
| `internal` | boolean | Whether from known LiaScript organizations | True/False |

Note: The `validity` column was removed in this dataset version.

**Analysis Use:**
- Repository popularity and adoption metrics
- Institutional vs. individual usage
- Temporal growth patterns
- Fork/reuse behavior

---

### 2. LiaScript_files_validated.p

**Purpose:** All markdown files with LiaScript indicators and AI validation results
**Rows:** 57,096 | **Columns:** 29

**Important:** Only 3,672 files (6.4%) are validated as genuine LiaScript courses (`pipe:is_valid_liascript == True`)

#### Core Fields

| Column | Type | Description |
|--------|------|-------------|
| `pipe:ID` | string | Unique file identifier (PRIMARY KEY) |
| `pipe:file_type` | string | Always "md" for markdown |
| `id` | integer | Legacy identifier |
| `repo_name` | string | Repository name |
| `repo_user` | string | Repository owner |
| `repo_url` | string | Repository URL |
| `file_name` | string | Original filename |
| `file_download_url` | string | GitHub raw download URL |
| `file_html_url` | string | GitHub web view URL |
| `repo_license_spdx` | string | Repository license SPDX identifier |
| `repo_license_name` | string | Repository license name |

#### New Repository Context Fields (added in v1.5)

| Column | Type | Description |
|--------|------|-------------|
| `description` | string | Repository description text |
| `topics` | list[str] | GitHub repository topics |
| `languages` | dict | Programming languages used |
| `default_branch` | string | Default branch name |
| `size_kb` | integer | Repository size in KB |
| `open_issues_count` | integer | Number of open issues |

#### LiaScript Indicator Flags (Boolean)

These heuristic indicators detect LiaScript-specific features:

| Column | Description | Strength |
|--------|-------------|----------|
| `liaIndi_Lia_in_filename` | "liascript" in filename | Weak |
| `liaIndi_liascript_in_content` | "liascript" mentioned in content | Weak |
| `liaIndi_lia_button` | LiaScript course button/badge present | Strong |
| `liaIndi_comment_in_beginning` | File starts with `<!--` comment | Moderate |
| `liaIndi_import_statement` | `import:` in header (module system) | Very Strong |
| `liaIndi_narrator_statement` | `narrator:` in header (TTS config) | Strong |
| `liaIndi_version_statement` | `version:` in header | Moderate |
| `liaIndi_video_syntax` | `!?[` video embedding syntax | Very Strong |
| `liaIndi_lia_in_h1` | "liascript" in main heading | Moderate |
| `liaIndi_liaTemplates_used` | Uses LiaScript template system | Strong |

#### Validation Fields

| Column | Type | Description |
|--------|------|-------------|
| `pipe:is_valid_liascript` | boolean | **AI validation result - use this for filtering!** |
| `pipe:validation_method` | string | Validation method: "ai" or "heuristic" |

**Analysis Use:**
- Filter for validated courses: `df[df['pipe:is_valid_liascript'] == True]`
- Feature usage analysis (which LiaScript features are most common?)
- False positive analysis (why were files flagged but not validated?)

---

### 3. LiaScript_commits.p

**Purpose:** Git commit history and collaboration metrics for validated LiaScript files
**Rows:** 3,672 | **Columns:** 35

This dataset contains all columns from `LiaScript_files_validated.p` plus collaboration metrics.

#### Additional Collaboration Fields

| Column | Type | Description | Use |
|--------|------|-------------|-----|
| `contributors_list` | list | List of GitHub usernames who contributed | Network analysis |
| `author_count` | integer | Number of unique contributors | Collaboration intensity |
| `commit_count` | integer | Total commits to this file | Activity/maintenance metric |
| `first_commit` | datetime | First commit timestamp | File age |
| `last_commit` | datetime | Most recent commit timestamp | Activity recency |
| `commit_hist_extracted` | boolean | Whether extraction succeeded | Data quality flag |

**Statistics:**
- Files with commit data: 3,669 (99.9%)
- Single-author files: 3,138 (85.5%)
- Collaborative files (>1 author): 531 (14.5%)
- Files without commit data: 3 (0.1%)
- Unique committers: 373 (including `unknown`, `Copilot`, `github-actions[bot]`)
- Total commits tracked: 21,356
- Commit extraction uses URL-decoded file paths and generic branch stripping (not limited to main/master)

**Analysis Use:**
- Collaborative vs. individual authorship patterns
- Repository activity over time
- Abandoned vs. actively maintained courses
- Contribution network analysis

---

### 4. LiaScript_metadata.p

**Purpose:** Comprehensive metadata extracted from LiaScript header blocks
**Rows:** 3,297 | **Columns:** 34

LiaScript files typically start with an HTML comment block containing metadata:
```markdown
<!--
author: Jane Doe
email: jane@example.com
version: 1.0.0
language: en
narrator: US English Female
comment: This is a comprehensive introduction to Python programming
         with interactive examples and quizzes.
import: https://github.com/LiaTemplates/...
link: https://cdn.example.com/styles.css
script: https://cdn.example.com/custom.js
tags: Python, Programming, Interactive, OER
-->
```

#### System Fields

| Column | Type | Description |
|--------|------|-------------|
| `pipe:ID` | string | File identifier (PRIMARY KEY) - links to all other datasets |
| `id` | string | Legacy file identifier (kept for backwards compatibility) |

#### Author Information

| Column | Type | Coverage | Description | Example |
|--------|------|----------|-------------|---------|
| `lia:author` | str | 97.7% | Course author name | "Jane Doe" |
| `lia:email` | str | 97.7% | Author contact email | "jane@example.edu" |

#### Content & Version Information

| Column | Type | Coverage | Description | Example |
|--------|------|----------|-------------|---------|
| `lia:version` | str | 97.7% | Course version number | "1.0.0", "2.3.1" |
| `lia:language` | str | 97.7% | Content language code (ISO 639-1 or custom) | "de", "en", "PT-BR" |
| `lia:narrator` | str | 97.7% | Text-to-speech narrator voice | "US English Female", "Deutsch Female" |
| `lia:mode` | str | ~5% | Display mode settings | "Presentation", "Textbook" |
| `lia:translation` | str | Rare | Translation file reference | "translations/de.md" |
| `lia:attribute` | str | Rare | Additional attributes | Custom LiaScript attributes |

#### Visual Elements

| Column | Type | Coverage | Description | Example |
|--------|------|----------|-------------|---------|
| `lia:icon` | str | 11.6% | Icon file path or URL | "img/icon.png", "https://..." |
| `lia:logo` | str | 30.3% | Logo file path or URL | "logo.jpg", "https://cdn.../logo.svg" |

#### External Resources (Multi-line Support)

| Column | Type | Coverage | Description | Example |
|--------|------|----------|-------------|---------|
| `lia:import` | list[str] | 30.3% | Imported LiaScript templates/macros | ["https://github.com/LiaTemplates/AVR8js", ...] |
| `lia:link` | list[str] | 27.9% | External CSS stylesheets | ["https://cdn.../style.css", ...] |
| `lia:script` | list[str] | 27.9% | External JavaScript libraries | ["https://cdn.../custom.js", ...] |

#### Categorization & Discovery

| Column | Type | Coverage | Description | Example |
|--------|------|----------|-------------|---------|
| `lia:tags` | list[str] | 50.6% | Content tags/keywords (multi-line) | ["Python", "Programming", "Interactive", "OER"] |
| `lia:title` | str | 11.6% | Course title | "Introduction to Python Programming" |
| `lia:date` | str | 11.6% | Publication/modification date | "2024-01-15", "15/01/2024" |
| `lia:comment` | list[str] | 97.7% | Course description/comments (multi-line) | ["This is a comprehensive introduction..."] |

#### License Information

| Column | Type | Coverage | Description | Example |
|--------|------|----------|-------------|---------|
| `lia:content_license` | str | ~16% | License extracted from markdown content | "CC-BY", "CC-BY-SA", "MIT", "GPL-3.0" |
| `lia:content_license_url` | str | ~16% | URL to license terms | "https://creativecommons.org/licenses/by/4.0/" |

**Note:** License information comes from two sources:
- **Repository license** (from GitHub API): Available for ALL files in `LiaScript_files_validated.p`
- **Content license** (from markdown): Extracted when explicitly mentioned in file content

#### Module Metadata

These fields are used in structured course collections:

| Column | Type | Coverage | Description | Example |
|--------|------|----------|-------------|---------|
| `lia:module_id` | str | ~3% | Unique module identifier | "python-basics-001" |
| `lia:module_type` | str | ~3% | Module category/type | "tutorial", "exercise", "reference" |
| `lia:docs_version` | str | Rare | Documentation version | "v2.1" |
| `lia:estimated_time_in_minutes` | str | ~2% | Expected completion time | "45", "120" |
| `lia:good_first_module` | str | Rare | Beginner-friendly indicator | "true", "yes" |

#### Coding Metadata

For programming courses:

| Column | Type | Coverage | Description | Example |
|--------|------|----------|-------------|---------|
| `lia:coding_required` | str | ~2% | Whether coding is required | "true", "false" |
| `lia:coding_level` | str | ~2% | Required coding skill level | "beginner", "intermediate", "advanced" |
| `lia:coding_language` | str | ~2% | Programming language used | "Python", "JavaScript", "C++" |

#### Detailed Descriptions

| Column | Type | Coverage | Description | Example |
|--------|------|----------|-------------|---------|
| `lia:current_version_description` | list[str] | Rare | Version-specific notes (multi-line) | ["Fixed typos", "Added exercises"] |
| `lia:long_description` | list[str] | Rare | Extended course description (multi-line) | ["This course provides..."] |
| `lia:collection` | str | Rare | Part of a collection/series | "Python Learning Path" |
| `lia:sequence_name` | str | Rare | Sequence within collection | "Part 1: Basics" |
| `lia:edit` | str | Rare | Edit link or reference | "https://github.com/.../edit/main/..." |

#### Multi-line Field Handling

Fields marked as `list[str]` support multi-line content in the header:
```markdown
<!--
comment: This is the first line
         and this continues on the next line
         until an empty line or next field.

import: https://github.com/LiaTemplates/Template1
import: https://github.com/LiaTemplates/Template2

tags: Python, Programming,
      Interactive, OER, Education
-->
```

Multi-line values continue until:
- An empty line
- A line starting with another field (contains `:` but not `http://` or `https://`)
- A line starting with `@` (special directive)

#### Data Quality Notes

- **High coverage (>95%)**: `author`, `email`, `version`, `language`, `narrator`, `comment`
- **Medium coverage (25-50%)**: `tags`, `logo`, `import`, `link`, `script`
- **Low coverage (<15%)**: `icon`, `title`, `date`, module/coding metadata
- All fields are optional in LiaScript specification
- Null values are normal - not all courses use all features
- List fields can contain multiple values (e.g., multiple imports)

**Analysis Use:**
- **Language distribution:** `lia:language` field
- **Template usage:** `lia:import` field (30.3% of courses use templates)
- **Custom JavaScript:** `lia:script` field (advanced interactive features)
- **TTS adoption:** `lia:narrator` field (accessibility feature)
- **Authorship analysis:** Combine `lia:author` with commit data
- **License research:** Compare `lia:content_license` with repository licenses
- **Thematic clustering:** Use `lia:tags` field (50.6% coverage)
- **Course complexity:** Presence of `coding_required`, `estimated_time` fields

---

### 5. LiaScript_features.p

**Purpose:** Detailed feature usage analysis for each validated LiaScript file
**Rows:** 3,672 | **Columns:** 104

This dataset quantifies the usage of LiaScript-specific features in each course.

**Feature-Dokumentation:** Alle erkannten Muster, Regex-Ausdrücke und zugehörigen Daten-Label sind dokumentiert in [`LiaScript_Features_Patterns.md`](cl-pipeline/run/pipelines/liascript/docs/LiaScript_Features_Patterns.md).

#### Core Identification Fields

| Column | Type | Description |
|--------|------|-------------|
| `pipe:ID` | string | File identifier (PRIMARY KEY) |
| `repo_name` | string | Repository name |
| `repo_user` | string | Repository owner |
| `file_name` | string | Original filename |

#### Feature Count and Flag Columns

Each feature has two columns: a count (`feature:X_count`) and a boolean flag (`feature:has_X`).

| Feature Category | Count Column | Flag Column | Description |
|-----------------|--------------|-------------|-------------|
| **Video** | `feature:video_count` | `feature:has_video` | Embedded videos (`!?[...]`) |
| **Audio** | `feature:audio_count` | `feature:has_audio` | Audio files |
| **Web Apps** | `feature:webapp_count` | `feature:has_webapp` | Embedded web applications |
| **LiaViz Tables** | `feature:lia_viz_table_count` | `feature:has_lia_viz_tables` | LiaScript visualization tables |
| **Tables** | `feature:table_count` | `feature:has_tables` | Standard markdown tables |
| **Text Quiz** | `feature:text_quiz_count` | `feature:has_text_quiz` | Text input quizzes |
| **Single Choice** | `feature:single_choice_count` | `feature:has_single_choice` | Single choice questions |
| **Multiple Choice** | `feature:multiple_choice_count` | `feature:has_multiple_choice` | Multiple choice questions |
| **Quiz Hints** | `feature:quiz_hint_count` | `feature:has_quiz_hints` | Quiz hint markers |
| **Total Quiz** | `feature:total_quiz_count` | `feature:has_quiz` | All quiz types combined |
| **Code Blocks** | `feature:code_block_count` | `feature:has_code_blocks` | Fenced code blocks |
| **Script Tags** | `feature:script_tag_count` | `feature:has_script_tags` | Inline `<script>` tags |
| **Imports** | `feature:import_count` | `feature:has_imports` | Template imports |
| **Narrator** | - | `feature:has_narrator` | TTS narrator configured |
| **TTS Comments** | `feature:tts_comment_count` | `feature:has_tts_comments` | Text-to-speech comments |
| **Animation Fragments** | `feature:animation_fragment_count` | `feature:has_animation_fragments` | Animation effects |
| **Animated CSS** | `feature:animated_css_count` | `feature:has_animated_css` | CSS animations |
| **Images** | `feature:image_count` | `feature:has_images` | Image embeds |
| **External Scripts** | `feature:external_script_count` | `feature:has_external_scripts` | External JavaScript |
| **External CSS** | `feature:external_css_count` | `feature:has_external_css` | External stylesheets |
| **QR Codes** | `feature:qr_code_count` | `feature:has_qr_codes` | QR code elements |
| **Preview LIA** | `feature:preview_lia_count` | `feature:has_preview_lia` | LiaScript preview links |
| **Math (Inline)** | `feature:inline_math_count` | - | Inline math expressions |
| **Math (Display)** | `feature:display_math_count` | - | Display math expressions |
| **Math (Any)** | - | `feature:has_math` | Any math expressions |

#### New Feature Columns (added in v1.5)

The feature set was substantially expanded. New count and flag columns include:

| Feature Category | Count Column | Flag Column | Description |
|-----------------|--------------|-------------|-------------|
| **Animation Blocks** | `feature:animation_block_count` | `feature:has_animation_blocks` | Multi-step animation blocks (`{{}}`) |
| **ASCII Diagrams** | `feature:ascii_diagram_count` | `feature:has_ascii_diagrams` | ASCII art diagram blocks |
| **TTS Blocks** | `feature:tts_block_count` | `feature:has_tts_blocks` | Block-level text-to-speech |
| **TTS Fragments** | `feature:tts_fragment_count` | `feature:has_tts_fragments` | Inline TTS fragments |
| **Surveys** | `feature:survey_count` | `feature:has_surveys` | Survey questions (`[(x)]`) |
| **Survey Text** | `feature:survey_text_count` | `feature:has_survey_text` | Open text survey inputs |
| **Matrix Quiz** | `feature:matrix_quiz_count` | `feature:has_matrix_quiz` | Matrix-style quiz tables |
| **Selection Quiz** | `feature:selection_quiz_count` | `feature:has_selection_quiz` | Dropdown/select quizzes |
| **Executable Code** | `feature:executable_code_count` | `feature:has_executable_code` | Code blocks with `@input`/`@output` |
| **Code Projects** | `feature:code_project_count` | `feature:has_code_projects` | Multi-file code projects |
| **HTML Embeds** | `feature:html_embed_count` | `feature:has_html_embeds` | Embedded HTML elements |
| **Galleries** | `feature:gallery_count` | `feature:has_galleries` | Auto-rendered image galleries |
| **Task Lists** | `feature:task_list_count` | `feature:has_task_lists` | Checkbox task lists (`[-]`/`[x]`) |
| **Footnotes** | `feature:footnote_count` | `feature:has_footnotes` | Markdown footnotes (`[^1]`) |
| **Effects** | `feature:effect_count` | `feature:has_effects` | LiaScript animation effects |
| **Classroom** | `feature:classroom_count` | `feature:has_classroom` | Live-classroom features |
| **Custom Macros** | `feature:custom_macro_def_count` | `feature:has_custom_macro_defs` | User-defined macros |
| **Macros (any)** | `feature:macro_count` | `feature:has_macros` | Any macro usage |
| **Links** | `feature:link_count` | `feature:has_links` | Hyperlinks |
| **Comments** | `feature:comment_count` | `feature:has_comments` | HTML comments |
| **Logo** | — | `feature:has_logo` | Course logo set |
| **Icon** | — | `feature:has_icon` | Course icon set |
| **SVG** | `feature:svg_count` | — | Inline SVG elements |
| **iFrame** | `feature:iframe_count` | — | Embedded iFrames |

#### Structural Count Columns

| Column | Description |
|--------|-------------|
| `feature:h1_count` | Number of H1 headings |
| `feature:h2_count` | Number of H2 headings |
| `feature:h3_count` | Number of H3 headings |
| `feature:total_headings` | Total heading count |
| `feature:total_survey_count` | All survey types combined |
| `feature:code_language_count` | Number of distinct programming languages |
| `feature:code_languages` | List of programming languages used |
| `feature:custom_macro_names` | List of defined macro names |
| `feature:template_categories` | Template category assignments |
| `feature:template_category_count` | Number of distinct template categories |
| `feature:details_count` | HTML `<details>` element count |
| `feature:footnote_def_count` | Footnote definitions count |

#### Template Tracking

| Column | Type | Description |
|--------|------|-------------|
| `feature:imported_templates` | list | List of imported template URLs |

**Analysis Use:**
- Feature adoption analysis (which features are most popular?)
- Course complexity metrics (feature counts as proxy)
- Template dependency analysis
- Interactivity levels (quiz, code, animation usage)

---

### 6. LiaScript_feature_statistics.p

**Purpose:** Aggregated statistics on feature usage across all validated courses
**Type:** Python dictionary (not a DataFrame)

#### Structure

```python
{
    'total_documents': 3672,
    'timestamp': datetime,
    'feature_usage': {
        'quiz': {'count': 1649, 'percentage': 59.94},
        'text_quiz': {'count': 1631, 'percentage': 59.29},
        ...
    },
    'template_usage': {
        'https://...': {'count': 1100, 'percentage': 39.99},
        ...
    },
    'summary': {
        'most_common_feature': 'quiz',
        'total_templates': 79,
        'most_used_template': 'https://...'
    }
}
```

#### Feature Usage Summary (3,672 documents analyzed)

| Feature | Documents | Percentage |
|---------|-----------|------------|
| macros | 2,461 | 67.0% |
| links | 2,109 | 57.4% |
| narrator | 1,972 | 53.7% |
| external_scripts | 1,701 | 46.3% |
| math | 1,630 | 44.4% |
| comments | 1,622 | 44.2% |
| images | 1,620 | 44.1% |
| quiz | 1,545 | 42.1% |
| code_blocks | 1,408 | 38.3% |
| tables | 1,208 | 32.9% |
| text_quiz | 969 | 26.4% |
| animation_fragments | 783 | 21.3% |
| external_css | 779 | 21.2% |
| script_tags | 715 | 19.5% |
| logo | 667 | 18.2% |
| html_embeds | 579 | 15.8% |
| audio | 535 | 14.6% |
| icon | 528 | 14.4% |
| video | 516 | 14.1% |
| lia_viz_tables | 413 | 11.2% |
| tts_fragments | 407 | 11.1% |
| animation_blocks | 371 | 10.1% |
| multiple_choice | 354 | 9.6% |
| ascii_diagrams | 332 | 9.0% |
| single_choice | 313 | 8.5% |

#### Top 10 Templates Used

| Rank | Template | Documents | Percentage |
|------|----------|-----------|------------|
| 1 | Tikz-Jax | 1,100 | 30.0% |
| 2 | Algebrite | 404 | 11.0% |
| 3 | CodeRunner | 237 | 6.5% |
| 4 | GGBScript | 234 | 6.4% |
| 5 | arcus/education_modules | 100 | 2.7% |
| 6 | ABCjs | 102 | 2.8% |
| 7 | Pyodide | 67 | 1.8% |
| 8 | PlantUML | 53 | 1.4% |
| 9 | AVR8js | 43 | 1.2% |
| 10 | Rextester | 29 | 0.8% |

**Total unique templates:** 79

A human-readable version is also available in `LiaScript_feature_statistics.txt`.

---

### 7. LiaScript_clusters.p

**Purpose:** User segmentation, author clustering, and template adoption analysis
**Type:** Python dictionary (not a DataFrame)
**Generated:** 2026-03-13

This dataset provides insights into how different authors use LiaScript features and templates, avoiding bias from prolific authors.

#### Structure

```python
{
    'cluster_assignments': {...},      # Document → cluster mapping
    'cluster_statistics': {...},       # Cluster sizes and percentages
    'cluster_profiles': {...},         # Feature profiles per cluster
    'official_templates': {...},       # Template usage analysis
    'author_analysis': {...},          # Author-based metrics (key!)
    'cooccurrence_matrix': {...},      # Feature co-occurrence
    'complexity_metrics': {...},       # Document complexity stats
    'documentation_gaps': {...},       # Underutilized features
    'total_documents': 3672,
    'total_authors': 254,
    'timestamp': datetime
}
```

#### Document Clusters

Documents are assigned to behavioral clusters (can belong to multiple):

| Cluster | Documents | % | Description |
|---------|-----------|---|-------------|
| template_power_user | 1,786 | 48.6% | Uses imports + external scripts/macros |
| minimalist | 1,363 | 37.1% | Few interactive/pedagogical features (≤2 of 17 checked) |
| presenter | 1,117 | 30.4% | Narrator + images, few quizzes |
| assessment_focus | 722 | 19.7% | Quiz-heavy with hints/MC/surveys |
| mint_author | 681 | 18.6% | Math + code + visualization |
| multimedia_course | 617 | 16.8% | Video/audio/webapp |
| general | 103 | 2.8% | No specific pattern |

#### Committer-Based Analysis (Avoiding Organizational Bias)

**Methodology change (March 2026):** The author analysis was refactored to use actual **git committers** from `contributors_list` instead of `repo_user` (repository owner). This resolves a key issue: organizations like `MINT-the-GAP` were previously counted as single "authors," masking the individual contributors behind them.

- **Source:** `LiaScript_commits.p` → `contributors_list` per document
- **Mapping:** Each unique committer is associated with all documents they contributed to (many-to-many)
- **Fallback:** Documents without commit data use `repo_user`
- **85% of documents** have exactly one committer; 15% have multiple

**Committer Distribution:**
- 194 committers (51.7%) have only 1 document
- 131 committers (34.9%) have 2-10 documents
- 34 committers (9.1%) have 11-50 documents
- 6 committers (1.6%) have 100+ documents

**Template Adoption - Document vs. Committer Comparison:**

| Template | Docs | Doc% | Committers | Committer% |
|----------|------|------|------------|------------|
| tikz-jax | 1,100 | 30.0% | 3 | 0.8% |
| algebrite | 404 | 11.0% | 13 | 3.5% |
| coderunner | 237 | 6.5% | **62** | **16.5%** |
| ggbscript | 234 | 6.4% | 2 | 0.5% |
| abcjs | 102 | 2.8% | 15 | 4.0% |
| pyodide | 67 | 1.8% | 18 | 4.8% |
| plantuml | 53 | 1.4% | **41** | **10.9%** |
| avr8js | 43 | 1.2% | 25 | 6.7% |

**Key Finding:** CodeRunner is the most broadly adopted template (62 committers, 16.5%), followed by plantuml (41 committers, 10.9%). In contrast, tikz-jax dominates by document count (30%) but is used by only 3 committers.

#### K-Means Committer Clusters (Features + Templates)

181 committers with ≥2 documents are clustered using 102 features (44 LiaScript features, 30 template indicators, 28 category indicators).

**Identified Committer Clusters:**

| Cluster | Committers | Docs | Top Features | Top Templates |
|---------|------------|------|--------------|---------------|
| **Minimalisten** | 99 | 1,297 | links (71%), narrator (58%), images (47%) | coderunner (15%), abcjs (4%) |
| **Präsentatoren** | 50 | 1,960 | links (91%), narrator (79%), macros (75%) | pyodide (26%), coderunner (16%) |
| **Präsentatoren (Technik)** | 27 | 168 | links (98%), images (96%), narrator (92%) | plantuml (74%), coderunner (56%), avr8js (33%) |
| **Playground/Demo (Core)** | 2 | 685 | links (89%), code_blocks (71%), images (70%) | tikz-jax, algebrite, plantuml (all 100%) |
| **Playground/Demo** | 2 | 7 | video/audio/tables (100%) | multiple templates (all 100%) |
| **Unknown commits** | 1 | 316 | narrator (85%), links (84%), macros (64%) | broad template usage |

#### Official LiaTemplates Categorization

Templates from `github.com/LiaTemplates` are categorized:

| Category | Templates | Committers | Description |
|----------|-----------|------------|-------------|
| code_execution | coderunner, rextester, avr8js | 62 (16.5%) | Run code in browser |
| diagrams | plantuml, mermaid | 41 (10.9%) | Diagram generation |
| microcontroller | avr8js | 25 (6.7%) | Arduino simulation |
| python_execution | pyodide, pyscript | 23 (6.1%) | Python in browser |
| music_notation | abcjs | 15 (4.0%) | Music notation rendering |
| math_visualization | tikz-jax | 1 (0.4%) | LaTeX graphics |
| symbolic_math | algebrite | 6 (2.4%) | Computer algebra |

#### Documentation Gap Hypotheses

Features with <10% usage may indicate documentation issues:

| Feature | Usage | Hypothesis |
|---------|-------|------------|
| effects | 0.0% | Unknown/undocumented |
| classroom | 0.04% | Complex setup required |
| tts_blocks | 0.07% | Unknown feature |
| code_projects | 0.29% | Complex syntax |
| animated_css | 0.47% | Requires CSS knowledge |
| surveys | 0.76% | Confused with quizzes |
| webapp `??[...]` | 3.6% | Non-intuitive syntax |
| quiz_hints | 4.3% | Feature not well-known |

#### Complexity Metrics

- Average features per document: 8.07
- Median features per document: 7.0
- Maximum features in a document: 37

A human-readable version is available in `LiaScript_clusters.txt`.

**Analysis Use:**
- Identify user segments for targeted documentation
- Understand template adoption patterns
- Detect documentation gaps from underutilized features
- Compare document-based vs. author-based metrics
- Analyze feature co-occurrence patterns

---

### 8. LiaScript_content.p

**Purpose:** Full text content and language detection results
**Rows:** 3,672 | **Columns:** 7

> **Note:** Section numbering adjusted - this was previously section 7.

#### Columns

| Column | Type | Description |
|--------|------|-------------|
| `pipe:ID` | string | File identifier (PRIMARY KEY) |
| `pipe:file_type` | string | Always "md" |
| `pipe:content_hash` | string | Hash of file content (deduplication) |
| `pipe:content_pages` | integer | Estimated page count |
| `pipe:content_words` | integer | Word count |
| `pipe:most_prob_language` | string | Detected language code |
| `pipe:language_probability` | float | Confidence score (0-1) |

**Analysis Use:**
- Course length analysis (word counts)
- Language distribution (detected vs. declared in metadata)
- Content deduplication
- Corpus size estimation

---

### 9. LiaScript_ai_meta.p

**Purpose:** AI-generated metadata for content analysis and classification
**Rows:** 3,444 | **Columns:** 10

**Status:** Processing complete for 3,444/3,672 courses (93.8%).

**Important:** This dataset is generated by LLM analysis (llama3.3:70b) and provides rich metadata for research purposes.

#### AI-Generated Fields

| Column | Type | Description | Coverage |
|--------|------|-------------|----------|
| `pipe:ID` | string | File identifier (PRIMARY KEY) | 100% |
| `pipe:file_type` | string | Always "md" | 100% |
| `ai:title` | string | AI-extracted course title | 100% |
| `ai:keywords_ext` | string | Comma-separated extracted keywords (exactly 15) | 100% |
| `ai:keywords_gen` | string | Comma-separated generated keywords (exactly 15) | 100% |
| `ai:summary` | string | 3-sentence course summary | 100% |
| `ai:dewey` | JSON | Dewey Decimal Classifications with scores | 100% |
| `ai:education_level` | string | Education level classification | 100% |
| `ai:target_audience` | string | Detailed target audience description | 100% |
| `ai:_errors` | dict | Error tracking per field (see below) | internal |

**Field Details:**

**`ai:title`**
- Extracted from LiaScript metadata header or first heading
- Fallback if manual `title:` field is missing or unclear
- More reliable than parsing headers directly

**`ai:keywords_ext`**
- Exactly 15 German keywords per course (extracted from content)
- Suitable for library cataloging
- Focus: technical topics, learning objectives, tools, scientific domains
- Example: "Python, Programmierung, Anfänger, Interaktiv, OER, ..."

**`ai:keywords_gen`**
- Exactly 15 German keywords per course (generated based on content understanding)
- Complements extracted keywords with inferred concepts
- Provides additional semantic context for cataloging

**`ai:summary`**
- Exactly 3 sentences in German
- Covers: main topic, interactive elements, target audience
- Designed for course catalogs and research overviews

**`ai:dewey`**
- JSON array with up to 3 Dewey Decimal Classifications
- Format: `[{"notation": "005.1", "label": "Programmierung", "score": 0.9}, ...]`
- Scored by relevance (0.0-1.0)
- Enables thematic clustering and subject analysis

**`ai:education_level`**
- Classification of the education level the course targets
- Values include: "Sekundarstufe I", "Sekundarstufe II", "Hochschulbildung", "Berufsausbildung", "Weiterbildung", etc.
- German language output

**`ai:target_audience`**
- Detailed description of the intended audience
- Includes prerequisites and prior knowledge requirements
- German language, 1-2 sentences
- Example: "Studierende der Life Sciences, insbesondere im Bereich Biologie, Biochemie oder Medizin, mit Grundkenntnissen in Mikroskopie und Imaging-Techniken, die ihre Fähigkeiten im Bereich Elektronenmikroskopie erweitern möchten."

**`ai:_errors`** *(internal pipeline field — exclude from research analysis)*
- Dictionary tracking extraction failures per field, introduced to prevent endless retry loops
- Structure:
  ```python
  {
    'ai:keywords_ext': {
      'count': 2,                          # Number of failed extraction attempts
      'last_error': 'timeout',             # Last error message
      'last_attempt': '2026-03-13T...'     # ISO timestamp of last attempt
    },
    ...
  }
  ```
- Fields tracked: `ai:author`, `ai:keywords_gen`, `ai:title`, `ai:type`, `ai:keywords_ext`, `ai:keywords_dnb`, `ai:summary`, `ai:education_level`, `ai:target_audience`
- Documents with `count >= max_error_retries` (default: 3) are permanently skipped for that field
- **Analysis note:** A non-empty `ai:_errors` entry indicates the corresponding `ai:*` field may be missing or of lower quality. Filter with:
  ```python
  # Documents where title extraction never failed
  reliable = df_ai[df_ai['ai:_errors'].apply(lambda e: 'ai:title' not in (e or {}))]
  ```
- Initialized via migration script [scripts/migrate_error_tracking.py](../../scripts/migrate_error_tracking.py)

#### Processing Configuration

From [config/full.yaml](../../run/pipelines/liascript/config/full.yaml):

```yaml
processing_mode:
  # Force processing: These fields are ALWAYS processed (overwrite existing data)
  force_processing:
    - ai:education_level     # Force re-process: Update education level classification
    - ai:target_audience     # Force re-process: Update target audience description

  # Conditional processing: These fields are only processed if empty/missing
  conditional_processing:
    - ai:keywords_ext        # Only process if missing/empty or invalid count
    - ai:keywords_gen        # Only process if missing/empty or invalid count
    - ai:dewey               # Only process if missing/empty
    - ai:title               # Only process if missing/empty
    - ai:summary             # Only process if missing/empty

  # Skip configuration
  allow_skip_when_all_conditional_filled: false  # Process all files for force_processing fields
```

**Behavior:**
- **Force processing fields** are ALWAYS re-processed to ensure data consistency and quality
- **Conditional processing fields** are only generated if missing (incremental updates)
- Current batch size: 50 documents (saves every 50 to reduce I/O overhead)

#### Model & Prompts

- **Model:** llama3.3:70b (local Ollama)
- **Prompts:** [prompts.yaml](../../run/pipelines/liascript/prompts/prompts.yaml)
- **Language:** All outputs in German (except Dewey labels when applicable)
- **Context window:** Up to 8,000 characters per file

**Analysis Use:**
- **Thematic clustering** via Dewey classifications
- **Keyword analysis** for feature and topic distribution (two complementary keyword sets)
- **Content summarization** for qualitative analysis
- **Missing metadata enhancement** when manual headers incomplete
- **Cross-language research** (German keywords/summaries even for English courses)
- **Semantic comparison** between extracted and generated keywords reveals content emphasis
- **Education level analysis** for understanding target demographics
- **Audience segmentation** based on prerequisites and skill levels

**Quality Notes:**
- AI-generated content should be validated for critical research
- `ai:keywords_ext` extracts from content; `ai:keywords_gen` infers semantic concepts
- Dewey classifications are AI-assigned, not librarian-verified
- 228 courses (6.2%) not yet processed - primarily due to content issues
- Check file modification timestamp for latest processing status
- Force processing ensures all entries have exactly 15 keywords and consistent Dewey classifications

---

### 10. LiaScript_user_profiles.p

**Purpose:** GitHub profile metadata for all repository owners in the dataset
**Rows:** 445 (one per unique `repo_user`) | **Columns:** 15

This dataset enables geographic distribution analysis and institutional affiliation mapping. Collected via the GitHub API for each unique user appearing in `LiaScript_repositories.p`.

#### Columns

| Column | Type | Coverage | Description | Example |
|--------|------|----------|-------------|---------|
| `login` | string | 100% | GitHub username (links to `repo_user`) | "andre-dietrich" |
| `name` | string | ~70% | Real name (if public) | "André Dietrich" |
| `company` | string | 22% | Employer or institution | "TU Bergakademie Freiberg" |
| `location` | string | 40% | Location string (self-reported) | "Freiberg, Germany" |
| `bio` | string | 35% | GitHub bio/description | "Educational technology researcher" |
| `email` | string | ~15% | Public email address | "name@university.de" |
| `blog` | string | ~25% | Personal website or ORCID | "https://example.com" |
| `twitter_username` | string | ~5% | Twitter/X handle | "example_handle" |
| `followers` | integer | 100% | GitHub follower count | 42 |
| `following` | integer | 100% | GitHub following count | 18 |
| `public_repos` | integer | 100% | Number of public repositories | 27 |
| `created_at` | datetime | 100% | Account creation timestamp | 2015-04-23 |
| `updated_at` | datetime | 100% | Last profile update | 2025-11-10 |
| `profile_url` | string | 100% | Full GitHub profile URL | "https://github.com/..." |
| `type` | string | 100% | Account type | "User" or "Organization" |

**Joining to other datasets:**
```python
df_repos = pd.read_pickle('LiaScript_repositories.p')
df_profiles = pd.read_pickle('LiaScript_user_profiles.p')

# Join on login == user
df = df_repos.merge(df_profiles, left_on='user', right_on='login', how='left')
```

**Analysis Use:**
- Geographic distribution of LiaScript authors (via `location`)
- Institutional affiliation analysis (via `company`)
- Community size and activity (via `followers`, `public_repos`)
- Account age as proxy for experience level
- Distinguish individual users from organizations (`type`)

---

### 11. LiaScript_consolidated.p

**Purpose:** Pre-joined wide-format DataFrame combining all per-file datasets for convenient analysis
**Rows:** 3,672 | **Columns:** 166 (all datasets merged on `pipe:ID`)

This dataset is the primary starting point for research analysis. It avoids repeated merge operations by pre-joining the four core per-file datasets. It covers only AI-validated LiaScript courses (`pipe:is_valid_liascript == True`).

#### Included Data Sources

| Source Dataset | Prefix | Fields Included |
|----------------|--------|-----------------|
| `LiaScript_files_validated.p` | `pipe:`, `repo_*`, `liaIndi_*` | All file and repo metadata |
| `LiaScript_metadata.p` | `lia:` | All LiaScript header fields |
| `LiaScript_features.p` | `feature:` | All feature counts and flags |
| `LiaScript_commits.p` | — | `commit_count`, `author_count`, `first_commit`, `last_commit`, `contributors_list` |

**Not included** (join separately if needed):
- `ai:*` fields from `LiaScript_ai_meta.p`
- Content data from `LiaScript_content.p`
- Cluster assignments from `LiaScript_clusters.p`
- User profiles from `LiaScript_user_profiles.p`

#### Loading and Extending

```python
import pandas as pd

# Load consolidated base
df = pd.read_pickle('LiaScript_consolidated.p')
print(f"Consolidated: {len(df)} rows, {len(df.columns)} columns")

# Extend with AI metadata
df_ai = pd.read_pickle('LiaScript_ai_meta.p')
df = df.merge(df_ai[['pipe:ID', 'ai:title', 'ai:keywords_ext',
                       'ai:summary', 'ai:dewey', 'ai:education_level',
                       'ai:target_audience']], on='pipe:ID', how='left')

# Extend with content stats
df_content = pd.read_pickle('LiaScript_content.p')
df = df.merge(df_content[['pipe:ID', 'pipe:content_words',
                            'pipe:most_prob_language']], on='pipe:ID', how='left')

# Filter for internal vs. community
internal = df[df['repo_user'].isin([
    'SebastianZug', 'andre-dietrich', 'LiaScript', 'LiaBooks',
    'LiaTemplates', 'LiaPlayground', 'TUBAF-IfI-LiaScript',
    'TUBAF-IUZ-LiaScript'
])]
community = df[~df['repo_user'].isin(internal['repo_user'].unique())]
```

**Also available as CSV:** `LiaScript_consolidated.csv` (same content, for use without pandas/pickle).

**Analysis Use:**
- Single-table analysis without repeated merges
- Feature-metadata correlations (e.g., quiz usage by declared language)
- Commit activity vs. feature richness
- Starting point for all paper analysis workflows

---

## Processed Data

Location: `/media/sz/Data/Connected_Lecturers/LiaScript/processed/`

### ChromaDB Vector Store

**Path:** `processed/chroma_db/`
**Collection:** `liascript_courses`

Contains vector embeddings for semantic similarity search using the `qwen3-embedding` model (Ollama).

### LiaScript_ai_similarity.p

**Purpose:** Precomputed similarity scores between courses
**Note:** Small file, may contain partial results

---

## Data Quality Notes

### Validation Pipeline
- **Initial detection:** 57,096 markdown files with LiaScript indicators
- **AI validation:** Only 3,672 files (6.4%) confirmed as genuine LiaScript courses
- **Validation model:** llama3.3:70b (local Ollama instance)
- **Excluded repositories:** Indices 439, 481, 497, 575 (see [config/full.yaml](../../run/pipelines/liascript/config/full.yaml))

### Missing Values
- **Metadata fields** (`lia:author`, `lia:language`, etc.): High null rates normal - only present in files with proper LiaScript headers
- **Commit fields**: 3 files missing (0.1%) — after URL-decode fix, down from 314 (8.6%)
- **Content extraction**: 0 files missing
- **AI metadata**: 3,444/3,672 = 93.8% complete
- **Feature analysis**: 100% complete (3,672/3,672)

### Data Linking

All datasets link via `pipe:ID`. Example join:

```python
import pandas as pd

# Load datasets
df_files = pd.read_pickle('LiaScript_files_validated.p')
df_commits = pd.read_pickle('LiaScript_commits.p')
df_metadata = pd.read_pickle('LiaScript_metadata.p')
df_content = pd.read_pickle('LiaScript_content.p')
df_features = pd.read_pickle('LiaScript_features.p')
df_ai_meta = pd.read_pickle('LiaScript_ai_meta.p')

# Filter for valid LiaScript courses only
df_valid = df_files[df_files['pipe:is_valid_liascript'] == True]

# Join all data
df_full = (df_valid
    .merge(df_commits, on='pipe:ID', how='left', suffixes=('', '_commits'))
    .merge(df_metadata, on='pipe:ID', how='left')
    .merge(df_content, on='pipe:ID', how='left')
    .merge(df_features, on='pipe:ID', how='left')
    .merge(df_ai_meta, on='pipe:ID', how='left')
)

print(f"Complete dataset: {len(df_full)} validated LiaScript courses")
print(f"With AI metadata: {df_full['ai:title'].notna().sum()} courses")
print(f"With feature data: {df_full['feature:has_quiz'].notna().sum()} courses")
```

---

## Usage for Research

### Recommended Workflow

1. **Always filter for validated courses first:**
   ```python
   df_valid = df_files[df_files['pipe:is_valid_liascript'] == True]
   ```

2. **Join relevant datasets** based on your research question

3. **Handle list-type fields** in metadata:
   ```python
   # Extract first author
   df_metadata['first_author'] = df_metadata['lia:author'].apply(
       lambda x: x[0] if isinstance(x, list) and len(x) > 0 else None
   )
   ```

### Example Research Questions

**Q1: Template usage patterns**
```python
df_features = pd.read_pickle('LiaScript_features.p')
template_users = df_features[df_features['feature:has_imports'] == True]
print(f"{len(template_users)} courses use templates ({len(template_users)/len(df_features)*100:.1f}%)")
```

**Q2: Collaboration vs. individual authorship**
```python
df_commits = pd.read_pickle('LiaScript_commits.p')
collab = df_commits[df_commits['author_count'] > 1]
print(f"Collaborative courses: {len(collab)} ({len(collab)/len(df_commits)*100:.1f}%)")
```

**Q3: Quiz adoption by education level**
```python
df_features = pd.read_pickle('LiaScript_features.p')
df_ai = pd.read_pickle('LiaScript_ai_meta.p')
df_merged = df_features.merge(df_ai[['pipe:ID', 'ai:education_level']], on='pipe:ID')
quiz_by_level = df_merged.groupby('ai:education_level')['feature:has_quiz'].mean()
print(quiz_by_level)
```

**Q4: Language distribution**
```python
df_meta = pd.read_pickle('LiaScript_metadata.p')
lang_dist = df_meta['lia:language'].apply(
    lambda x: x[0] if isinstance(x, list) and len(x) > 0 else None
).value_counts()
print(lang_dist)
```

**Q5: Course complexity by feature count**
```python
df_features = pd.read_pickle('LiaScript_features.p')
# Count total features used
feature_cols = [c for c in df_features.columns if c.startswith('feature:has_')]
df_features['total_features'] = df_features[feature_cols].sum(axis=1)
print(df_features['total_features'].describe())
```

**Q6: Most common Dewey categories**
```python
import json
df_ai = pd.read_pickle('LiaScript_ai_meta.p')
dewey_counts = {}
for dewey in df_ai['ai:dewey'].dropna():
    if isinstance(dewey, str):
        dewey = json.loads(dewey)
    for entry in dewey:
        label = entry.get('label', 'Unknown')
        dewey_counts[label] = dewey_counts.get(label, 0) + 1
print(sorted(dewey_counts.items(), key=lambda x: -x[1])[:10])
```

**Q7: Target audience analysis**
```python
df_ai = pd.read_pickle('LiaScript_ai_meta.p')
# Education level distribution
print(df_ai['ai:education_level'].value_counts())
```

**Q8: Author-based template adoption (avoiding single-author bias)**
```python
clusters = pd.read_pickle('LiaScript_clusters.p')
author_analysis = clusters['author_analysis']

# Compare document-based vs author-based template metrics
print("Template adoption by unique authors:")
for tmpl, data in list(author_analysis['template_adoption'].items())[:10]:
    print(f"  {tmpl}: {data['author_count']} authors ({data['adoption_pct']:.1f}%)")

# Most broadly adopted template categories
print("\nTemplate category adoption:")
for cat, data in author_analysis['category_adoption'].items():
    print(f"  {cat}: {data['author_count']} authors ({data['adoption_pct']:.1f}%)")
```

**Q9: Identify author clusters**
```python
clusters = pd.read_pickle('LiaScript_clusters.p')
author_clusters = clusters['author_analysis']['author_clusters']

print(f"K-Means clustering with {author_clusters['n_features_used']} features")
print(f"Feature types: {author_clusters['feature_types']}")

for cluster_id, data in author_clusters['clusters'].items():
    print(f"\n{data['name']} ({data['author_count']} authors, {data['total_docs']} docs):")
    print(f"  Top features: {list(data['top_features'].keys())[:5]}")
    print(f"  Top templates: {list(data['top_templates'].keys())[:3]}")
```

**Q10: Internal vs. community feature usage**
```python
df = pd.read_pickle('LiaScript_consolidated.p')

INTERNAL_ACCOUNTS = [
    'SebastianZug', 'andre-dietrich', 'LiaScript', 'LiaBooks',
    'LiaTemplates', 'LiaPlayground', 'TUBAF-IfI-LiaScript',
    'TUBAF-IUZ-LiaScript',
]

df['is_internal'] = df['repo_user'].isin(INTERNAL_ACCOUNTS)
feature_cols = [c for c in df.columns if c.startswith('feature:has_')]

comparison = df.groupby('is_internal')[feature_cols].mean().T
comparison.columns = ['Community', 'Internal']
comparison['diff'] = comparison['Internal'] - comparison['Community']
print(comparison.sort_values('diff', ascending=False).head(10))
```

**Q11: Geographic distribution of authors**
```python
df_repos = pd.read_pickle('LiaScript_repositories.p')
df_profiles = pd.read_pickle('LiaScript_user_profiles.p')

df = df_repos.merge(df_profiles, left_on='user', right_on='login', how='left')
# Location strings are freeform; simple country extraction:
country_counts = df['location'].dropna().str.extract(r',\s*([^,]+)$')[0].value_counts()
print(country_counts.head(15))
```

**Q12: AI metadata completeness and Dewey main-class distribution**
```python
import json
from collections import Counter

df_ai = pd.read_pickle('LiaScript_ai_meta.p')

# Documents with no extraction errors
fully_complete = df_ai[df_ai['ai:_errors'].apply(lambda e: len(e or {}) == 0)]
print(f"Fully complete AI metadata: {len(fully_complete)} / {len(df_ai)}")

# Dewey main-class distribution (first digit → hundreds)
dewey_classes = []
for dewey in df_ai['ai:dewey'].dropna():
    if isinstance(dewey, str):
        dewey = json.loads(dewey)
    for entry in dewey:
        main_class = str(entry.get('notation', ''))[:1] + '00'
        dewey_classes.append(main_class)
print(Counter(dewey_classes).most_common(10))
```

---

## Pipeline Configuration

**Configuration file:** [run/pipelines/liascript/config/full.yaml](../../run/pipelines/liascript/config/full.yaml)

**Key settings:**
- LLM model: llama3.3:70b (Ollama)
- Embedding model: qwen3-embedding (Ollama)
- ChromaDB collection: liascript_courses
- Excluded repository indices: [439, 481, 497, 575]

**Pipeline stages:**
1. ProvideDataFolders
2. CrawlGithubForLiaScript
3. AggregateLiaScriptFiles
4. ValidateLiaScriptFiles (AI validation)
5. AggregateLiaScriptCommits
6. ExtractLiaScriptMetadata
7. AnalyzeLiaScriptFeatures
8. ExtractFileContent
9. AIEmbeddingsGeneration
10. AIMetaDataExtraction
11. CollectGithubUserProfiles
12. MergeLiaScriptData (consolidation)

**Utility scripts:**
- [add_internal_repos.py](../../add_internal_repos.py) — Back-fills internal LiaScript organization repositories with `internal=True` flag
- [check_internal_coverage.py](../../check_internal_coverage.py) — Verifies that all internal-account files are present across all pipeline stages (coverage QA tool)
- [scripts/migrate_error_tracking.py](../../scripts/migrate_error_tracking.py) — One-time migration to initialize `ai:_errors` for documents with previously empty AI fields

---

## Data Quality Tools

### check_internal_coverage.py

Verifies that all repositories from internal LiaScript accounts (core developers and organizations) are present and fully processed through every pipeline stage.

**Usage:**
```bash
python check_internal_coverage.py --data-root /path/to/liascript/raw
```

**Output example:**
```
=== Downstream Coverage (of 127 valid internal files) ===
  Stage                Total   Internal    Missing   Coverage
  ------------------------------------------------------------
  metadata              2523        125          2      98.4%
  features              2751        127          0     100.0%
  content               2749        127          0     100.0%
  ai_meta               2557        121          6      95.3%
  consolidated          2751        127          0     100.0%
  embeddings (chroma)  38421        127          0     100.0%

=== AI Meta Field Completeness (internal files: 121) ===
  ai:dewey                         121 / 121 filled
  ai:education_level               121 / 121 filled
  ai:keywords_ext                  121 / 121 filled
  ai:keywords_gen                  119 / 121 filled
  ai:summary                       121 / 121 filled
  ai:target_audience               121 / 121 filled
  ai:title                         121 / 121 filled
  ai:_errors                        14 / 121 filled
```

**Interpretation:** A non-zero `Missing` count means those files did not reach that pipeline stage. `ai:_errors` filled count indicates documents where at least one AI extraction attempt failed.

### migrate_error_tracking.py

One-time migration script that initializes the `ai:_errors` field for documents whose conditional AI fields are empty (likely due to past extraction failures). Sets error count to `max_retries - 1`, giving each stuck document exactly one more processing attempt.

**Usage:**
```bash
python scripts/migrate_error_tracking.py /path/to/LiaScript_ai_meta.p --max-retries 3
# Use --dry-run to preview changes without writing
```

---

## Additional Resources

### Related Folders
- **Files:** `/media/sz/Data/Connected_Lecturers/LiaScript/raw/files/` - Actual markdown files (named by `pipe:ID`)
- **Content:** `/media/sz/Data/Connected_Lecturers/LiaScript/raw/content/` - Extracted text content
- **Processed:** `/media/sz/Data/Connected_Lecturers/LiaScript/processed/` - Derived datasets and vector store

### Pipeline Code
- **Stages:** `cl-pipeline/stages/liascript/`
- **Configuration:** `cl-pipeline/run/pipelines/liascript/config/`
- **Prompts:** `cl-pipeline/run/pipelines/liascript/prompts/prompts.yaml`
- **Feature-Muster:** [`cl-pipeline/run/pipelines/liascript/docs/LiaScript_Features_Patterns.md`](cl-pipeline/run/pipelines/liascript/docs/LiaScript_Features_Patterns.md) — Regex-Muster und Daten-Label aller erkannten Features
- **Utility scripts:** `cl-pipeline/scripts/` and `cl-pipeline/add_internal_repos.py`, `cl-pipeline/check_internal_coverage.py`

### Dataset Overview Table

| File | Type | Rows | Key Join | Description |
|------|------|------|----------|-------------|
| `LiaScript_repositories.p` | DataFrame | 1,076 | `user`+`name` | Repository metadata |
| `LiaScript_files.p` | DataFrame | 57,096+ | `pipe:ID` | All markdown files (pre-validation) |
| `LiaScript_files_validated.p` | DataFrame | 57,096 | `pipe:ID` | Files + AI validation flag |
| `LiaScript_commits.p` | DataFrame | 3,672 | `pipe:ID` | Git history + collaboration |
| `LiaScript_metadata.p` | DataFrame | 3,297 | `pipe:ID` | LiaScript header fields |
| `LiaScript_features.p` | DataFrame | 3,672 | `pipe:ID` | Feature counts and flags (104 cols) |
| `LiaScript_feature_statistics.p` | dict | — | — | Aggregated feature stats |
| `LiaScript_content.p` | DataFrame | 3,672 | `pipe:ID` | Word counts, language detection |
| `LiaScript_ai_meta.p` | DataFrame | 3,444 | `pipe:ID` | AI-generated metadata |
| `LiaScript_clusters.p` | dict | — | — | Author/doc clusters |
| `LiaScript_user_profiles.p` | DataFrame | 445 | `login` | GitHub author profiles |
| `LiaScript_consolidated.p` | DataFrame | 3,672 | `pipe:ID` | Pre-joined wide table (166 cols) |

### Contact & Issues
For questions about this dataset or to report data quality issues:
- Pipeline repository: `/media/sz/Data/Veits_pdfs/Data_aggregation/cl-pipeline/`
- Check pipeline logs for errors

---

## Citation

If you use this dataset in your research, please cite:

```
LiaScript Community Analysis Dataset (2026)
Generated via automated GitHub repository analysis
1,076 repositories, 57,096 files analyzed, 3,672 validated LiaScript courses
Analysis period: [Repository creation dates range from 2018 to 2025]
```

---

**Last updated:** 2026-03-15
**Version:** 1.6
**Generated by:** Automated pipeline with AI validation

**Changelog:**
- **2026-03-15 (v1.6):**
  - `LiaScript_repositories.p`: Added `subscribers` column (GitHub subscribers_count) via `patch_subscribers.py`; now 21 columns
  - Added `patch_subscribers.py` utility script for patching individual columns into existing pickle files
- **2026-03-13 (v1.5):**
  - Updated dataset location to `liascript_march_2026/` (March 2026 pipeline run)
  - Updated all dataset row counts: repositories 1,076 (+396), files_validated 57,096 (−18,164), validated courses 3,672 (+921)
  - `LiaScript_repositories.p`: 8 new columns (`description`, `default_branch`, `size_kb`, `open_issues_count`, `topics`, `languages`, `license_name`, `license_spdx`); removed `validity`
  - `LiaScript_files_validated.p`: 6 new columns from repository context (`description`, `topics`, `languages`, `default_branch`, `size_kb`, `open_issues_count`)
  - `LiaScript_features.p`: massively expanded from 51 to 104 columns; 57 new feature columns including `has_macros`, `has_links`, `has_comments`, `has_tts_blocks/fragments`, `has_animation_blocks`, `has_galleries`, `has_task_lists`, `has_footnotes`, `has_effects`, `has_classroom`, `has_executable_code`, `has_html_embeds`, structural heading counts, and more
  - `LiaScript_ai_meta.p`: removed experimental `ai:author` and `ai:revisedAuthor` columns; updated to 3,444 rows (93.8% complete)
  - `LiaScript_user_profiles.p`: updated to 445 users (was ~234)
  - `LiaScript_clusters.p`: regenerated with new features; updated cluster names and sizes; `minimalist` criterion now uses 17 interactive/pedagogical features; new K-Means cluster names include "Praktiker (Hardware + Code)", "Playground / Demo-Autoren"
  - Updated embedding model from `jina/jina-embeddings-v2-base-de` to `qwen3-embedding`
  - Added `clusters_only.yaml` pipeline config for standalone cluster regeneration
- **2026-03-13 (v1.4):**
  - Added new dataset: `LiaScript_user_profiles.p` — GitHub author profile metadata (location, company, bio, followers, etc.)
  - Added new dataset: `LiaScript_consolidated.p` — pre-joined wide-format table for convenient analysis (all core datasets merged on `pipe:ID`)
  - Added internal accounts section: documents the 10 known LiaScript core developer accounts and the `internal` boolean field
  - Added `ai:_errors` field documentation in `LiaScript_ai_meta.p`: error tracking dictionary preventing endless retry loops
  - Added pipeline stages 11 (CollectGithubUserProfiles) and 12 (MergeLiaScriptData)
  - Added utility script documentation: `check_internal_coverage.py`, `migrate_error_tracking.py`, `add_internal_repos.py`
  - Added Data Quality Tools section with usage examples and output interpretation
  - Added research examples Q10 (internal vs. community), Q11 (geographic distribution), Q12 (AI metadata completeness)
  - Added Dataset Overview Table for quick reference
- **2026-01-19 (v1.3):**
  - Added new dataset: `LiaScript_clusters.p` with user segmentation and author clustering
  - Added K-Means author clustering using 95 features (37 features + 30 templates + 28 categories)
  - Added author-based analysis to avoid single-author bias (MINT-the-GAP = 40% of docs)
  - Added official LiaTemplates categorization from github.com/LiaTemplates
  - Added documentation gap hypotheses for underutilized features
  - Added template adoption comparison: document-based vs. author-based metrics
  - Key finding: CodeRunner most broadly adopted template (30 authors, 12.8%)
  - Added human-readable report: `LiaScript_clusters.txt`
- **2026-01-03 (v1.2):**
  - Added new dataset: `LiaScript_features.p` (51 columns, 2,751 rows) with detailed feature usage
  - Added new dataset: `LiaScript_feature_statistics.p` (dict) with aggregated statistics
  - Added new AI fields: `ai:education_level`, `ai:target_audience`
  - Updated AI metadata statistics: 2,557/2,751 courses processed (92.9%)
  - Updated pipeline stages: Now 10 stages including AnalyzeLiaScriptFeatures
  - Added feature usage summary table and top 10 templates
  - Updated processing configuration with new force/conditional fields
  - Added new research examples for feature and audience analysis
- **2025-12-31 (v1.1):**
  - Updated AI metadata section with new fields (`ai:keywords_gen`, `ai:author`, `ai:revisedAuthor`)
  - Updated processing configuration with force/conditional processing modes
  - Updated dataset row counts (LiaScript_metadata.p: 2,523, LiaScript_ai_meta.p: 1,312 - in progress)
  - Updated column counts (LiaScript_files_validated.p: 23 columns)
  - Fixed field naming inconsistencies (author: → lia:author, etc.)
  - Added processing status notes for AI metadata
- **2025-12-29 (v1.0):** Initial documentation version
