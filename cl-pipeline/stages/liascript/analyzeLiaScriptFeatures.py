import pandas as pd
from pathlib import Path
from tqdm import tqdm
import logging
import re
from collections import defaultdict

from pipeline.taskfactory import TaskWithInputFileMonitor
from pipeline.taskfactory import loggedExecution


class AnalyzeLiaScriptFeatures(TaskWithInputFileMonitor):
    """
    Analyze LiaScript feature usage across all documents.

    This stage detects and counts the usage of:
    - Imported templates (import: statements in header)
    - Videos (!?[ syntax - movies)
    - Audio (?[ syntax)
    - Webapps (??[ syntax - interactive embeds)
    - Tables (| syntax for visualization/charts)
    - Quizzes ([[ ]] syntax)
    - Code blocks and projects (``` syntax with various options)
    - Script tags (<script> for JavaScript execution)
    - TTS narrator comments (--{{number}}-- syntax)
    - Animation fragments ({{number}} syntax)
    - External scripts and stylesheets
    - QR codes ([qr-code] syntax)
    - Galleries (multiple media in one paragraph)

    Results are stored both per-document and as aggregated statistics.
    """

    def __init__(self, config_stage, config_global):
        super().__init__(config_stage, config_global)
        stage_param = config_stage['parameters']
        self.data_folder = Path(config_global['raw_data_folder'])
        self.file_folder = Path(config_global['file_folder'])

        self.lia_files_name = Path(config_global['raw_data_folder']) / stage_param['lia_files_name_input']
        self.feature_analysis_name = Path(config_global['raw_data_folder']) / stage_param['feature_analysis_name']
        self.feature_stats_name = Path(config_global['raw_data_folder']) / stage_param['feature_stats_name']
        self.force_run = stage_param.get('force_run', False)


    @loggedExecution
    def execute_task(self):
        df_files = pd.read_pickle(Path(self.lia_files_name))

        # Filter only validated LiaScript files
        if 'pipe:is_valid_liascript' in df_files.columns:
            initial_count = len(df_files)
            df_files = df_files[df_files['pipe:is_valid_liascript'] == True]
            filtered_count = len(df_files)
            logging.info(f"Filtered files: {initial_count} -> {filtered_count} (removed {initial_count - filtered_count} invalid files)")

        if Path(self.feature_analysis_name).exists() and not self.force_run:
            df_features = pd.read_pickle(Path(self.feature_analysis_name))
        else:
            df_features = pd.DataFrame()

        # Statistics aggregation
        feature_counts = defaultdict(int)
        template_usage = defaultdict(int)
        total_documents = 0

        for index, row in tqdm(df_files.iterrows(), total=df_files.shape[0]):
            # Check if feature analysis already exists for this file (unless force_run is True)
            if not self.force_run and df_features.shape[0] > 0:
                if df_features[df_features['pipe:ID'] == row['pipe:ID']].shape[0] > 0:
                    # Load existing features for statistics
                    existing = df_features[df_features['pipe:ID'] == row['pipe:ID']].iloc[0]
                    self._update_statistics(existing, feature_counts, template_usage)
                    total_documents += 1
                    continue

            # Use pipe:ID for consistent file naming
            file_name = f"{row['pipe:ID']}.md"
            file_path = self.file_folder / file_name

            if not file_path.exists():
                logging.warning(f"File not found: {file_path}")
                continue

            with open(file_path, "r", encoding='utf-8') as f:
                content = f.read()

            feature_data = self.analyze_features(content)
            if feature_data is not None:
                feature_data['pipe:ID'] = row['pipe:ID']
                feature_data['repo_name'] = row.get('repo_name', '')
                feature_data['repo_user'] = row.get('repo_user', '')
                feature_data['file_name'] = row.get('file_name', '')

                df_features = pd.concat([df_features, pd.DataFrame([feature_data])])

                # Update statistics
                self._update_statistics(feature_data, feature_counts, template_usage)
                total_documents += 1

                # Save periodically
                if total_documents % 50 == 0:
                    df_features.to_pickle(Path(self.feature_analysis_name))

        df_features.reset_index(drop=True, inplace=True)
        df_features.to_pickle(Path(self.feature_analysis_name))

        # Generate and save statistics
        stats = self._generate_statistics(feature_counts, template_usage, total_documents)
        self._save_statistics(stats)

        logging.info(f"Analyzed {total_documents} documents")
        logging.info(f"Feature usage statistics saved to {self.feature_stats_name}")


    def analyze_features(self, content):
        """
        Analyze LiaScript features in the content.

        Returns a dictionary with feature counts and usage flags.
        """
        features = {}

        # Count videos (!?[ syntax - movie embeds)
        video_count = len(re.findall(r'!\?\[', content))
        features['feature:video_count'] = video_count
        features['feature:has_video'] = video_count > 0

        # Count audio (?[ syntax - NOT ?? which is webapp)
        # Need to exclude ??[ from matching
        audio_count = len(re.findall(r'(?<!\?)\?\[', content))
        features['feature:audio_count'] = audio_count
        features['feature:has_audio'] = audio_count > 0

        # Count webapps (??[ syntax - interactive embeds)
        webapp_count = len(re.findall(r'\?\?\[', content))
        features['feature:webapp_count'] = webapp_count
        features['feature:has_webapp'] = webapp_count > 0

        # Count LiaScript visualization tables (HTML comment ending with --> followed by table)
        # Pattern: -->  (optional whitespace/newlines)  | ... |
        # This detects tables with LiaScript data-type annotations for charts/visualizations
        lia_viz_tables = len(re.findall(r'-->\s*\n\s*\|', content))
        features['feature:lia_viz_table_count'] = lia_viz_tables
        features['feature:has_lia_viz_tables'] = lia_viz_tables > 0

        # Also count regular markdown tables (for comparison)
        # Count table header separators (line with |---|---|)
        table_headers = len(re.findall(r'^\s*\|[\s\-:|]+\|\s*$', content, re.MULTILINE))
        features['feature:table_count'] = table_headers
        features['feature:has_tables'] = table_headers > 0

        # Count different quiz types
        # Text input quiz: [[solution text]] - exclude MC markers [[X]], [[ ]], hints [[?]], and script output [[...]]
        # Pattern excludes: [[X]], [[ ]], [[?]], and lines starting with [[ that are MC quiz items
        text_quiz_count = len(re.findall(r'\[\[(?![Xx ]\]\]|\?\]\])[^\[\]]+\]\]', content))
        # Subtract MC quiz items that may have been matched (those at line start are MC)
        mc_quiz_items = len(re.findall(r'^\s*\[\[[^\[\]]+\]\]', content, re.MULTILINE))
        text_quiz_count = max(0, text_quiz_count - mc_quiz_items)
        features['feature:text_quiz_count'] = text_quiz_count
        features['feature:has_text_quiz'] = text_quiz_count > 0

        # Single choice quiz: [( )] or [(X)]
        single_choice_count = len(re.findall(r'^\s*\[\([Xx ]\)\]', content, re.MULTILINE))
        features['feature:single_choice_count'] = single_choice_count
        features['feature:has_single_choice'] = single_choice_count > 0

        # Multiple choice quiz: [[ ]] or [[X]]
        multiple_choice_count = len(re.findall(r'^\s*\[\[[Xx ]\]\]', content, re.MULTILINE))
        features['feature:multiple_choice_count'] = multiple_choice_count
        features['feature:has_multiple_choice'] = multiple_choice_count > 0

        # Quiz hints: [[?]]
        quiz_hint_count = len(re.findall(r'\[\[\?\]\]', content))
        features['feature:quiz_hint_count'] = quiz_hint_count
        features['feature:has_quiz_hints'] = quiz_hint_count > 0

        # Total quiz elements (any type)
        total_quiz_count = text_quiz_count + single_choice_count + multiple_choice_count
        features['feature:total_quiz_count'] = total_quiz_count
        features['feature:has_quiz'] = total_quiz_count > 0

        # Count code blocks (``` with or without language specification)
        # Matches: ```python, ```js, ``` (without language), ```+ (LiaScript project marker)
        code_block_count = len(re.findall(r'^```[\w+]*\s*$', content, re.MULTILINE))
        features['feature:code_block_count'] = code_block_count
        features['feature:has_code_blocks'] = code_block_count > 0

        # Count code projects specifically (```+ or @file.ext markers)
        code_project_count = len(re.findall(r'^```\w*\+', content, re.MULTILINE))
        features['feature:code_project_count'] = code_project_count
        features['feature:has_code_projects'] = code_project_count > 0

        # Count script tags (<script>...</script>)
        script_tag_count = len(re.findall(r'<script[^>]*>.*?</script>', content, re.DOTALL | re.IGNORECASE))
        features['feature:script_tag_count'] = script_tag_count
        features['feature:has_script_tags'] = script_tag_count > 0

        # Extract imported templates
        templates = self._extract_templates(content)
        features['feature:import_count'] = len(templates)
        features['feature:has_imports'] = len(templates) > 0
        features['feature:imported_templates'] = templates if templates else None

        # Check for narrator (TTS configuration in header)
        has_narrator = self._has_header_field(content, 'narrator')
        features['feature:has_narrator'] = has_narrator

        # Count TTS narrator fragments (--{{number}}-- or --{{number-number}}-- syntax)
        # Matches: --{{1}}--, --{{2}}--, --{{1-3}}-- etc.
        tts_fragment_count = len(re.findall(r'--\{\{\d+(?:-\d+)?\}\}--', content))
        features['feature:tts_fragment_count'] = tts_fragment_count
        features['feature:has_tts_fragments'] = tts_fragment_count > 0

        # Count TTS narrator blocks (multiline blocks with **** delimiters)
        # Pattern: --{{number}}-- or --{{number-number}}-- followed by newline and ****
        tts_block_count = len(re.findall(r'--\{\{\d+(?:-\d+)?\}\}--\s*\n\*{4,}', content))
        features['feature:tts_block_count'] = tts_block_count
        features['feature:has_tts_blocks'] = tts_block_count > 0

        # Count animation fragments ({{number}} or {{number-number}} syntax - not preceded by --)
        # Matches: {{1}}, {{2}}, {{1-3}}, {{0-5}} etc.
        animation_fragment_count = len(re.findall(r'(?<!--)\{\{\d+(?:-\d+)?\}\}', content))
        features['feature:animation_fragment_count'] = animation_fragment_count
        features['feature:has_animation_fragments'] = animation_fragment_count > 0

        # Count animation blocks (multiline blocks with **** delimiters)
        # Pattern: {{number}} or {{number-number}} followed by newline and ****
        animation_block_count = len(re.findall(r'\{\{\d+(?:-\d+)?\}\}\s*\n\*{4,}', content))
        features['feature:animation_block_count'] = animation_block_count
        features['feature:has_animation_blocks'] = animation_block_count > 0

        # Count animate.css animations (class="animated ..." or animate__)
        animated_css_count = len(re.findall(r'class=["\'"][^"\']*\banimated\b|animate__\w+', content))
        features['feature:animated_css_count'] = animated_css_count
        features['feature:has_animated_css'] = animated_css_count > 0

        # Count images (![alt](url) syntax - NOT !?[ which is video)
        image_count = len(re.findall(r'(?<!\?)\!\[.*?\]\(.*?\)', content))
        features['feature:image_count'] = image_count
        features['feature:has_images'] = image_count > 0

        # Count external scripts in header (script: URL)
        header_scripts = self._count_header_field(content, 'script')
        features['feature:external_script_count'] = header_scripts
        features['feature:has_external_scripts'] = header_scripts > 0

        # Count external CSS links in header (link: URL)
        header_links = self._count_header_field(content, 'link')
        features['feature:external_css_count'] = header_links
        features['feature:has_external_css'] = header_links > 0

        # Count formulas ($ ... $ and $$ ... $$)
        # First remove code blocks to avoid false positives from shell variables ($PATH, $HOME)
        content_no_code = re.sub(r'```.*?```', '', content, flags=re.DOTALL)
        # Also remove inline code to avoid false positives
        content_no_code = re.sub(r'`[^`]+`', '', content_no_code)
        # Inline math: $...$ but not $$ and not currency like $10.99
        inline_math_count = len(re.findall(r'(?<!\$)\$(?!\$)[^$\n]+\$(?!\$)', content_no_code))
        # Display math: $$...$$
        display_math_count = len(re.findall(r'\$\$[^$]+\$\$', content_no_code, re.DOTALL))
        features['feature:inline_math_count'] = inline_math_count
        features['feature:display_math_count'] = display_math_count
        features['feature:has_math'] = (inline_math_count + display_math_count) > 0

        # Check for logo in header (logo: URL)
        has_logo = self._has_header_field(content, 'logo')
        features['feature:has_logo'] = has_logo

        # Check for icon in header (icon: URL)
        has_icon = self._has_header_field(content, 'icon')
        features['feature:has_icon'] = has_icon

        # ===== NEW FEATURES =====

        # Count QR codes ([qr-code](data) or qr-code syntax)
        qr_code_count = len(re.findall(r'\[qr-code\]|\[qr-code\s*\(', content, re.IGNORECASE))
        features['feature:qr_code_count'] = qr_code_count
        features['feature:has_qr_codes'] = qr_code_count > 0

        # Count surveys/polls ([(text)] where text is not X or space - Likert scale items)
        # Survey items have text descriptions instead of X/ for selection
        survey_count = len(re.findall(r'^\s*\[\([^Xx ][^\)]+\)\]', content, re.MULTILINE))
        features['feature:survey_count'] = survey_count
        features['feature:has_surveys'] = survey_count > 0

        # Count footnotes ([^1], [^note], etc.)
        footnote_ref_count = len(re.findall(r'\[\^[^\]]+\]', content))
        footnote_def_count = len(re.findall(r'^\[\^[^\]]+\]:', content, re.MULTILINE))
        features['feature:footnote_count'] = footnote_ref_count
        features['feature:footnote_def_count'] = footnote_def_count
        features['feature:has_footnotes'] = footnote_ref_count > 0

        # Count macro calls (@macroname or @macroname(...))
        # Exclude common false positives like email addresses
        macro_count = len(re.findall(r'(?<![a-zA-Z0-9.])@[a-zA-Z_][a-zA-Z0-9_]*(?:\s*\(|(?=\s|$|[^a-zA-Z0-9_@.]))', content))
        features['feature:macro_count'] = macro_count
        features['feature:has_macros'] = macro_count > 0

        # Count effect annotations (<!-- effect="..." --> or data-effect)
        effect_count = len(re.findall(r'effect\s*=\s*["\']|data-effect\s*=', content, re.IGNORECASE))
        features['feature:effect_count'] = effect_count
        features['feature:has_effects'] = effect_count > 0

        # Count classroom/collaborative features (@classroom)
        classroom_count = len(re.findall(r'@classroom', content, re.IGNORECASE))
        features['feature:classroom_count'] = classroom_count
        features['feature:has_classroom'] = classroom_count > 0

        # Count ASCII art diagrams (``` followed by ascii, diagram, art etc.)
        ascii_diagram_count = len(re.findall(r'^```\s*(ascii|diagram|art|svgbob)\s*$', content, re.MULTILINE | re.IGNORECASE))
        features['feature:ascii_diagram_count'] = ascii_diagram_count
        features['feature:has_ascii_diagrams'] = ascii_diagram_count > 0

        # Count matrix quizzes (complex quiz patterns with multiple rows)
        # Matrix quiz: multiple [[...]] on consecutive lines with same structure
        matrix_quiz_count = len(re.findall(r'(\[\[[^\]]+\]\]\s*)+\n(\[\[[^\]]+\]\]\s*)+', content))
        features['feature:matrix_quiz_count'] = matrix_quiz_count
        features['feature:has_matrix_quiz'] = matrix_quiz_count > 0

        # Count headings for structural analysis
        h1_count = len(re.findall(r'^#\s+[^#]', content, re.MULTILINE))
        h2_count = len(re.findall(r'^##\s+[^#]', content, re.MULTILINE))
        h3_count = len(re.findall(r'^###\s+[^#]', content, re.MULTILINE))
        features['feature:h1_count'] = h1_count
        features['feature:h2_count'] = h2_count
        features['feature:h3_count'] = h3_count
        features['feature:total_headings'] = h1_count + h2_count + h3_count

        # Count links (for reference density analysis)
        link_count = len(re.findall(r'\[([^\]]+)\]\(([^\)]+)\)', content))
        features['feature:link_count'] = link_count
        features['feature:has_links'] = link_count > 0

        # Count comments (<!-- ... --> in body, not header)
        # Find all comments, then subtract header comment
        all_comments = re.findall(r'<!--.*?-->', content, re.DOTALL)
        # First comment is usually the header
        body_comment_count = max(0, len(all_comments) - 1)
        features['feature:comment_count'] = body_comment_count
        features['feature:has_comments'] = body_comment_count > 0

        # Count galleries (multiple images in same paragraph/line)
        # Pattern: two or more images on same or adjacent lines
        gallery_count = len(re.findall(r'!\[.*?\]\(.*?\)\s*!\[.*?\]\(.*?\)', content))
        features['feature:gallery_count'] = gallery_count
        features['feature:has_galleries'] = gallery_count > 0

        return features


    def _normalize_template_url(self, url):
        """
        Normalize template URL to avoid duplicates from different URL variants.

        Handles:
        - Case differences (LiaScript vs liascript)
        - refs/heads/main vs main
        - Different GitHub URL formats
        """
        normalized = url.lower()
        # Remove refs/heads/ prefix
        normalized = re.sub(r'/refs/heads/', '/', normalized)
        # Standardize raw.githubusercontent.com URLs
        normalized = re.sub(r'github\.com/([^/]+)/([^/]+)/blob/', r'raw.githubusercontent.com/\1/\2/', normalized)
        return normalized

    def _extract_templates(self, content):
        """
        Extract all imported templates from the header.

        Returns a list of template URLs (normalized for deduplication in statistics).
        """
        # Extract header comment block
        start = content.find("<!--")
        end = content.find("-->")
        if start == -1 or end == -1:
            return []

        header = content[start:end]

        # Find all import statements
        templates = []
        lines = header.split('\n')
        in_import = False

        for line in lines:
            stripped = line.strip()

            # Check if this line starts an import
            if stripped.lower().startswith('import:'):
                in_import = True
                # Get value after colon
                value = stripped.split(':', 1)[1].strip()
                if value and value.startswith('http'):
                    templates.append(self._normalize_template_url(value))
                continue

            # If we're in import and line continues (URL on next line)
            if in_import:
                if stripped == '' or ':' in stripped:
                    in_import = False
                elif stripped.startswith('http'):
                    templates.append(self._normalize_template_url(stripped))

        return templates


    def _count_header_field(self, content, field_name):
        """
        Count occurrences of a field in the header (e.g., 'script:', 'link:').

        Handles multi-line definitions where URLs continue on following lines:
        script: https://example.com/lib1.js
                https://example.com/lib2.js

        Returns count of URLs found.
        """
        # Extract header comment block
        start = content.find("<!--")
        end = content.find("-->")
        if start == -1 or end == -1:
            return 0

        header = content[start:end]

        # Count URLs in field (including continuation lines)
        url_count = 0
        lines = header.split('\n')
        in_field = False

        for line in lines:
            stripped = line.strip()

            # Check if this line starts the field
            if stripped.lower().startswith(f'{field_name}:'):
                in_field = True
                # Get value after colon
                value = stripped.split(':', 1)[1].strip()
                if value and value.startswith('http'):
                    url_count += 1
                continue

            # If we're in field and line continues (URL on next line)
            if in_field:
                if stripped == '' or ':' in stripped:
                    in_field = False
                elif stripped.startswith('http'):
                    url_count += 1

        return url_count


    def _has_header_field(self, content, field_name):
        """
        Check if a field exists in the header (e.g., 'logo:', 'icon:').

        Returns True if field is present, False otherwise.
        """
        # Extract header comment block
        start = content.find("<!--")
        end = content.find("-->")
        if start == -1 or end == -1:
            return False

        header = content[start:end]

        # Check if field exists
        pattern = rf'^\s*{field_name}\s*:'
        return bool(re.search(pattern, header, re.MULTILINE | re.IGNORECASE))


    def _update_statistics(self, feature_data, feature_counts, template_usage):
        """Update aggregated statistics from a single document's features."""
        # Count documents using each feature
        if feature_data.get('feature:has_video', False):
            feature_counts['video'] += 1
        if feature_data.get('feature:has_audio', False):
            feature_counts['audio'] += 1
        if feature_data.get('feature:has_webapp', False):
            feature_counts['webapp'] += 1
        if feature_data.get('feature:has_tables', False):
            feature_counts['tables'] += 1
        if feature_data.get('feature:has_lia_viz_tables', False):
            feature_counts['lia_viz_tables'] += 1
        if feature_data.get('feature:has_quiz', False):
            feature_counts['quiz'] += 1
        if feature_data.get('feature:has_text_quiz', False):
            feature_counts['text_quiz'] += 1
        if feature_data.get('feature:has_single_choice', False):
            feature_counts['single_choice'] += 1
        if feature_data.get('feature:has_multiple_choice', False):
            feature_counts['multiple_choice'] += 1
        if feature_data.get('feature:has_quiz_hints', False):
            feature_counts['quiz_hints'] += 1
        if feature_data.get('feature:has_code_blocks', False):
            feature_counts['code_blocks'] += 1
        if feature_data.get('feature:has_script_tags', False):
            feature_counts['script_tags'] += 1
        if feature_data.get('feature:has_narrator', False):
            feature_counts['narrator'] += 1
        if feature_data.get('feature:has_tts_fragments', False):
            feature_counts['tts_fragments'] += 1
        if feature_data.get('feature:has_tts_blocks', False):
            feature_counts['tts_blocks'] += 1
        if feature_data.get('feature:has_animation_fragments', False):
            feature_counts['animation_fragments'] += 1
        if feature_data.get('feature:has_animation_blocks', False):
            feature_counts['animation_blocks'] += 1
        if feature_data.get('feature:has_animated_css', False):
            feature_counts['animated_css'] += 1
        if feature_data.get('feature:has_images', False):
            feature_counts['images'] += 1
        if feature_data.get('feature:has_external_scripts', False):
            feature_counts['external_scripts'] += 1
        if feature_data.get('feature:has_external_css', False):
            feature_counts['external_css'] += 1
        if feature_data.get('feature:has_math', False):
            feature_counts['math'] += 1
        if feature_data.get('feature:has_logo', False):
            feature_counts['logo'] += 1
        if feature_data.get('feature:has_icon', False):
            feature_counts['icon'] += 1

        # NEW FEATURES
        if feature_data.get('feature:has_qr_codes', False):
            feature_counts['qr_codes'] += 1
        if feature_data.get('feature:has_surveys', False):
            feature_counts['surveys'] += 1
        if feature_data.get('feature:has_footnotes', False):
            feature_counts['footnotes'] += 1
        if feature_data.get('feature:has_macros', False):
            feature_counts['macros'] += 1
        if feature_data.get('feature:has_effects', False):
            feature_counts['effects'] += 1
        if feature_data.get('feature:has_classroom', False):
            feature_counts['classroom'] += 1
        if feature_data.get('feature:has_ascii_diagrams', False):
            feature_counts['ascii_diagrams'] += 1
        if feature_data.get('feature:has_matrix_quiz', False):
            feature_counts['matrix_quiz'] += 1
        if feature_data.get('feature:has_links', False):
            feature_counts['links'] += 1
        if feature_data.get('feature:has_comments', False):
            feature_counts['comments'] += 1
        if feature_data.get('feature:has_galleries', False):
            feature_counts['galleries'] += 1
        if feature_data.get('feature:has_code_projects', False):
            feature_counts['code_projects'] += 1

        # Count template usage
        templates = feature_data.get('feature:imported_templates', [])
        if templates:
            for template in templates:
                template_usage[template] += 1


    def _generate_statistics(self, feature_counts, template_usage, total_documents):
        """Generate comprehensive statistics summary."""
        stats = {
            'total_documents': total_documents,
            'timestamp': pd.Timestamp.now(),
            'feature_usage': {},
            'template_usage': {},
            'summary': {}
        }

        # Feature usage percentages
        for feature, count in feature_counts.items():
            percentage = (count / total_documents * 100) if total_documents > 0 else 0
            stats['feature_usage'][feature] = {
                'count': count,
                'percentage': round(percentage, 2)
            }

        # Template usage (sorted by frequency)
        stats['template_usage'] = dict(sorted(template_usage.items(),
                                              key=lambda x: x[1],
                                              reverse=True))

        # Summary
        stats['summary']['most_common_feature'] = max(feature_counts.items(),
                                                       key=lambda x: x[1])[0] if feature_counts else None
        stats['summary']['total_templates'] = len(template_usage)
        stats['summary']['most_used_template'] = max(template_usage.items(),
                                                      key=lambda x: x[1])[0] if template_usage else None

        return stats


    def _save_statistics(self, stats):
        """Save statistics to file and log key findings."""
        # Save as pickle
        pd.to_pickle(stats, self.feature_stats_name)

        # Save as readable text file
        text_file = self.feature_stats_name.with_suffix('.txt')
        with open(text_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("LiaScript Feature Usage Statistics\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"Total Documents Analyzed: {stats['total_documents']}\n")
            f.write(f"Analysis Timestamp: {stats['timestamp']}\n\n")

            f.write("-" * 80 + "\n")
            f.write("Feature Usage\n")
            f.write("-" * 80 + "\n")
            for feature, data in sorted(stats['feature_usage'].items(),
                                       key=lambda x: x[1]['count'],
                                       reverse=True):
                f.write(f"{feature:20s}: {data['count']:5d} documents ({data['percentage']:6.2f}%)\n")

            f.write("\n" + "-" * 80 + "\n")
            f.write(f"Template Usage (Top 20 of {stats['summary']['total_templates']} total)\n")
            f.write("-" * 80 + "\n")
            for i, (template, count) in enumerate(list(stats['template_usage'].items())[:20], 1):
                percentage = (count / stats['total_documents'] * 100) if stats['total_documents'] > 0 else 0
                f.write(f"{i:2d}. {template}\n")
                f.write(f"    Used in {count} documents ({percentage:.2f}%)\n\n")

        logging.info(f"Statistics saved to {text_file}")
