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
        # Text input quiz: [[solution text]]
        text_quiz_count = len(re.findall(r'\[\[[^\[\]]+\]\]', content))
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

        # Count code blocks (``` language)
        code_block_count = len(re.findall(r'^```\w+', content, re.MULTILINE))
        features['feature:code_block_count'] = code_block_count
        features['feature:has_code_blocks'] = code_block_count > 0

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
        has_narrator = bool(re.search(r'narrator\s*:', content[:2000], re.IGNORECASE))
        features['feature:has_narrator'] = has_narrator

        # Count TTS narrator comments (--{{number}}-- syntax)
        tts_comment_count = len(re.findall(r'--\{\{\d+\}\}--', content))
        features['feature:tts_comment_count'] = tts_comment_count
        features['feature:has_tts_comments'] = tts_comment_count > 0

        # Count animation fragments ({{number}} syntax - not preceded by --)
        animation_fragment_count = len(re.findall(r'(?<!--)\{\{\d+\}\}', content))
        features['feature:animation_fragment_count'] = animation_fragment_count
        features['feature:has_animation_fragments'] = animation_fragment_count > 0

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

        # Count QR codes ([qr-code](url))
        qr_code_count = len(re.findall(r'\[qr-code\]\(.*?\)', content, re.IGNORECASE))
        features['feature:qr_code_count'] = qr_code_count
        features['feature:has_qr_codes'] = qr_code_count > 0

        # Count preview-lia links ([preview-lia](url))
        preview_lia_count = len(re.findall(r'\[preview-lia\]\(.*?\)', content, re.IGNORECASE))
        features['feature:preview_lia_count'] = preview_lia_count
        features['feature:has_preview_lia'] = preview_lia_count > 0

        # Count formulas ($ ... $ and $$ ... $$)
        inline_math_count = len(re.findall(r'\$[^$]+\$', content))
        display_math_count = len(re.findall(r'\$\$[^$]+\$\$', content, re.DOTALL))
        features['feature:inline_math_count'] = inline_math_count
        features['feature:display_math_count'] = display_math_count
        features['feature:has_math'] = (inline_math_count + display_math_count) > 0

        return features


    def _extract_templates(self, content):
        """
        Extract all imported templates from the header.

        Returns a list of template URLs.
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
                    templates.append(value)
                continue

            # If we're in import and line continues (URL on next line)
            if in_import:
                if stripped == '' or ':' in stripped:
                    in_import = False
                elif stripped.startswith('http'):
                    templates.append(stripped)

        return templates


    def _count_header_field(self, content, field_name):
        """
        Count occurrences of a field in the header (e.g., 'script:', 'link:').

        Returns count of field occurrences.
        """
        # Extract header comment block
        start = content.find("<!--")
        end = content.find("-->")
        if start == -1 or end == -1:
            return 0

        header = content[start:end]

        # Count field occurrences
        pattern = rf'^{field_name}\s*:'
        matches = re.findall(pattern, header, re.MULTILINE | re.IGNORECASE)
        return len(matches)


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
        if feature_data.get('feature:has_tts_comments', False):
            feature_counts['tts_comments'] += 1
        if feature_data.get('feature:has_animation_fragments', False):
            feature_counts['animation_fragments'] += 1
        if feature_data.get('feature:has_animated_css', False):
            feature_counts['animated_css'] += 1
        if feature_data.get('feature:has_images', False):
            feature_counts['images'] += 1
        if feature_data.get('feature:has_external_scripts', False):
            feature_counts['external_scripts'] += 1
        if feature_data.get('feature:has_external_css', False):
            feature_counts['external_css'] += 1
        if feature_data.get('feature:has_qr_codes', False):
            feature_counts['qr_codes'] += 1
        if feature_data.get('feature:has_preview_lia', False):
            feature_counts['preview_lia'] += 1
        if feature_data.get('feature:has_math', False):
            feature_counts['math'] += 1

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
