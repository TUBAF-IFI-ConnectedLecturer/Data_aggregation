import pandas as pd
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import logging
import re

from pipeline.taskfactory import TaskWithInputFileMonitor
from pipeline.taskfactory import loggedExecution


class ExtractLiaScriptMetadata(TaskWithInputFileMonitor):
    """
    Comprehensive extraction of LiaScript metadata from markdown headers.

    This version correctly handles:
    - Multi-line fields (comment, import, tags, etc.)
    - Multiple values for the same field
    - Both import: and link: fields
    - License information from content
    - Logo/icon URLs
    - All standard LiaScript header fields
    - Additional metadata (date, title, tags)
    - Module-specific metadata (module_id, module_type, coding info)
    """

    def __init__(self, config_stage, config_global):
        super().__init__(config_stage, config_global)
        stage_param = config_stage['parameters']
        self.data_folder = Path(config_global['raw_data_folder'])
        self.file_folder = Path(config_global['file_folder'])

        self.lia_files_name = Path(config_global['raw_data_folder']) / stage_param['lia_files_name_input']
        self.lia_metadata_name = Path(config_global['raw_data_folder']) / stage_param['lia_metadata_name']
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

        if Path(self.lia_metadata_name).exists() and not self.force_run:
            df_meta = pd.read_pickle(Path(self.lia_metadata_name))
        else:
            df_meta = pd.DataFrame()

        for index, row in tqdm(df_files.iterrows(), total=df_files.shape[0]):
            # Check if metadata already exists for this file (unless force_run is True)
            if not self.force_run and df_meta.shape[0] > 0:
                if df_meta[df_meta['pipe:ID'] == row['pipe:ID']].shape[0] > 0:
                    continue

            # file starts with <!--
            if row['liaIndi_comment_in_beginning']:
                # Use pipe:ID for consistent file naming
                file_name = f"{row['pipe:ID']}.md"
                file_path = self.file_folder / file_name

                if not file_path.exists():
                    logging.warning(f"File not found: {file_path}")
                    continue

                with open(file_path, "r", encoding='utf-8') as f:
                    content = f.read()

                meta_data = self.extract_metadata(content)
                if meta_data is not None:
                    meta_data['pipe:ID'] = row['pipe:ID']
                    meta_data['id'] = row['id']  # Keep old ID for backwards compatibility
                    df_meta = pd.concat([df_meta, pd.DataFrame([meta_data])])
                    df_meta.to_pickle(Path(self.lia_metadata_name))

        df_meta.reset_index(drop=True, inplace=True)
        df_meta.to_pickle(Path(self.lia_metadata_name))


    def extract_metadata(self, content):
        """
        Extract metadata from LiaScript header comment block.

        Handles:
        - Single-line fields: author, email, version, language, narrator, icon, logo,
          date, title, module metadata, coding metadata
        - Multi-line fields: comment, import, link, script, tags, descriptions
        - License information from content
        """
        # Extract header comment block
        start = content.find("<!--")
        end = content.find("-->")
        if start == -1 or end == -1:
            return None

        header = content[start:end]

        # Initialize metadata dictionary
        meta_data = {}

        # Define field patterns
        # Single-line fields: extract value after colon
        single_line_fields = [
            'author', 'email', 'version', 'language', 'narrator',
            'icon', 'logo', 'mode', 'translation', 'attribute',
            'date', 'title',
            # Module metadata
            'module_id', 'module_type', 'docs_version',
            'estimated_time_in_minutes', 'good_first_module',
            'coding_required', 'coding_level', 'coding_language',
            'collection', 'sequence_name', 'edit'
        ]

        # Multi-line fields: collect all lines until next field or empty line
        multi_line_fields = [
            'comment', 'import', 'link', 'script',
            'tags', 'current_version_description', 'long_description'
        ]

        # Extract single-line fields
        for field in single_line_fields:
            pattern = rf'^{field}:\s*(.+)$'
            matches = re.findall(pattern, header, re.MULTILINE | re.IGNORECASE)
            if matches:
                # Store as list if multiple occurrences
                meta_data[f'lia:{field}'] = matches if len(matches) > 1 else matches[0]

        # Extract multi-line fields
        for field in multi_line_fields:
            values = self._extract_multiline_field(header, field)
            if values:
                meta_data[f'lia:{field}'] = values

        # Extract license information from full content
        license_info = self._extract_license_from_content(content)
        if license_info:
            if 'license' in license_info:
                meta_data['lia:content_license'] = license_info['license']
            if license_info.get('license_url'):
                meta_data['lia:content_license_url'] = license_info['license_url']

        return meta_data if meta_data else None


    def _extract_multiline_field(self, header, field_name):
        """
        Extract multi-line field values.

        A multi-line field starts with "field:" and continues until:
        - An empty line
        - A line starting with another field (contains ":")
        - A line starting with @ (special directive)
        """
        lines = header.split('\n')
        values = []
        in_field = False
        current_value = []

        for line in lines:
            stripped = line.strip()

            # Check if this line starts the field
            if stripped.lower().startswith(f'{field_name}:'):
                in_field = True
                # Get value after colon
                value = stripped.split(':', 1)[1].strip()
                if value:
                    current_value.append(value)
                continue

            # If we're in the field
            if in_field:
                # Check for end conditions
                if (stripped == '' or                    # Empty line
                    stripped.startswith('@') or           # Special directive
                    (':' in stripped and                  # Another field
                     not stripped.startswith('http://') and
                     not stripped.startswith('https://'))):
                    # Save current value and reset
                    if current_value:
                        values.append(' '.join(current_value))
                        current_value = []
                    in_field = False
                else:
                    # Continue collecting the multi-line value
                    if stripped:
                        current_value.append(stripped)

        # Don't forget last value
        if current_value:
            values.append(' '.join(current_value))

        return values if values else None


    def _extract_license_from_content(self, content):
        """
        Extract license information from content (not just header).

        Looks for:
        - CC-BY, CC-BY-SA, CC-BY-NC, CC0
        - MIT, GPL, Apache, BSD
        - License URLs
        """
        content_lower = content.lower()

        # License patterns with priority order
        license_patterns = [
            (r'cc[\s-]?by[\s-]?nc[\s-]?sa|creative\s+commons.*non[\s-]?commercial.*share[\s-]?alike', 'CC-BY-NC-SA'),
            (r'cc[\s-]?by[\s-]?nc[\s-]?nd|creative\s+commons.*non[\s-]?commercial.*no\s+deriv', 'CC-BY-NC-ND'),
            (r'cc[\s-]?by[\s-]?nc|creative\s+commons.*non[\s-]?commercial', 'CC-BY-NC'),
            (r'cc[\s-]?by[\s-]?sa|creative\s+commons.*share[\s-]?alike', 'CC-BY-SA'),
            (r'cc[\s-]?by[\s-]?nd|creative\s+commons.*no\s+deriv', 'CC-BY-ND'),
            (r'cc[\s-]?by|creative\s+commons\s+attribution', 'CC-BY'),
            (r'\bcc0\b|creative\s+commons\s+zero|public\s+domain', 'CC0'),
            (r'\bmit\s+licen[cs]e', 'MIT'),
            (r'apache\s+licen[cs]e\s+2\.0', 'Apache-2.0'),
            (r'\bgpl[\s-]?v?3', 'GPL-3.0'),
            (r'\bgpl[\s-]?v?2', 'GPL-2.0'),
            (r'bsd[\s-]3[\s-]clause', 'BSD-3-Clause'),
            (r'bsd[\s-]2[\s-]clause', 'BSD-2-Clause'),
        ]

        result = {}

        for pattern, license_name in license_patterns:
            if re.search(pattern, content_lower):
                result['license'] = license_name
                break

        # Extract license URL
        license_url_patterns = [
            r'https?://creativecommons\.org/licenses/[^\s\)]+',
            r'https?://opensource\.org/licenses/[^\s\)]+',
            r'https?://www\.gnu\.org/licenses/[^\s\)]+',
        ]

        for pattern in license_url_patterns:
            match = re.search(pattern, content)
            if match:
                result['license_url'] = match.group(0)
                break

        return result if result else None
