"""
LiaScript Validation Stage - AI-based False Positive Filtering

This stage validates potential LiaScript files using AI to filter out false positives.
It uses a two-stage approach:
1. Rule-based pre-filtering (strong indicators pass automatically)
2. AI-based validation (for files with weak indicators)
"""

import pandas as pd
import logging
from pathlib import Path
from tqdm import tqdm
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate

from pipeline.taskfactory import TaskWithInputFileMonitor

# Import zentrale Logging-Konfiguration
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))
from pipeline_logging import setup_stage_logging


class ValidateLiaScriptFiles(TaskWithInputFileMonitor):
    """
    Validates LiaScript files using AI to reduce false positives.

    Strategy:
    - Files with strong LiaScript indicators pass automatically
    - Files with weak indicators are validated by AI
    - Invalid files are marked as pipe:is_valid_liascript = False
    """

    def __init__(self, config_stage, config_global):
        super().__init__(config_stage, config_global)

        # Setup zentrale Logging-Konfiguration
        self.logger_configurator = setup_stage_logging(config_global)

        # Initialize configuration
        stage_param = config_stage['parameters']
        self._setup_paths_and_config(config_global, stage_param)

        # Initialize LLM
        self._initialize_llm()

    def _setup_paths_and_config(self, config_global, stage_param):
        """Setup file paths and basic configuration"""
        self.file_name_input = Path(config_global['raw_data_folder']) / stage_param['file_name_input']
        self.file_name_output = Path(config_global['raw_data_folder']) / stage_param['file_name_output']
        self.file_folder = Path(config_global['file_folder'])

        # LLM configuration
        self.llm_config = stage_param.get('llm_config', {})
        self.base_url = self.llm_config.get('base_url', 'http://localhost:11434')
        self.llm_model = stage_param.get('model_name', 'llama3.3:70b')
        self.timeout_seconds = self.llm_config.get('timeout_seconds', 120)

        # Validation configuration
        self.max_content_length = stage_param.get('max_content_length', 8000)
        self.validate_all = stage_param.get('validate_all', False)  # Force validation of all files

    def _initialize_llm(self):
        """Initialize Ollama LLM"""
        self.llm = OllamaLLM(
            base_url=self.base_url,
            model=self.llm_model,
            temperature=0,
            timeout=self.timeout_seconds
        )

        # Create validation prompt
        self.validation_prompt = PromptTemplate.from_template(
            """Du bist ein Experte für LiaScript, ein interaktives Markdown-Format für Bildungsinhalte.

Analysiere die folgende Markdown-Datei und entscheide, ob es sich um ein echtes LiaScript-Dokument handelt.

ECHTE LiaScript-Dokumente haben typischerweise:
- HTML-Kommentar-Header am Anfang (<!--) mit LiaScript-Metadaten wie:
  * narrator: oder voice: (Text-to-Speech Einstellungen)
  * version: (Versionsnummer)
  * import: (LiaScript-Module imports)
  * author:, email:, comment:
- LiaScript-spezifische Syntax wie:
  * @eval, @.eval (Code-Ausführung)
  * Interaktive Quizze mit [[ ]] oder [( )]
  * Script-Tags mit @input oder @output
  * LiaScript-Animationen mit --{{
  * Video-Einbettung mit !?[Beschreibung](URL)
- Import-Statements im Header (import: https://...)
- LiaScript-Kurs-Links (LiaScript.github.io/course/?)
- Typische LiaScript-Struktur mit Kapiteln und interaktiven Elementen

HINWEIS: Ein HTML-Kommentar am Anfang allein ist KEIN ausreichender Beweis.
Viele normale Markdown-Dokumente haben HTML-Kommentare für andere Zwecke.

KEINE echten LiaScript-Dokumente sind:
- Blog-Posts oder README-Dateien, die nur ÜBER LiaScript schreiben
- Normale Dokumentation ÜBER LiaScript (Anleitungen, Tutorials zum Erstellen von LiaScript)
- Markdown-Dateien, die nur das Wort "LiaScript" erwähnen
- GitHub Issues oder Diskussionen über LiaScript
- Normale Markdown-Dokumente ohne LiaScript-spezifische Syntax
- Dateien mit HTML-Kommentaren, aber ohne LiaScript-Metadaten

WICHTIG - Besonders vorsichtig bei:
- Wenn "LiaScript" im Dateinamen oder Hauptüberschrift (# ...) steht:
  * Ist es ein Tutorial/Anleitung ÜBER LiaScript selbst? → FALSE
  * Oder ist es ein Tutorial zu einem anderen Thema, das MIT LiaScript erstellt wurde? → TRUE
  * Prüfe den Inhalt: Erklärt es WIE man LiaScript benutzt, oder NUTZT es LiaScript?

DATEINAME: {filename}

INHALT (erste {max_length} Zeichen):
{content}

Antworte nur mit einem einzigen Wort:
- TRUE wenn es ein echtes LiaScript-Dokument ist
- FALSE wenn es kein LiaScript-Dokument ist

Antwort:"""
        )

    def _has_strong_indicators(self, row):
        """
        Check if file has strong LiaScript indicators (auto-pass).

        Strong indicators:
        - Uses LiaTemplates
        - Has LiaScript button/badge
        - Has 'liascript' in filename
        - Has 'liascript' in H1 heading
        - Has import: statement in header (LiaScript-specific)
        - Has narrator: statement in header (LiaScript-specific)
        - Has video syntax !?[...](url) (LiaScript-specific)
        """
        return (
            row.get('liaIndi_liaTemplates_used', False) or
            row.get('liaIndi_lia_button', False) or
            row.get('liaIndi_Lia_in_filename', False) or
            row.get('liaIndi_lia_in_h1', False) or
            row.get('liaIndi_import_statement', False) or
            row.get('liaIndi_narrator_statement', False) or
            row.get('liaIndi_video_syntax', False)
        )

    def _has_weak_indicators(self, row):
        """
        Check if file has weak LiaScript indicators (needs AI validation).

        Weak indicators:
        - Only has 'liascript' in content
        - Only has HTML comment at beginning
        - Has version: statement in header
        """
        return (
            row.get('liaIndi_liascript_in_content', False) or
            row.get('liaIndi_comment_in_beginning', False) or
            row.get('liaIndi_version_statement', False)
        )

    def _has_no_indicators(self, row):
        """
        Check if file has NO LiaScript indicators at all.

        Files without any indicators are automatically rejected.
        """
        return not (
            # Strong indicators
            row.get('liaIndi_liaTemplates_used', False) or
            row.get('liaIndi_lia_button', False) or
            row.get('liaIndi_Lia_in_filename', False) or
            row.get('liaIndi_lia_in_h1', False) or
            row.get('liaIndi_import_statement', False) or
            row.get('liaIndi_narrator_statement', False) or
            row.get('liaIndi_video_syntax', False) or
            # Weak indicators
            row.get('liaIndi_liascript_in_content', False) or
            row.get('liaIndi_comment_in_beginning', False) or
            row.get('liaIndi_version_statement', False)
        )

    def _read_file_content(self, file_id, file_type):
        """Read content from markdown file"""
        file_path = self.file_folder / f"{file_id}.{file_type}"

        if not file_path.exists():
            logging.warning(f"File not found: {file_path}")
            return None

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return content
        except Exception as e:
            logging.error(f"Error reading file {file_path}: {e}")
            return None

    def _validate_with_ai(self, file_id, file_type, filename):
        """
        Validate file content using AI.

        Returns:
            bool: True if valid LiaScript, False otherwise
        """
        # Read file content
        content = self._read_file_content(file_id, file_type)

        if content is None:
            return False

        # Truncate content if too long
        content_preview = content[:self.max_content_length]

        # Create prompt
        prompt_text = self.validation_prompt.format(
            filename=filename,
            content=content_preview,
            max_length=self.max_content_length
        )

        try:
            # Get AI response
            response = self.llm.invoke(prompt_text)

            # Parse response
            response_clean = response.strip().upper()

            # Check for TRUE/FALSE
            if 'TRUE' in response_clean:
                return True
            elif 'FALSE' in response_clean:
                return False
            else:
                logging.warning(f"Unexpected AI response for {filename}: {response}")
                return False  # Default to False for unclear responses

        except Exception as e:
            logging.error(f"Error validating file {filename} with AI: {e}")
            return False

    def _validate_file(self, row):
        """
        Validate a single file.

        Returns:
            tuple: (is_valid, validation_method)
                - is_valid: True if valid LiaScript
                - validation_method: 'strong_indicator', 'ai_validated', 'ai_rejected', 'no_indicators'
        """
        # Check for strong indicators (auto-pass)
        if self._has_strong_indicators(row):
            return True, 'strong_indicator'

        # Check if file has NO indicators (auto-reject)
        if self._has_no_indicators(row) and not self.validate_all:
            return False, 'no_indicators'

        # Check for weak indicators (needs AI validation)
        if self._has_weak_indicators(row) or self.validate_all:
            filename = row.get('file_name', f"{row['pipe:ID']}.md")
            is_valid = self._validate_with_ai(row['pipe:ID'], row['pipe:file_type'], filename)

            if is_valid:
                return True, 'ai_validated'
            else:
                return False, 'ai_rejected'

        # Should not reach here, but default to invalid
        return False, 'no_indicators'

    def _load_previous_validations(self):
        """
        Load validation results from previous run if output file exists.

        Returns:
            dict: Dictionary mapping pipe:ID to (is_valid, validation_method) tuples
        """
        if not self.file_name_output.exists():
            logging.info("No previous validation results found")
            return {}

        try:
            df_previous = pd.read_pickle(self.file_name_output)

            # Check if validation columns exist
            if 'pipe:is_valid_liascript' not in df_previous.columns:
                logging.info("Previous output has no validation results")
                return {}

            # Create lookup dictionary
            validation_cache = {}
            for _, row in df_previous.iterrows():
                if row.get('pipe:is_valid_liascript') is not None:
                    file_id = row['pipe:ID']
                    is_valid = row['pipe:is_valid_liascript']
                    method = row.get('pipe:validation_method', 'unknown')
                    validation_cache[file_id] = (is_valid, method)

            logging.info(f"Loaded {len(validation_cache)} previous validation results")
            return validation_cache

        except Exception as e:
            logging.warning(f"Could not load previous validation results: {e}")
            return {}

    def execute_task(self):
        """Main execution method"""
        logging.info("Starting LiaScript validation")

        # Load input data
        df_files = pd.read_pickle(self.file_name_input)
        initial_count = len(df_files)
        logging.info(f"Loaded {initial_count} files to validate")

        # Load previous validation results
        validation_cache = self._load_previous_validations()
        reused_count = 0

        # Statistics
        stats = {
            'strong_indicator': 0,
            'ai_validated': 0,
            'ai_rejected': 0,
            'no_indicators': 0,
            'cached': 0
        }

        # Add validation column if not exists
        if 'pipe:is_valid_liascript' not in df_files.columns:
            df_files['pipe:is_valid_liascript'] = None

        if 'pipe:validation_method' not in df_files.columns:
            df_files['pipe:validation_method'] = None

        # Transfer previous validation results for existing files
        if validation_cache and not self.validate_all:
            for idx, row in df_files.iterrows():
                file_id = row['pipe:ID']
                if file_id in validation_cache and row.get('pipe:is_valid_liascript') is None:
                    is_valid, method = validation_cache[file_id]
                    df_files.at[idx, 'pipe:is_valid_liascript'] = is_valid
                    df_files.at[idx, 'pipe:validation_method'] = method
                    stats['cached'] += 1
                    reused_count += 1

            logging.info(f"Reused {reused_count} validation results from previous run")

        # Validate each file
        print("\n=== LiaScript Validation ===\n")

        for idx, row in tqdm(df_files.iterrows(), total=len(df_files), desc="Validating files"):
            # Skip if already validated (unless validate_all is True)
            if not self.validate_all and row.get('pipe:is_valid_liascript') is not None:
                stats[row.get('pipe:validation_method', 'unknown')] = stats.get(row.get('pipe:validation_method', 'unknown'), 0) + 1
                continue

            # Validate file
            is_valid, method = self._validate_file(row)

            # Update dataframe
            df_files.at[idx, 'pipe:is_valid_liascript'] = is_valid
            df_files.at[idx, 'pipe:validation_method'] = method

            # Update statistics
            stats[method] += 1

            # Log result
            filename = row.get('file_name', row['pipe:ID'])
            repo_path = f"{row.get('repo_user', '')}/{row.get('repo_name', '')}"

            if method == 'ai_validated':
                logging.info(f"✓ AI validated: {filename} (repo: {repo_path})")
            elif method == 'ai_rejected':
                logging.info(f"✗ AI rejected: {filename} (repo: {repo_path})")

        # Save results
        df_files.to_pickle(self.file_name_output)

        # Print statistics
        valid_count = len(df_files[df_files['pipe:is_valid_liascript'] == True])
        invalid_count = len(df_files[df_files['pipe:is_valid_liascript'] == False])

        print(f"\n{'='*70}")
        print(f"=== VALIDATION STATISTICS ===")
        print(f"Total files: {initial_count}")
        print(f"Valid LiaScript files: {valid_count} ({valid_count/initial_count*100:.1f}%)")
        print(f"Invalid files (filtered out): {invalid_count} ({invalid_count/initial_count*100:.1f}%)")
        print(f"\nValidation methods:")
        print(f"  - Cached (reused from previous run): {stats['cached']}")
        print(f"  - Strong indicators (auto-pass): {stats['strong_indicator']}")
        print(f"  - AI validated (weak indicators): {stats['ai_validated']}")
        print(f"  - AI rejected: {stats['ai_rejected']}")
        print(f"  - No indicators: {stats['no_indicators']}")
        print(f"{'='*70}\n")

        logging.info(f"Validation complete. Valid: {valid_count}, Invalid: {invalid_count}")
