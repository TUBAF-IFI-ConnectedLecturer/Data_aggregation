"""
Prompt management utilities for AI metadata extraction.
Handles loading and formatting of YAML prompts.
"""

import yaml
from pathlib import Path
from typing import Dict, Any
import logging


class PromptManager:
    """Manages YAML prompts and template substitutions"""
    
    def __init__(self, prompts_file: str):
        self.prompts_file = prompts_file
        self.prompts = self._load_prompts()
    
    def _load_prompts(self) -> Dict[str, Any]:
        """Load prompts from YAML file"""
        try:
            with open(self.prompts_file, 'r', encoding='utf-8') as f:
                prompts = yaml.safe_load(f)
            return prompts
        except (IOError, yaml.YAMLError) as e:
            logging.error("Failed to load prompts from %s: %s", self.prompts_file, e)
            return {}
    
    def get_system_template(self) -> str:
        """Get the system template for LangChain"""
        return self.prompts.get("system", {}).get("template", "")
    
    def get_document_prompt(self, prompt_type: str, file: str) -> str:
        """Get document-related prompt with file substitution"""
        template = self.prompts.get("document", {}).get(prompt_type, "")
        return template.replace("{file}", file)
    
    def get_affiliation_prompt(self, prompt_type: str, file: str) -> str:
        """Get affiliation-related prompt with file substitution"""
        template = self.prompts.get("affiliation", {}).get(prompt_type, "")
        return template.replace("{file}", file)
    
    def get_keywords_prompt(self, prompt_type: str, file: str) -> str:
        """Get keywords-related prompt with file substitution"""
        template = self.prompts.get("keywords", {}).get(prompt_type, "")
        return template.replace("{file}", file)
    
    def get_summary_prompt(self, prompt_type: str, file: str) -> str:
        """Get summary-related prompt with file substitution"""
        template = self.prompts.get("summary", {}).get(prompt_type, "")
        return template.replace("{file}", file)
    
    def get_classification_prompt(self, prompt_type: str, **kwargs) -> str:
        """Get classification-related prompt with variable substitution"""
        template = self.prompts.get("classification", {}).get(prompt_type, "")
        
        # Replace all provided kwargs
        for key, value in kwargs.items():
            placeholder = "{" + key + "}"
            template = template.replace(placeholder, str(value))
        
        return template
    
    def reload_prompts(self) -> bool:
        """Reload prompts from file"""
        try:
            self.prompts = self._load_prompts()
            return True
        except Exception as e:
            logging.error("Failed to reload prompts: %s", e)
            return False
