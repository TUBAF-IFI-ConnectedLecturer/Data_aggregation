"""
Response filtering utilities for AI metadata extraction.
Handles cleaning and validation of LLM responses.
"""

import re
import json
import logging
from typing import Optional, Any


class ResponseFilter:
    """Handles filtering and cleaning of LLM responses"""
    
    def __init__(self):
        self.blacklist = [
            "don't know", "weiß nicht", "weiß es nicht", "Ich kenne ",
            "Ich sehe kein", "Es gibt kein", "Ich kann ", "Ich sehe", "Es wird keine ",
            "Entschuldigung", "Leider kann ich", "Keine Antwort", "Die Antwort kann ich",
            "Der Autor", "die Frage", "Ich habe keine", "Ich habe ", "Ich brauche",
            "Bitte geben", "Das Dokument ", "Es tut mir leid", "Es handelt sich", 
            "Es ist nicht", "Es konnte kein", "I'm ready to help", "Please provide", 
            "I'll respond with"
        ]
    
    def filter_response(self, ai_response: str) -> str:
        """Clean and filter AI response"""
        if not ai_response:
            return ""
        
        # Check for thinking tags and remove explanations
        if "<think>" in ai_response:
            ai_response = re.sub(r'<think>.*?</think>', '', ai_response, flags=re.DOTALL)
        
        # For JSON responses, be more lenient with filtering
        if ai_response.strip().startswith('[') or ai_response.strip().startswith('{'):
            # This looks like JSON, only remove obvious non-JSON parts
            ai_response = ai_response.replace("\n", "")
            return ai_response.strip()
        
        # Remove newlines for non-JSON responses
        ai_response = ai_response.replace("\n", "")
        
        # Check for blacklist words and remove them (but not for JSON-like responses)
        if any(phrase in ai_response for phrase in self.blacklist):
            return ""
        
        return ai_response.strip()
    
    def clean_and_parse_json_response(self, raw_response: str) -> Optional[Any]:
        """Clean and parse JSON response from LLM"""
        if not raw_response:
            return None
        
        # Remove markdown code block markers
        cleaned = re.sub(r"^```json\s*|\s*```$", "", raw_response.strip(), flags=re.DOTALL)
        cleaned = cleaned.strip()
        
        # Try to find JSON array or object in the response
        json_match = re.search(r'(\[.*?\]|\{.*?\})', cleaned, re.DOTALL)
        if json_match:
            cleaned = json_match.group(1)
        
        try:
            parsed = json.loads(cleaned)
            return parsed
        except json.JSONDecodeError as e:
            logging.debug(f"Error parsing JSON '{cleaned[:100]}...': {e}")
            
            # Try to fix common JSON issues
            try:
                # Replace single quotes with double quotes
                fixed = cleaned.replace("'", '"')
                parsed = json.loads(fixed)
                return parsed
            except json.JSONDecodeError:
                logging.debug("Manual extraction attempt failed")
                return None
    
    def safe_is_empty(self, value: Any) -> bool:
        """Safely check if a value is empty, handling arrays/lists"""
        if value is None or value == "":
            return True
        
        try:
            import pandas as pd
            # For scalar values, use pd.isna()
            if pd.isna(value):
                return True
        except (TypeError, ValueError, ImportError):
            # If pd.isna fails (e.g., for arrays), check if it's an empty list
            if isinstance(value, list) and len(value) == 0:
                return True
        
        return False
