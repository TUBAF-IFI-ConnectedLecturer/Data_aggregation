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

        # Phrases to remove from the beginning of responses
        self.strip_prefixes = [
            "Here is the extracted data:",
            "Here are the extracted",
            "Here is the",
            "Here are the",
            "Based on the",
            "The answer is:",
            "The extracted data is:",
            "Extracted data:",
            "Returned data:",
            "returned data:",
            "Let me know if",
            "Please let me know",
            "I can provide you with",
            "There is no specific question",
            "No answer provided",
            "No specific answer",
            "Answer:",
            "Data:"
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

        # Remove unwanted prefixes (case-insensitive)
        for prefix in self.strip_prefixes:
            if ai_response.lower().startswith(prefix.lower()):
                ai_response = ai_response[len(prefix):].strip()
                # Remove leading colon or dash if present
                ai_response = re.sub(r'^[:\-\s]+', '', ai_response)
                break

        # Remove trailing helpful phrases (case-insensitive)
        for suffix in ["Let me know if you need", "Please let me know", "I can provide", "if you need any"]:
            # Find the suffix and remove everything from there
            pos = ai_response.lower().find(suffix.lower())
            if pos > 0:
                ai_response = ai_response[:pos].strip()
                # Remove trailing punctuation
                ai_response = re.sub(r'[\.!]+$', '', ai_response)
                break

        # Check for blacklist words and remove them (but not for JSON-like responses)
        if any(phrase in ai_response for phrase in self.blacklist):
            return ""

        # Filter out very short or unhelpful responses
        cleaned = ai_response.strip()

        # Remove responses that are just "No", "Yes", "Ja", "Nein", etc.
        if cleaned.lower() in ["no", "yes", "ja", "nein", "none", "n/a", "unknown"]:
            return ""

        # Remove responses that are too short (less than 3 characters) unless they look like codes
        if len(cleaned) < 3 and not cleaned.isdigit():
            return ""

        # Check if response looks like it's answering a yes/no question instead of providing data
        if cleaned.lower().startswith(("no,", "yes,", "ja,", "nein,")):
            # Try to extract the actual data after the yes/no
            parts = cleaned.split(",", 1)
            if len(parts) > 1:
                cleaned = parts[1].strip()
            else:
                return ""

        return cleaned
    
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
