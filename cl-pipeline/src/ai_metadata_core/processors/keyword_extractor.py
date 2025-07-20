"""
Keyword extractor for different types of keyword extraction.
Handles extracted, generated, and controlled vocabulary keywords.
"""

import logging
from typing import Dict, Any
from ..utils.response_filtering import ResponseFilter
from ..utils.prompt_manager import PromptManager
from ..utils.llm_interface import LLMInterface


class KeywordExtractor:
    """Specialized processor for keyword extraction"""
    
    def __init__(self, prompt_manager: PromptManager, llm_interface: LLMInterface):
        self.prompt_manager = prompt_manager
        self.llm_interface = llm_interface
        self.response_filter = ResponseFilter()
    
    def extract_keywords(self, file: str, chain: Any) -> Dict[str, str]:
        """Extract keywords directly from document content"""
        extract_query = self.prompt_manager.get_keywords_prompt("extract", file)
        keywords = self.llm_interface.get_monitored_response(extract_query, chain)
        
        return {'ai:keywords_ext': self.response_filter.filter_response(keywords)}
    
    def generate_keywords(self, file: str, chain: Any) -> Dict[str, str]:
        """Generate keywords describing document content"""
        generate_query = self.prompt_manager.get_keywords_prompt("generate", file)
        keywords = self.llm_interface.get_monitored_response(generate_query, chain)
        
        return {'ai:keywords_gen': self.response_filter.filter_response(keywords)}
    
    def extract_controlled_vocabulary_keywords(self, file: str, chain: Any) -> Dict[str, str]:
        """Extract keywords from controlled vocabularies"""
        cv_query = self.prompt_manager.get_keywords_prompt("controlled_vocabulary", file)
        keywords = self.llm_interface.get_monitored_response(cv_query, chain)
        
        return {'ai:keywords_dnb': self.response_filter.filter_response(keywords)}
    
    def process_all_keywords(self, file: str, chain: Any) -> Dict[str, str]:
        """Process all types of keywords"""
        metadata = {}
        
        try:
            # Extract keywords from content
            extracted = self.extract_keywords(file, chain)
            metadata.update(extracted)
            
            # Generate descriptive keywords
            generated = self.generate_keywords(file, chain)
            metadata.update(generated)
            
            # Get controlled vocabulary keywords
            controlled = self.extract_controlled_vocabulary_keywords(file, chain)
            metadata.update(controlled)
            
        except Exception as e:
            logging.error("Error processing keywords for %s: %s", file, e)
        
        return metadata
