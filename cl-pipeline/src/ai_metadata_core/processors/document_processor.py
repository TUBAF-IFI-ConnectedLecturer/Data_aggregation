"""
Document metadata processor for basic document information.
Handles author, title, and document type extraction.
"""

import logging
from typing import Dict, Any, Optional
from ..utils.response_filtering import ResponseFilter
from ..utils.prompt_manager import PromptManager
from ..utils.llm_interface import LLMInterface


class DocumentProcessor:
    """Processes basic document metadata (author, title, type)"""
    
    def __init__(self, prompt_manager: PromptManager, llm_interface: LLMInterface):
        self.prompt_manager = prompt_manager
        self.llm_interface = llm_interface
        self.response_filter = ResponseFilter()
        self.name_checker = self._initialize_name_checker()
    
    def _initialize_name_checker(self):
        """Initialize name checker if available"""
        try:
            import sys
            sys.path.append('../src/general/')
            from checkAuthorNames import NameChecker
            return NameChecker()
        except ImportError:
            logging.warning("NameChecker not available, author revision will be skipped")
            return None
    
    def process_author(self, file: str, chain: Any) -> Dict[str, str]:
        """Extract and process author information"""
        author_query = self.prompt_manager.get_document_prompt("author", file)
        authors = self.llm_interface.get_monitored_response(author_query, chain)
        
        result = {'ai:author': authors}
        
        # Add revised author if name checker is available
        if self.name_checker and authors:
            result['ai:revisedAuthor'] = self.name_checker.get_all_names(authors)
        
        return result
    
    def process_title(self, file: str, chain: Any) -> Dict[str, str]:
        """Extract document title"""
        title_query = self.prompt_manager.get_document_prompt("title", file)
        title = self.llm_interface.get_monitored_response(title_query, chain)
        
        return {'ai:title': self.response_filter.filter_response(title)}
    
    def process_document_type(self, file: str, chain: Any) -> Dict[str, str]:
        """Extract document type"""
        type_query = self.prompt_manager.get_document_prompt("type", file)
        doc_type = self.llm_interface.get_monitored_response(type_query, chain)
        
        return {'ai:type': self.response_filter.filter_response(doc_type)}
    
    def process_all_document_metadata(self, file: str, chain: Any) -> Dict[str, str]:
        """Process all basic document metadata"""
        metadata = {}
        
        try:
            # Process author
            author_data = self.process_author(file, chain)
            metadata.update(author_data)
            
            # Process title
            title_data = self.process_title(file, chain)
            metadata.update(title_data)
            
            # Process document type
            type_data = self.process_document_type(file, chain)
            metadata.update(type_data)
            
        except Exception as e:
            logging.error("Error processing document metadata for %s: %s", file, e)
        
        return metadata
