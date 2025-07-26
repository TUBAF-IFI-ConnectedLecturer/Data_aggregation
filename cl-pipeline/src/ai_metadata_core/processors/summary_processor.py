"""
Summary processor for generating concise document summaries.
Handles the creation of brief, 3-sentence summaries for educational materials.
"""

from typing import Dict, Any
from ..utils.response_filtering import ResponseFilter
from ..utils.prompt_manager import PromptManager
from ..utils.llm_interface import LLMInterface


class SummaryProcessor:
    """Processes document summarization for educational materials"""
    
    def __init__(self, prompt_manager: PromptManager, llm_interface: LLMInterface):
        self.prompt_manager = prompt_manager
        self.llm_interface = llm_interface
        self.response_filter = ResponseFilter()
    
    def process_summary(self, file: str, chain: Any) -> Dict[str, str]:
        """Generate a concise summary of the document"""
        summary_query = self.prompt_manager.get_summary_prompt("generate", file)
        summary = self.llm_interface.get_monitored_response(summary_query, chain)
        
        # Filter and clean the response
        filtered_summary = self.response_filter.filter_response(summary)
        
        # Ensure the summary is not empty and is appropriately concise
        if filtered_summary and len(filtered_summary.split('.')) > 5:
            # If summary is too long, try to truncate to first 3 sentences
            sentences = filtered_summary.split('. ')
            if len(sentences) > 3:
                filtered_summary = '. '.join(sentences[:3]) + '.'
        
        return {'ai:summary': filtered_summary}
    
    def validate_summary_length(self, summary: str) -> bool:
        """Validate that summary meets length requirements (max 3 sentences)"""
        if not summary:
            return False
        
        # Count sentences by splitting on periods
        sentences = [s.strip() for s in summary.split('.') if s.strip()]
        return len(sentences) <= 3 and len(summary) <= 500  # Max 500 characters as additional check
    
    def process_structured_summary(self, file: str, chain: Any) -> Dict[str, str]:
        """Generate a structured summary with specific focus areas"""
        structured_query = self.prompt_manager.get_summary_prompt("structured", file)
        summary = self.llm_interface.get_monitored_response(structured_query, chain)
        
        return {'ai:summary_structured': self.response_filter.filter_response(summary)}
