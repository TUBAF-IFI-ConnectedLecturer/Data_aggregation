"""
LLM interface utilities for AI metadata extraction.
Handles LLM communication and timeouts.
"""

import logging
from typing import Any

try:
    from wrapt_timeout_decorator import timeout
except ImportError:
    # Fallback wenn wrapt_timeout_decorator nicht verfügbar ist
    def timeout(seconds):
        def decorator(func):
            return func
        return decorator


class LLMInterface:
    """Handles LLM communication with timeout management"""
    
    def __init__(self, timeout_seconds: int = 240):
        self.timeout_seconds = timeout_seconds
    
    @timeout(240)
    def _get_response_with_timeout(self, query: str, chain: Any) -> str:
        """Get response from chain with timeout protection"""
        try:
            response = chain.invoke({'query': query})
            return response['result']
        except Exception as e:
            logging.error("Error processing query %s: %s", query, str(e))
            return ""
    
    def get_monitored_response(self, query: str, chain: Any) -> str:
        """Get response with timeout and error monitoring"""
        try:
            return self._get_response_with_timeout(query, chain)
        except Exception as e:
            logging.error("Timeout of %ds for query %s: %s", self.timeout_seconds, query, str(e))
            return ""
    
    def create_qa_chain(self, retriever: Any, llm: Any, prompt: Any) -> Any:
        """Create RetrievalQA chain"""
        from langchain_classic.chains import RetrievalQA
        
        return RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever, 
            chain_type="stuff",
            return_source_documents=True, 
            chain_type_kwargs={'prompt': prompt} 
        )
