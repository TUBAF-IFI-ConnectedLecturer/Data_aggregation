"""
LLM interface utilities for AI metadata extraction.
Handles LLM communication and timeouts.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Try to import timeout decorator, but provide explicit fallback
_timeout_available = True
try:
    from wrapt_timeout_decorator import timeout
except ImportError:
    _timeout_available = False
    logger.warning(
        "wrapt_timeout_decorator not installed. Timeout protection disabled. "
        "Install with: pip install wrapt-timeout-decorator"
    )

    def timeout(seconds):
        """No-op timeout decorator - timeouts are disabled"""
        def decorator(func):
            def wrapper(*args, **kwargs):
                logger.warning(
                    f"Timeout decorator not available for {func.__name__}. "
                    f"Request may hang indefinitely. Install wrapt-timeout-decorator."
                )
                return func(*args, **kwargs)
            return wrapper
        return decorator


class LLMInterface:
    """Handles LLM communication with timeout management"""

    def __init__(self, timeout_seconds: int = 240):
        self.timeout_seconds = timeout_seconds
        self.last_error = None

    @timeout(240)
    def _get_response_with_timeout(self, query: str, chain: Any) -> str:
        """Get response from chain with timeout protection"""
        response = chain.invoke({'query': query})
        return response['result']

    def get_monitored_response(self, query: str, chain: Any) -> str:
        """Get response with timeout and error monitoring.

        Sets self.last_error to indicate error type:
            - None: Success (or empty response from LLM)
            - "timeout": Request timed out
            - "llm_error": LLM/Ollama returned an error
        """
        try:
            result = self._get_response_with_timeout(query, chain)
            self.last_error = None
            return result
        except TimeoutError as e:
            logging.error("Timeout of %ds for query %s: %s", self.timeout_seconds, query, str(e))
            self.last_error = "timeout"
            return ""
        except Exception as e:
            error_str = str(e).lower()
            if "timeout" in error_str or "timed out" in error_str:
                logging.error("Timeout of %ds for query %s: %s", self.timeout_seconds, query, str(e))
                self.last_error = "timeout"
                return ""
            logging.error("LLM error for query %s: %s", query, str(e))
            self.last_error = "llm_error"
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
