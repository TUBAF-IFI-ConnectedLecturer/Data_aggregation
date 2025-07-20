"""
AI Metadata Extraction Core - Modular Implementation

This package provides a clean, modular implementation of AI metadata extraction
with separate processors for different types of metadata.
"""

from .processors.affiliation_processor import AffiliationProcessor
from .processors.dewey_classifier import DeweyClassifier
from .processors.keyword_extractor import KeywordExtractor
from .processors.document_processor import DocumentProcessor
from .utils.prompt_manager import PromptManager
from .utils.response_filtering import ResponseFilter
from .utils.configuration_manager import ProcessingConfigManager
from .utils.llm_interface import LLMInterface

__all__ = [
    'AffiliationProcessor',
    'DeweyClassifier', 
    'KeywordExtractor',
    'DocumentProcessor',
    'PromptManager',
    'ResponseFilter',
    'ProcessingConfigManager',
    'LLMInterface'
]
