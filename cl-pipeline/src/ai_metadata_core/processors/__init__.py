"""Processor modules for AI metadata extraction"""

from .document_processor import DocumentProcessor
from .affiliation_processor import AffiliationProcessor
from .keyword_extractor import KeywordExtractor
from .dewey_classifier import DeweyClassifier

__all__ = ['DocumentProcessor', 'AffiliationProcessor', 'KeywordExtractor', 'DeweyClassifier']
