"""
Configuration manager for processing logic.
Handles force/conditional processing decisions.
"""

from typing import Dict, List, Tuple, Any, Optional
import logging


class ProcessingConfigManager:
    """Manages processing configuration for different metadata fields"""
    
    def __init__(self, processing_mode: Dict[str, Any]):
        self.processing_mode = processing_mode
        self.force_processing_fields = processing_mode.get('force_processing', ['ai:affiliation', 'ai:dewey'])
        self.conditional_processing_fields = processing_mode.get('conditional_processing', [
            'ai:author', 'ai:keywords_gen', 'ai:title', 'ai:type', 'ai:keywords_ext', 'ai:keywords_dnb'
        ])
        self.allow_skip_when_all_conditional_filled = processing_mode.get('allow_skip_when_all_conditional_filled', False)
    
    def should_process_field(self, field_name: str, existing_metadata: Optional[Any]) -> Tuple[bool, str]:
        """Determine if a field should be processed based on configuration"""
        # Force processing fields are always processed
        if field_name in self.force_processing_fields:
            return True, "force"
        
        # Conditional processing fields are only processed if empty/missing
        if field_name in self.conditional_processing_fields:
            field_needs_processing = (
                existing_metadata is None or 
                field_name not in existing_metadata or 
                self._is_empty(existing_metadata.get(field_name))
            )
            return field_needs_processing, "conditional"
        
        # Unknown fields default to conditional processing
        field_needs_processing = (
            existing_metadata is None or 
            field_name not in existing_metadata or 
            self._is_empty(existing_metadata.get(field_name))
        )
        return field_needs_processing, "conditional"
    
    def should_skip_file(self, existing_metadata: Optional[Any]) -> bool:
        """Determine if entire file should be skipped based on configuration"""
        # If force processing is enabled for any field, never skip
        if self.force_processing_fields:
            return False
        
        # If skipping is disabled, never skip
        if not self.allow_skip_when_all_conditional_filled:
            return False
        
        # Check if all conditional fields are filled
        for field_name in self.conditional_processing_fields:
            should_process, _ = self.should_process_field(field_name, existing_metadata)
            if should_process:
                return False  # At least one field needs processing
        
        return True  # All conditional fields filled and no force processing
    
    def _is_empty(self, value: Any) -> bool:
        """Check if a value is considered empty"""
        if value is None or value == "":
            return True
        
        try:
            import pandas as pd
            if pd.isna(value):
                return True
        except (TypeError, ValueError, ImportError):
            if isinstance(value, list) and len(value) == 0:
                return True
        
        return False
    
    def get_processing_summary(self) -> Dict[str, Any]:
        """Get summary of current processing configuration"""
        return {
            'force_processing_fields': self.force_processing_fields,
            'conditional_processing_fields': self.conditional_processing_fields,
            'allow_skip_when_all_conditional_filled': self.allow_skip_when_all_conditional_filled,
            'total_fields': len(self.force_processing_fields) + len(self.conditional_processing_fields)
        }
