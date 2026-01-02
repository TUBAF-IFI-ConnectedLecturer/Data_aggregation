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
            'ai:author', 'ai:keywords_gen', 'ai:title', 'ai:type', 'ai:keywords_ext', 'ai:keywords_dnb', 'ai:summary'
        ])
        self.allow_skip_when_all_conditional_filled = processing_mode.get('allow_skip_when_all_conditional_filled', False)
        self.skip_if_any_field_exists = processing_mode.get('skip_if_any_field_exists', False)
    
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
        # If no existing metadata, never skip (new document)
        if existing_metadata is None:
            return False

        # NEW: Skip if ANY field exists (document has been analyzed before)
        # This mode skips all documents that have been processed, even if incomplete
        if self.skip_if_any_field_exists:
            # Check if at least one AI field exists and is not empty
            for field_name in self.conditional_processing_fields:
                if field_name in existing_metadata and not self._is_empty(existing_metadata[field_name]):
                    return True  # Skip because document has been analyzed (at least one field exists)
            return False  # Don't skip - no AI fields exist yet

        # CRITICAL FIX: If force_processing fields are configured, NEVER skip
        # Force processing fields must ALWAYS be re-processed
        if len(self.force_processing_fields) > 0:
            return False  # Never skip when force_processing is configured

        # Only check conditional fields if no force_processing is configured
        # Check if ANY conditional field needs processing
        for field_name in self.conditional_processing_fields:
            should_process, _ = self.should_process_field(field_name, existing_metadata)
            if should_process:
                return False  # At least one conditional field needs processing

        return True  # All conditional fields are complete and no force_processing configured
    
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
