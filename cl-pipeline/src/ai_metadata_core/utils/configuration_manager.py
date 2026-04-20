"""
Configuration manager for processing logic.
Handles force/conditional processing decisions and error retry tracking.
"""

from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
import logging


class ProcessingConfigManager:
    """Manages processing configuration for different metadata fields"""

    def __init__(self, processing_mode: Dict[str, Any]):
        self.processing_mode = processing_mode
        # Raw storage mode: extract raw data without validation (validation in separate stage)
        self.force_processing_fields = processing_mode.get('force_processing', ['ai:affiliation_raw', 'ai:dewey'])
        self.conditional_processing_fields = processing_mode.get('conditional_processing', [
            'ai:author_raw', 'ai:keywords_gen', 'ai:title', 'ai:type', 'ai:keywords_ext', 'ai:keywords_dnb', 'ai:summary'
        ])
        self.allow_skip_when_all_conditional_filled = processing_mode.get('allow_skip_when_all_conditional_filled', False)
        self.skip_if_any_field_exists = processing_mode.get('skip_if_any_field_exists', False)
        self.max_error_retries = processing_mode.get('max_error_retries', 0)  # 0 = unlimited

    def should_process_field(self, field_name: str, existing_metadata: Optional[Any]) -> Tuple[bool, str]:
        """Determine if a field should be processed based on configuration"""
        # Force processing fields are always processed
        if field_name in self.force_processing_fields:
            # Even force-processing respects retry limits
            if self.max_error_retries > 0 and existing_metadata is not None:
                if self._field_exceeded_retries(field_name, existing_metadata):
                    return False, "max_retries_exceeded"
            return True, "force"

        # Conditional processing fields are only processed if empty/missing
        if field_name in self.conditional_processing_fields:
            field_needs_processing = (
                existing_metadata is None or
                field_name not in existing_metadata or
                self._is_empty(existing_metadata.get(field_name))
            )
            if field_needs_processing and self.max_error_retries > 0 and existing_metadata is not None:
                if self._field_exceeded_retries(field_name, existing_metadata):
                    return False, "max_retries_exceeded"
            return field_needs_processing, "conditional"

        # Unknown fields default to conditional processing
        field_needs_processing = (
            existing_metadata is None or
            field_name not in existing_metadata or
            self._is_empty(existing_metadata.get(field_name))
        )
        return field_needs_processing, "conditional"

    def _field_exceeded_retries(self, field_name: str, existing_metadata: Any) -> bool:
        """Check if a field has exceeded its maximum error retry count"""
        errors = self._get_errors_dict(existing_metadata)
        if field_name in errors:
            count = errors[field_name].get('count', 0)
            if count >= self.max_error_retries:
                logging.debug(
                    "Skipping field %s: %d errors >= max_error_retries %d",
                    field_name, count, self.max_error_retries
                )
                return True
        return False

    @staticmethod
    def _get_errors_dict(existing_metadata: Any) -> dict:
        """Extract the ai:_errors dict from existing metadata"""
        if existing_metadata is None:
            return {}
        try:
            errors = existing_metadata.get('ai:_errors')
        except (AttributeError, TypeError):
            return {}
        if isinstance(errors, dict):
            return errors
        return {}

    @staticmethod
    def record_field_error(metadata: dict, field_name: str, error_type: str) -> None:
        """Record an error for a specific field in the metadata dict.

        Args:
            metadata: The mutable metadata dict being built for this document
            field_name: The AI field that failed
            error_type: Error category ("timeout", "llm_error")
        """
        if 'ai:_errors' not in metadata or not isinstance(metadata.get('ai:_errors'), dict):
            metadata['ai:_errors'] = {}
        errors = metadata['ai:_errors']
        if field_name not in errors:
            errors[field_name] = {'count': 0, 'last_error': None, 'last_attempt': None}
        errors[field_name]['count'] += 1
        errors[field_name]['last_error'] = error_type
        errors[field_name]['last_attempt'] = datetime.now().isoformat()

    @staticmethod
    def clear_field_error(metadata: dict, field_name: str) -> None:
        """Clear error tracking for a field after successful processing"""
        errors = metadata.get('ai:_errors')
        if isinstance(errors, dict) and field_name in errors:
            del errors[field_name]

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
            # Exception: skip if ALL force fields have exceeded retries
            if self.max_error_retries > 0:
                all_force_exceeded = all(
                    self._field_exceeded_retries(f, existing_metadata)
                    for f in self.force_processing_fields
                )
                all_conditional_done = all(
                    not self._is_empty(existing_metadata.get(f))
                    or self._field_exceeded_retries(f, existing_metadata)
                    for f in self.conditional_processing_fields
                    if f in (existing_metadata if existing_metadata is not None else {})
                    or True  # check all fields
                )
                if all_force_exceeded and all_conditional_done:
                    return True
            return False  # Never skip when force_processing is configured

        # Only check conditional fields if no force_processing is configured
        # Check if ANY conditional field needs processing
        for field_name in self.conditional_processing_fields:
            should_process, reason = self.should_process_field(field_name, existing_metadata)
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
            'max_error_retries': self.max_error_retries,
            'total_fields': len(self.force_processing_fields) + len(self.conditional_processing_fields)
        }
