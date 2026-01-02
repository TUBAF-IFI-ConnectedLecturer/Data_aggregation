"""
Education processor for identifying educational level and target audience.
Handles the classification of educational materials by grade level and audience description.
"""

from typing import Dict, Any
from ..utils.response_filtering import ResponseFilter
from ..utils.prompt_manager import PromptManager
from ..utils.llm_interface import LLMInterface


class EducationProcessor:
    """Processes educational level and target audience identification for educational materials"""

    # Valid educational levels (must match prompt configuration)
    VALID_EDUCATION_LEVELS = [
        "Frühkindliche Bildung",
        "Primarstufe",
        "Sekundarstufe I",
        "Sekundarstufe II",
        "Berufsbildung",
        "Hochschulbildung",
        "Wissenschaftliche Forschung",
        "Weiterbildung",
        "Nicht spezifisch"
    ]

    def __init__(self, prompt_manager: PromptManager, llm_interface: LLMInterface):
        self.prompt_manager = prompt_manager
        self.llm_interface = llm_interface
        self.response_filter = ResponseFilter()

    def process_education_level(self, file: str, chain: Any) -> Dict[str, str]:
        """
        Identify the primary educational level for the document.

        Args:
            file: Filename being processed
            chain: LangChain retrieval chain

        Returns:
            Dictionary with ai:education_level field
        """
        level_query = self.prompt_manager.get_education_prompt("level", file)
        level = self.llm_interface.get_monitored_response(level_query, chain)

        # Filter and clean the response
        filtered_level = self.response_filter.filter_response(level)

        # Validate that the response is one of the valid categories
        validated_level = self._validate_education_level(filtered_level)

        return {'ai:education_level': validated_level}

    def process_target_audience(self, file: str, chain: Any) -> Dict[str, str]:
        """
        Generate a detailed description of the target audience.

        Args:
            file: Filename being processed
            chain: LangChain retrieval chain

        Returns:
            Dictionary with ai:target_audience field
        """
        audience_query = self.prompt_manager.get_education_prompt("audience_detail", file)
        audience = self.llm_interface.get_monitored_response(audience_query, chain)

        # Filter and clean the response
        filtered_audience = self.response_filter.filter_response(audience)

        # Ensure the description is concise (1-2 sentences)
        validated_audience = self._validate_audience_description(filtered_audience)

        return {'ai:target_audience': validated_audience}

    def _validate_education_level(self, level: str) -> str:
        """
        Validate that the education level is one of the predefined categories.

        Args:
            level: The education level returned by the LLM

        Returns:
            Validated education level or "Nicht spezifisch" if invalid
        """
        if not level:
            return "Nicht spezifisch"

        # Check if the level matches one of the valid categories
        # Allow for minor variations in whitespace
        level_stripped = level.strip()

        for valid_level in self.VALID_EDUCATION_LEVELS:
            if valid_level.lower() == level_stripped.lower():
                return valid_level

        # If no exact match, check if it's contained in the response
        for valid_level in self.VALID_EDUCATION_LEVELS:
            if valid_level.lower() in level_stripped.lower():
                return valid_level

        # Default to "Nicht spezifisch" if no match found
        return "Nicht spezifisch"

    def _validate_audience_description(self, description: str) -> str:
        """
        Validate and potentially truncate the audience description.

        Args:
            description: The target audience description

        Returns:
            Validated and potentially truncated description
        """
        if not description:
            return ""

        # If description is too long, truncate to first 2 sentences
        sentences = description.split('. ')
        if len(sentences) > 2:
            description = '. '.join(sentences[:2])
            if not description.endswith('.'):
                description += '.'

        # Ensure reasonable length (max 300 characters)
        if len(description) > 300:
            description = description[:297] + '...'

        return description
