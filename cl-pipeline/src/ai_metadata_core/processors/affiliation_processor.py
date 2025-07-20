"""
Affiliation processor for institutional affiliation detection.
Implements multi-step affiliation detection with validation and normalization.
"""

import re
import logging
from typing import Dict, List, Any
from ..utils.response_filtering import ResponseFilter
from ..utils.prompt_manager import PromptManager
from ..utils.llm_interface import LLMInterface


class AffiliationProcessor:
    """Specialized processor for institutional affiliation detection"""
    
    def __init__(self, prompt_manager: PromptManager, llm_interface: LLMInterface):
        self.prompt_manager = prompt_manager
        self.llm_interface = llm_interface
        self.response_filter = ResponseFilter()
        self.university_mappings = self._create_university_mappings()
    
    def _create_university_mappings(self) -> Dict[str, str]:
        """Create mapping of university abbreviations to full names"""
        return {
            # Technical Universities
            'tu dresden': 'Technische Universität Dresden',
            'tu berlin': 'Technische Universität Berlin', 
            'tu münchen': 'Technische Universität München',
            'tu darmstadt': 'Technische Universität Darmstadt',
            'tu braunschweig': 'Technische Universität Braunschweig',
            'tu clausthal': 'Technische Universität Clausthal',
            'tu dortmund': 'Technische Universität Dortmund',
            'tum': 'Technische Universität München',
            'tud': 'Technische Universität Dresden',
            'tub': 'Technische Universität Berlin',
            
            # Universities of Applied Sciences
            'fh aachen': 'FH Aachen',
            'fh köln': 'FH Köln',
            'fh münchen': 'Hochschule München',
            'hs münchen': 'Hochschule München',
            
            # Sächsische Hochschulen (besonders relevant für diese Daten)
            'tu bergakademie freiberg': 'TU Bergakademie Freiberg',
            'tubaf': 'TU Bergakademie Freiberg',
            'tu freiberg': 'TU Bergakademie Freiberg',
            'bergakademie freiberg': 'TU Bergakademie Freiberg',
            'technische universität bergakademie freiberg': 'TU Bergakademie Freiberg',
            'htw dresden': 'Hochschule für Technik und Wirtschaft Dresden',
            'htwdd': 'Hochschule für Technik und Wirtschaft Dresden',
            'hochschule für technik und wirtschaft dresden': 'Hochschule für Technik und Wirtschaft Dresden',
            'westsächsische hochschule': 'Westsächsische Hochschule Zwickau',
            'whs': 'Westsächsische Hochschule Zwickau',
            'hs zwickau': 'Westsächsische Hochschule Zwickau',
            'hochschule zwickau': 'Westsächsische Hochschule Zwickau',
            'hochschule mittweida': 'Hochschule Mittweida',
            'hsmw': 'Hochschule Mittweida',
            'hs mittweida': 'Hochschule Mittweida',
            'hochschule für angewandte wissenschaften mittweida': 'Hochschule Mittweida',
            'berufsakademie sachsen': 'Berufsakademie Sachsen',
            'ba sachsen': 'Berufsakademie Sachsen',
            'ba dresden': 'Berufsakademie Sachsen - Staatliche Studienakademie Dresden',
            'ba leipzig': 'Berufsakademie Sachsen - Staatliche Studienakademie Leipzig',
            'ba glauchau': 'Berufsakademie Sachsen - Staatliche Studienakademie Glauchau',
            'ba riesa': 'Berufsakademie Sachsen - Staatliche Studienakademie Riesa',
            'ba bautzen': 'Berufsakademie Sachsen - Staatliche Studienakademie Bautzen',
            'ba plauen': 'Berufsakademie Sachsen - Staatliche Studienakademie Plauen',
            'ba breitenbrunn': 'Berufsakademie Sachsen - Staatliche Studienakademie Breitenbrunn',
            'evangelische hochschule dresden': 'Evangelische Hochschule Dresden',
            'ehs dresden': 'Evangelische Hochschule Dresden',
            'palucca hochschule': 'Palucca Hochschule für Tanz Dresden',
            'hochschule für bildende künste dresden': 'Hochschule für Bildende Künste Dresden',
            'hfbk dresden': 'Hochschule für Bildende Künste Dresden',
            'hochschule für musik': 'Hochschule für Musik Carl Maria von Weber Dresden',
            'hfm dresden': 'Hochschule für Musik Carl Maria von Weber Dresden',
            'hhl leipzig': 'HHL Leipzig Graduate School of Management',
            'handelshochschule leipzig': 'HHL Leipzig Graduate School of Management',
            'dipf leipzig': 'DIPF | Leibniz-Institut für Bildungsforschung und Bildungsinformation',
            'leibniz-institut für bildungsforschung': 'DIPF | Leibniz-Institut für Bildungsforschung und Bildungsinformation',
            
            # Special cases
            'rwth aachen': 'RWTH Aachen',
            'kit karlsruhe': 'Karlsruher Institut für Technologie',
            'lmu münchen': 'Ludwig-Maximilians-Universität München',
            'uni hamburg': 'Universität Hamburg',
            'uni köln': 'Universität zu Köln',
            'uni frankfurt': 'Goethe-Universität Frankfurt am Main',
            'uni heidelberg': 'Ruprecht-Karls-Universität Heidelberg',
            'uni freiburg': 'Albert-Ludwigs-Universität Freiburg',
            'uni würzburg': 'Julius-Maximilians-Universität Würzburg',
            'uni göttingen': 'Georg-August-Universität Göttingen',
            'uni bonn': 'Rheinische Friedrich-Wilhelms-Universität Bonn',
            'uni mainz': 'Johannes Gutenberg-Universität Mainz',
            'uni stuttgart': 'Universität Stuttgart',
            'uni karlsruhe': 'Karlsruher Institut für Technologie',
            
            # International
            'mit': 'Massachusetts Institute of Technology',
            'stanford': 'Stanford University',
            'harvard': 'Harvard University',
            'oxford': 'University of Oxford',
            'cambridge': 'University of Cambridge'
        }
    
    def normalize_affiliation(self, affiliation: str) -> str:
        """Normalize and clean up affiliation names"""
        if not affiliation:
            return ""
        
        # Clean the input
        cleaned = affiliation.strip()
        
        # Remove common prefixes/suffixes that don't add value
        remove_patterns = [
            r'^an der\s+', r'^von der\s+', r'^der\s+', r'^die\s+', r'^das\s+',
            r'\s*-\s*universität$', r'\s*university$', r'\s*institut$', r'\s*institute$'
        ]
        
        for pattern in remove_patterns:
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE).strip()
        
        # Check for direct mappings
        cleaned_lower = cleaned.lower()
        for abbrev, full_name in self.university_mappings.items():
            if abbrev in cleaned_lower:
                return full_name
        
        # If no mapping found, clean up common patterns
        words = cleaned.split()
        capitalized_words = []
        
        for word in words:
            # Keep certain abbreviations in uppercase
            if word.upper() in ['TU', 'FH', 'TUM', 'TUD', 'TUB', 'RWTH', 'KIT', 'LMU', 'MIT']:
                capitalized_words.append(word.upper())
            elif word.lower() in ['universität', 'university', 'hochschule', 'institut', 'institute', 'college']:
                capitalized_words.append(word.capitalize())
            elif len(word) > 1:
                capitalized_words.append(word.capitalize())
            else:
                capitalized_words.append(word)
        
        result = ' '.join(capitalized_words)
        return re.sub(r'\s+', ' ', result).strip()
    
    def validate_affiliation_in_content(self, affiliation: str, collection: Any, file: str) -> bool:
        """Validate that affiliation appears in document content"""
        try:
            search_results = collection.get(
                where={"filename": {"$eq": file}},
                include=["documents"]
            )
            content = " ".join(search_results['documents']).replace("\n", " ").lower()
            
            affiliation_lower = affiliation.lower()
            affiliation_words = affiliation_lower.split()
            
            if len(affiliation_words) >= 2:
                # Check if at least 2 significant words appear in content
                significant_words = [w for w in affiliation_words if len(w) > 3]
                matches = sum(1 for word in significant_words if word in content)
                return matches >= min(2, len(significant_words))
            else:
                return affiliation_lower in content
        except Exception as e:
            logging.warning("Error validating affiliation in content: %s", e)
            return False
    
    def process_affiliation(self, file: str, chain: Any, collection: Any) -> Dict[str, str]:
        """Multi-step affiliation detection process"""
        
        # Step 1: Basic affiliation search
        basic_query = self.prompt_manager.get_affiliation_prompt("basic", file)
        basic_response = self.llm_interface.get_monitored_response(basic_query, chain)
        
        # Step 2: Enhanced affiliation search
        enhanced_query = self.prompt_manager.get_affiliation_prompt("enhanced", file)
        enhanced_response = self.llm_interface.get_monitored_response(enhanced_query, chain)
        
        # Step 3: Email domain analysis
        email_query = self.prompt_manager.get_affiliation_prompt("email_domain", file)
        email_response = self.llm_interface.get_monitored_response(email_query, chain)
        
        # Process all responses
        affiliations = []
        for response in [basic_response, enhanced_response, email_response]:
            filtered_response = self.response_filter.filter_response(response)
            if filtered_response and len(filtered_response.strip()) > 3:
                affiliations.append(filtered_response.strip())
        
        # Step 4: Validation and normalization
        final_affiliation = ""
        if affiliations:
            valid_affiliations = []
            
            # University-related keywords for filtering
            university_keywords = [
                'universität', 'university', 'hochschule', 'institut', 'college',
                'tu ', 'fh ', 'uni ', 'rwth', 'kit', 'lmu', 'tum',
                'fachhochschule', 'technische universität', 'akademie'
            ]
            
            for affiliation in affiliations:
                affiliation_lower = affiliation.lower()
                if any(keyword in affiliation_lower for keyword in university_keywords):
                    # Validate against document content
                    if self.validate_affiliation_in_content(affiliation, collection, file):
                        valid_affiliations.append(affiliation)
            
            # Select the best affiliation (longest/most detailed)
            if valid_affiliations:
                final_affiliation = max(valid_affiliations, key=len)
                final_affiliation = self.normalize_affiliation(final_affiliation)
        
        return {'ai:affiliation': final_affiliation}
