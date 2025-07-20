"""
Dewey Decimal Classification processor.
Implements both direct and two-step thematic classification approaches.
"""

import json
import re
import logging
from pathlib import Path
from typing import Dict, List, Any, Set, Optional
from ..utils.response_filtering import ResponseFilter
from ..utils.prompt_manager import PromptManager
from ..utils.llm_interface import LLMInterface


class DeweyClassifier:
    """Specialized processor for Dewey Decimal Classification"""
    
    def __init__(self, prompt_manager: PromptManager, llm_interface: LLMInterface, 
                 dewey_classification_file: str = "dewey_classification.txt"):
        self.prompt_manager = prompt_manager
        self.llm_interface = llm_interface
        self.response_filter = ResponseFilter()
        self.dewey_classifications = self._load_dewey_classifications(dewey_classification_file)
        self.theme_keywords = self._create_theme_keywords_mapping()
    
    def _load_dewey_classifications(self, dewey_file: str) -> str:
        """Load Dewey classifications from file"""
        dewey_path = Path(dewey_file)
        
        if not dewey_path.is_absolute():
            dewey_path = Path.cwd() / dewey_path
        
        if not dewey_path.exists():
            logging.warning("Dewey classification file not found at: %s", dewey_path)
            return ""
        
        try:
            with open(dewey_path, 'r', encoding='utf-8') as f:
                return f.read()
        except (IOError, OSError) as e:
            logging.error("Error loading Dewey classifications: %s", e)
            return ""
    
    def _create_theme_keywords_mapping(self) -> Dict[str, List[str]]:
        """Create mapping of themes to Dewey prefixes"""
        return {
            'informatik': ['000:', '004:', '005:', '006:'],
            'computer': ['000:', '004:', '005:', '006:'],
            'mathematik': ['510:', '511:', '512:', '513:', '514:', '515:', '516:', '517:', '518:', '519:'],
            'physik': ['530:', '531:', '532:', '533:', '534:', '535:', '536:', '537:', '538:', '539:'],
            'chemie': ['540:', '541:', '542:', '543:', '546:', '547:', '548:', '549:'],
            'biologie': ['570:', '571:', '572:', '573:', '574:', '575:', '576:', '577:', '578:', '579:'],
            'medizin': ['610:', '611:', '612:', '613:', '614:', '615:', '616:', '617:', '618:'],
            'technik': ['600:', '620:', '621:', '622:', '623:', '624:', '625:', '627:', '628:', '629:'],
            'ingenieur': ['620:', '621:', '622:', '623:', '624:', '625:', '627:', '628:', '629:'],
            'bildung': ['370:', '371:', '372:', '373:', '374:', '375:', '378:', '379:'],
            'pädagogik': ['370:', '371:', '372:', '373:', '374:', '375:', '378:', '379:'],
            'erziehung': ['370:', '371:', '372:', '373:', '374:', '375:', '378:', '379:'],
            'philosophie': ['100:', '101:', '110:', '120:', '130:', '140:', '150:', '160:', '170:', '180:', '190:'],
            'psychologie': ['150:', '151:', '152:', '153:', '154:', '155:', '156:', '158:'],
            'religion': ['200:', '210:', '220:', '230:', '240:', '250:', '260:', '270:', '280:', '290:'],
            'geschichte': ['900:', '930:', '940:', '950:', '960:', '970:', '980:', '990:'],
            'geografie': ['910:', '911:', '912:', '913:', '914:', '915:', '916:', '917:', '918:', '919:'],
            'wirtschaft': ['330:', '331:', '332:', '333:', '334:', '335:', '336:', '337:', '338:', '339:'],
            'recht': ['340:', '341:', '342:', '343:', '344:', '345:', '346:', '347:', '348:', '349:'],
            'politik': ['320:', '321:', '322:', '323:', '324:', '325:', '326:', '327:', '328:'],
            'soziologie': ['301:', '302:', '303:', '304:', '305:', '306:', '307:'],
            'sprache': ['400:', '410:', '420:', '430:', '440:', '450:', '460:', '470:', '480:', '490:'],
            'literatur': ['800:', '810:', '820:', '830:', '840:', '850:', '860:', '870:', '880:', '890:'],
            'kunst': ['700:', '710:', '720:', '730:', '740:', '750:', '760:', '770:', '780:', '790:'],
            'musik': ['780:', '781:', '782:', '783:', '784:', '785:', '786:', '787:', '788:']
        }
    
    def get_valid_dewey_notations(self) -> Set[str]:
        """Extract valid Dewey notations from classifications"""
        if not self.dewey_classifications:
            return set()
        
        valid_notations = set()
        for line in self.dewey_classifications.split('\n'):
            if ':' in line:
                notation = line.split(':')[0].strip()
                if notation and notation.replace('.', '').isdigit():
                    valid_notations.add(notation)
        return valid_notations
    
    def get_dewey_sample(self, thematic_areas: Optional[str] = None) -> str:
        """Get representative sample of Dewey classifications"""
        if not self.dewey_classifications:
            return ""
        
        lines = [line for line in self.dewey_classifications.split('\n') 
                if ':' in line and '[Unbesetzt]' not in line]
        sample_lines = []
        
        # If thematic areas provided, prioritize relevant classifications
        if thematic_areas:
            thematic_lower = thematic_areas.lower()
            for theme, prefixes in self.theme_keywords.items():
                if theme in thematic_lower:
                    for line in lines:
                        if any(line.startswith(prefix) for prefix in prefixes):
                            sample_lines.append(line)
            
            if sample_lines:
                # Add general categories for context
                general_categories = ['000:', '100:', '200:', '300:', '400:', '500:', '600:', '700:', '800:', '900:']
                for line in lines[:50]:
                    if any(line.startswith(cat) for cat in general_categories):
                        if line not in sample_lines:
                            sample_lines.append(line)
                
                return '\n'.join(sorted(sample_lines[:50]))
        
        # Fallback: general coverage
        for i in range(0, len(lines), 15):
            sample_lines.append(lines[i])
        
        # Include important categories
        important_categories = ['000:', '004:', '370:', '500:', '600:', '700:', '800:', '900:']
        for line in lines[:100]:
            if any(line.startswith(cat) for cat in important_categories):
                if line not in sample_lines:
                    sample_lines.append(line)
        
        return '\n'.join(sorted(list(set(sample_lines))))
    
    def has_valid_dewey_classification(self, data: Any, valid_notations: Optional[Set[str]] = None) -> bool:
        """Check if data contains valid Dewey classification"""
        try:
            if isinstance(data, str):
                data = json.loads(data)
            
            if not isinstance(data, list) or len(data) == 0:
                return False
            
            dewey_pattern = re.compile(r'^\d{3}(\.\d+)?$')
            
            for entry in data:
                notation = entry.get("notation", "")
                if isinstance(notation, str) and dewey_pattern.match(notation):
                    if valid_notations is None or notation in valid_notations:
                        return True
            return False
        except (json.JSONDecodeError, TypeError):
            return False
    
    def process_dewey_classification(self, file: str, chain: Any) -> Dict[str, List[Any]]:
        """Process Dewey classification using two-step approach"""
        
        # Step 1: Thematic analysis
        thematic_query = self.prompt_manager.get_classification_prompt("thematic_analysis", file=file)
        thematic_areas = self.llm_interface.get_monitored_response(thematic_query, chain)
        thematic_areas_filtered = self.response_filter.filter_response(thematic_areas)
        
        logging.debug("Thematic areas identified: %s", thematic_areas_filtered)
        
        # Step 2: Map to Dewey classifications
        if thematic_areas_filtered:
            dewey_sample = self.get_dewey_sample(thematic_areas_filtered)
            mapping_query = self.prompt_manager.get_classification_prompt(
                "dewey_mapping", 
                thematic_areas=thematic_areas_filtered,
                dewey_sample=dewey_sample
            )
            dewey_response = self.llm_interface.get_monitored_response(mapping_query, chain)
        else:
            # Fallback to direct classification
            direct_query = self.prompt_manager.get_classification_prompt("dewey_direct", file=file)
            if self.dewey_classifications:
                dewey_sample = self.get_dewey_sample()
                full_query = f"{direct_query}\n\nVerfügbare Klassifikationen:\n{dewey_sample}"
                dewey_response = self.llm_interface.get_monitored_response(full_query, chain)
            else:
                dewey_response = self.llm_interface.get_monitored_response(direct_query, chain)
        
        # Process response
        dewey_answer = self.response_filter.filter_response(dewey_response)
        result = {'ai:dewey': []}  # Always initialize as empty list
        
        if dewey_answer:
            dewey_parsed = self.response_filter.clean_and_parse_json_response(dewey_answer)
            
            if dewey_parsed is not None:
                valid_notations = self.get_valid_dewey_notations()
                
                if self.has_valid_dewey_classification(dewey_parsed, valid_notations):
                    result['ai:dewey'] = dewey_parsed
                    logging.debug("Successfully assigned Dewey classification: %s", dewey_parsed)
                else:
                    logging.debug("No valid Dewey classification found")
            else:
                logging.debug("Failed to parse Dewey JSON response")
        
        return result
