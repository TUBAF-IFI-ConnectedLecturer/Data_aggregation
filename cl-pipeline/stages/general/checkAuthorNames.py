from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel
from typing import Optional
import re

class Name(BaseModel):
    Vorname: Optional[str] = None
    Familienname: Optional[str] = None
    Titel: Optional[str] = None

black_list = ["Ich kann ", "Name", "name", "Vorname", "vorname", "Prof", "Dr",
              "Kein Hinweis", "home", "unknown", "None", "nicht ", "keiner", "kein",
              "?", "!", "keine ", "anwender", "user", "nan", "null", "undefined"]

# Institutional keywords for identifying organizations vs. persons
institutional_keywords = [
    # Universitäten & Hochschulen
    "universität", "university", "hochschule", "college",
    # TU/FH Prefix (with space to avoid false positives)
    "tu ", "fh ", "rwth", "kit ",
    # Organisationen
    "institut", "institute", "zentrum", "center", "centre",
    "gesellschaft", "verband", "association", "foundation", "stiftung",
    # Abteilungen
    "medienzentrum", "fakultät", "department", "fachbereich", "lehrstuhl",
    "chair", "arbeitsgruppe", "forschungsgruppe",
    # Firmen
    "gmbh", "ag ", "inc", "ltd", "llc", "corporation", "corp",
    # Regierung & Verwaltung
    "ministerium", "ministry", "behörde", "agency", "amt",
    # Bibliotheken & Archive
    "bibliothek", "library", "archiv", "archive",
    # Andere
    "verlag", "publisher", "redaktion",
    # Sächsische Hochschulen (spezifisch für OPAL-Kontext)
    "tu dresden", "htw dresden", "tu bergakademie", "bergakademie freiberg",
    "hochschule mittweida", "westsächsische hochschule", "berufsakademie sachsen",
    "evangelische hochschule dresden", "palucca", "hfbk dresden", "hfm dresden",
    "hhl leipzig", "dipf leipzig", "tubaf", "htwdd", "hsmw",
    # Weitere deutsche Hochschulen
    "tu berlin", "tu münchen", "lmu münchen", "rwth aachen", "uni hamburg",
    "uni köln", "uni frankfurt", "uni heidelberg", "uni freiburg"
]

class NameChecker():
    def __init__(self, model="llama3:70b"):
        """
        Initialize NameChecker with configurable LLM model.

        Args:
            model: Ollama model name (default: llama3:70b for improved accuracy)
        """
        self.llm = OllamaLLM(model=model, temperature=0.0)
    
    def get_all_names(self, name_string):
        """Extract all names with minimal processing."""
        prompt = PromptTemplate(
            template="""
                You are a specialized tool for extracting person names from text. Your task is to carefully analyze the following input and identify ALL person names:

                INPUT TEXT: {name_string}

                INSTRUCTIONS:
                1. Identify ALL person names in the input text, including partial names.
                2. Extract each name with its components (title, first name, last name).
                3. For academic titles (Dr., Prof., etc.), place them in the TITLE field.
                4. Place first/given names in the FIRST_NAME field.
                5. Place last/family names in the LAST_NAME field.
                6. Include compound last names (e.g., "von Goethe", "van der Meer") completely in the LAST_NAME field.
                7. For hyphenated names (e.g., "Hans-Peter", "Müller-Schmidt"), keep them together in their respective fields.
                8. If a component is missing (e.g., no title), leave that field empty.
                9. **CRITICAL FOR SINGLE NAMES**: If only ONE word is given (e.g., "Hillenkötter", "Blumensberger"), treat it as a LAST_NAME and leave FIRST_NAME empty. Single words are typically last names in bibliographic contexts.

                FORMAT:
                Return ONLY lines with the following pattern, with fields separated by the pipe symbol (|):
                TITLE|FIRST_NAME|LAST_NAME

                EXAMPLES:
                - For "Dr. Max Mustermann": Dr.|Max|Mustermann
                - For "Angela Schmidt": |Angela|Schmidt
                - For "Prof. Dr. Hans-Peter von der Müller-Schmidt": Prof. Dr.|Hans-Peter|von der Müller-Schmidt
                - For "Hillenkötter" (single name): ||Hillenkötter
                - For "Blumensberger" (single name): ||Blumensberger
                - For "Faust" (single name): ||Faust

                IMPORTANT:
                - Return ONLY the formatted lines as shown above
                - Include NO explanations, headers, or additional text
                - Each name should be on a separate line
                - If no names are found, return an empty response
                - Single words WITHOUT comma separation are ALWAYS last names
                """,
            input_variables=["name_string"]
        ).format(name_string=name_string)
        
        result = self.llm.invoke(prompt)
        names = []
        
        # Simple line-by-line parsing
        for line in result.strip().split('\n'):
            if '|' in line:
                parts = line.split('|')
                if len(parts) == 3:
                    names.append(Name(
                        Titel=parts[0].strip(),
                        Vorname=parts[1].strip(),
                        Familienname=parts[2].strip()
                    ))
        
        return names

    def is_likely_institution(self, name):
        """
        Check if a name is likely an institution rather than a person.

        Args:
            name: Name object with Vorname, Familienname, Titel

        Returns:
            True if the name appears to be an institution
        """

        # Pattern 1: Kein Vorname + institutionelle Keywords im Familienname
        if not name.Vorname or name.Vorname == "":
            lastname_lower = name.Familienname.lower()

            # Check for institutional keywords
            if any(keyword in lastname_lower for keyword in institutional_keywords):
                return True

            # Pattern 2: Multiple kapitalisierte Wörter ohne Vorname
            # z.B. "Dresden Medienzentrum", "Berlin Institute"
            words = name.Familienname.split()
            if len(words) >= 2:
                # Check if most words start with uppercase (allowing for "und", "of", etc.)
                capitalized_count = sum(1 for w in words if w and w[0].isupper())
                if capitalized_count >= len(words) * 0.6:  # 60% threshold
                    return True

        # Pattern 3: Abteilungscodes (z.B. "FIN A 2.3", "CS 101")
        if re.match(r'^[A-Z]{2,}\s+[A-Z]\s+\d', name.Familienname):
            return True

        # Pattern 4: Nur Großbuchstaben (Akronyme wie "TUD", "HTWK", "MIT")
        if name.Familienname.isupper() and len(name.Familienname) > 2:
            return True

        # Pattern 5: Username-ähnliche Muster
        # z.B. "uhc;pschoeps", "user@domain", "admin_user"
        if any(char in name.Familienname for char in [';', '@', '_']):
            return True

        # Pattern 6: Einzelne Wörter die typisch für Organisationen sind
        # z.B. wenn jemand nur "Medienzentrum" als Familienname hat
        if not name.Vorname or name.Vorname == "":
            single_word_lower = name.Familienname.lower().strip()
            for keyword in institutional_keywords:
                if single_word_lower == keyword.strip():
                    return True

        return False

    def validate_name(self, name):
        """Validate a single name object."""
        # Last name is required
        if name.Familienname is None or name.Familienname == "":
            return False

        # Check if it's an institution
        if self.is_likely_institution(name):
            return False

        # Check blacklist for last name
        if any([x in name.Familienname for x in black_list]):
            return False

        # Check blacklist for first name only if it exists
        if name.Vorname and any([x in name.Vorname for x in black_list]):
            return False

        return True

    def get_validated_names(self, name_string):
        """Get all valid names from the text."""
        names = self.get_all_names(name_string)

        # Additional check: if multiple names without first names are extracted,
        # and at least one is an institution, filter all of them
        # (e.g., "TU Dresden Medienzentrum" → ["Dresden", "Medienzentrum"])
        if len(names) > 1:
            all_without_firstname = all(not n.Vorname or n.Vorname == "" for n in names)
            any_is_institution = any(self.is_likely_institution(n) for n in names)

            if all_without_firstname and any_is_institution:
                # This is likely an institutional name split into parts
                return []

        return [name for name in names if self.validate_name(name)]

    def get_validated_name(self, name_string):
        """Legacy method for backward compatibility - returns first valid name."""
        validated_names = self.get_validated_names(name_string)
        return validated_names[0] if validated_names else None
    

if __name__ == "__main__":
    nc = NameChecker()
    test_string = "Johannes Giese-Hinz, Jens Oman, Margherita Spiluttini"

    test_strings =[
        "Dr. Max Mustermann",
        "Prof. Dr. Angela Schmidt",
        "Hans-Peter von der Mühlen-Schulze und Maria Weber",
        "Ellerbrock Dagmar",
        "Johannes Giese-Hinz, Jens Oman, Margherita Spiluttini"
        "Prof. Dr. Jost A. Studer, Martin G. Koller, Jan Laue",
        "Dagmar Ellerbrock, Prof. Dr. Jost A. Studer, Martin G. Koller, Jan Laue"    
    ]

    name_checker = NameChecker()
    # Teste die Extraktion von Namen
    for test_string in test_strings:
        print(f"Testing: {test_string}")
        names = name_checker.get_all_names(test_string)
        for name in names:
            print(f"Extracted Name: {name.Titel} {name.Vorname} {name.Familienname}")
            if name_checker.validate_name(name):
                print("Valid Name")
            else:
                print("Invalid Name")
        print("-" * 40)
