from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from pydantic import BaseModel
from typing import Optional

class Name(BaseModel):
    Vorname: Optional[str] = None
    Familienname: Optional[str] = None
    Titel: Optional[str] = None

black_list = ["Ich kann ", "Name", "name", "Vorname", "vorname", "Prof", "Dr", 
              "Kein Hinweis", "home", "unknown", "None", "nicht ", "keiner", "kein"
              "?", "!", "keine "]

class NameChecker():
    def __init__(self):
        self.llm = OllamaLLM(model="gemma3:27b", temperature=0.0)
    
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

                FORMAT:
                Return ONLY lines with the following pattern, with fields separated by the pipe symbol (|):
                TITLE|FIRST_NAME|LAST_NAME

                EXAMPLES:
                - For "Dr. Max Mustermann": Dr.|Max|Mustermann
                - For "Angela Schmidt": |Angela|Schmidt  
                - For "Prof. Dr. Hans-Peter von der Müller-Schmidt": Prof. Dr.|Hans-Peter|von der Müller-Schmidt

                IMPORTANT:
                - Return ONLY the formatted lines as shown above
                - Include NO explanations, headers, or additional text
                - Each name should be on a separate line
                - If no names are found, return an empty response
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

    def validate_name(self, name):
        """Validate a single name object."""
        if name.Vorname is None or name.Familienname is None:
            return False
        if name.Vorname == "" or name.Familienname == "":
            return False
        if any([x in name.Vorname for x in black_list]) or any([x in name.Familienname for x in black_list]):
            return False
        return True

    def get_validated_names(self, name_string):
        """Get all valid names from the text."""
        names = self.get_all_names(name_string)
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
