from ollama import Client
from pydantic import BaseModel

class Name(BaseModel):
    Vorname: str | None
    Familienname: str | None
    Titel: str | None

black_list = ["Ich kann ", "Name", "name", "Vorname", "vorname", "Prof", "Dr", 
              "Kein Hinweis", "home", "unknown", "None", "nicht ", "keiner", "kein"
              "?", "!", "keine "]

class NameChecker():
    def __init__(self):
        self.client = Client()

    def get_name(self, name_string):
        response = self.client.chat(
            messages=[
                {
                    'role': 'user',
                    'content': f"""You are a tool for detecting first and last names as well as academic titles in text.
                    Analyze the following input: {name_string}
                    Extract only the first full name, including any academic titles that may appear.
                    Return only the extracted name — no explanations, no additional names, no formatting or commentary."""
                }
            ],
            model='llama3:latest',
            format = Name.model_json_schema(),
        )
        return Name.model_validate_json(response['message']['content'])

    def get_validated_name(self, name_string):
        name = self.get_name(name_string)
        if name.Vorname is None or name.Familienname is None:
            return None
        if name.Vorname == "" or name.Familienname == "":
            return None
        if any([x in name.Vorname for x in black_list]) or any([x in name.Familienname for x in black_list]):
            return None
        return name