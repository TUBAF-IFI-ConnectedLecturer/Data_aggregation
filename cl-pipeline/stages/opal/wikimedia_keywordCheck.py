import pandas as pd
from pathlib import Path
from tqdm import tqdm
import requests
import logging
import json
import re
import sys
import time

# Für die Lemmafizierung
import spacy
from typing import List, Optional, Tuple, Union

# Logger konfigurieren
logger = logging.getLogger(__name__)

# Ollama Integration für KI-basierte Schlagwortauswahl
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate

from pipeline.taskfactory import TaskWithInputFileMonitor

# Import zentrale Logging-Konfiguration
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))
from pipeline_logging import setup_stage_logging

# Import Prompt-Manager
sys.path.append(str(Path(__file__).parent.parent.parent / 'src'))
from ai_metadata_core.utils.prompt_manager import PromptManager

# Laden des spacy-Modells für Deutsch (wird lazy geladen, wenn benötigt)
_nlp = None

def get_spacy_model():
    """
    Lädt das deutsche spacy-Modell lazy (erst wenn benötigt)
    
    Returns:
    --------
    spacy.lang.de.German: Das geladene spacy-Modell für Deutsch
    """
    global _nlp
    if _nlp is None:
        try:
            _nlp = spacy.load("de_core_news_sm")
            logger.info("Deutsches spaCy-Modell erfolgreich geladen")
        except Exception as e:
            logger.error(f"Fehler beim Laden des spaCy-Modells: {str(e)}")
            logger.warning("Die Lemmafizierung wird nicht verfügbar sein!")
            _nlp = None
    return _nlp

_lemmatization_cache = {}  # Cache für bereits lemmafizierte Wörter

def lemmatize_german_word(word: str) -> str:
    """
    Lemmafiziert ein deutsches Wort (Grundform-Ermittlung) mit Caching
    
    Parameters:
    -----------
    word: str
        Das zu lemmafizierte Wort
        
    Returns:
    --------
    str: Die Grundform des Wortes oder das ursprüngliche Wort, falls keine Lemmafizierung möglich
    """
    global _lemmatization_cache
    
    # Cache-Lookup
    if word in _lemmatization_cache:
        return _lemmatization_cache[word]
        
    # Prüfe, ob das spaCy-Modell geladen werden kann
    nlp = get_spacy_model()
    if nlp is None:
        _lemmatization_cache[word] = word
        return word
        
    # Prüfe, ob wir ein einzelnes Wort haben (keine zusammengesetzten Begriffe)
    if len(word.split()) > 1:
        # Bei mehreren Worten keine Lemmafizierung durchführen
        _lemmatization_cache[word] = word
        return word
        
    try:
        # Verarbeite das Wort mit spaCy
        doc = nlp(word)
        
        # Wenn es nur ein Token gibt, verwenden wir dessen Lemma
        if len(doc) == 1:
            lemma = doc[0].lemma_
            
            # Prüfe, ob das Lemma sinnvoll ist (nicht leer und nicht identisch mit Eingabe)
            if lemma and lemma != word:
                logger.debug(f"Lemmafizierung: '{word}' → '{lemma}'")
                _lemmatization_cache[word] = lemma
                return lemma
    except Exception as e:
        logger.warning(f"Fehler bei der Lemmafizierung von '{word}': {str(e)}")
    
    # Fallback: Ursprüngliches Wort zurückgeben
    _lemmatization_cache[word] = word
    return word

_normalization_cache = {}  # Cache für bereits normalisierte Keywords

def get_normalized_keyword(word):
    """
    Normalisiert ein Schlagwort durch Kleinschreibung, Trimmen und optional Lemmafizierung
    
    Parameters:
    -----------
    word: str or any
        Das zu normalisierende Wort, kann auch numerisch sein
        
    Returns:
    --------
    str: Normalisierte Form des Keywords
    """
    global _normalization_cache
    
    # Bei nicht-String-Werten zunächst in String umwandeln
    if not isinstance(word, str):
        word = str(word)
    
    # Cache-Lookup für die gesamte Normalisierungsfunktion
    if word in _normalization_cache:
        return _normalization_cache[word]
        
    # Grundlegende Normalisierung: lowercase und Leerzeichen trimmen
    normalized = word.lower().strip()
    
    # Lemmafizierung für einzelne Wörter durchführen
    if len(normalized.split()) == 1:
        lemma = lemmatize_german_word(normalized)
        _normalization_cache[word] = lemma
        return lemma
    
    # Bei mehreren Wörtern keine Lemmafizierung
    _normalization_cache[word] = normalized
    return normalized


def get_wikidata_entities(keyword, language="de", limit=10, rate_limit_delay=0.5):
    """
    Sucht nach Wikidata-Entitäten, die dem Keyword entsprechen.
    
    Parameters:
    -----------
    keyword: str
        Das zu suchende Keyword
    language: str, optional
        Die Sprache für die Suche (Standard: "de")
    limit: int, optional
        Maximale Anzahl der zurückzugebenden Ergebnisse
    rate_limit_delay: float, optional
        Verzögerung in Sekunden zwischen API-Anfragen, um Rate Limits zu respektieren
        
    Returns:
    --------
    dict: JSON-Antwort von der Wikidata API oder None bei Fehlern
    """
    wikidata_api_url = "https://www.wikidata.org/w/api.php"
    
    # Parameter für die Wikidata-Suche
    params = {
        "action": "wbsearchentities",
        "format": "json",
        "search": keyword,
        "language": language,
        "uselang": language,
        "type": "item",
        "limit": limit
    }
    
    try:
        # Kurze Verzögerung vor der Anfrage, um Rate Limits zu respektieren
        time.sleep(rate_limit_delay)
        
        response = requests.get(wikidata_api_url, params=params, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Fehler bei der Wikidata API Anfrage für '{keyword}': {str(e)}")
        
        # Bei Rate-Limit-Fehlern längere Pause und erneuter Versuch
        if hasattr(e, 'response') and e.response and e.response.status_code == 429:
            wait_time = 5  # 5 Sekunden bei Rate-Limit-Fehler
            logger.warning(f"Rate Limit erreicht. Warte {wait_time} Sekunden und versuche erneut...")
            time.sleep(wait_time)
            
            try:
                response = requests.get(wikidata_api_url, params=params, timeout=10)
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as retry_e:
                logger.error(f"Erneuter Fehler bei der Wikidata API Anfrage: {str(retry_e)}")
                
        return None


def get_entity_details(entity_id, language="de", rate_limit_delay=0.5):
    """
    Ruft detaillierte Informationen zu einer Wikidata-Entität ab.
    
    Parameters:
    -----------
    entity_id: str
        Die Wikidata-Entity-ID (z.B. "Q42")
    language: str, optional
        Die Sprache für die Ergebnisse (Standard: "de")
    rate_limit_delay: float, optional
        Verzögerung in Sekunden zwischen API-Anfragen, um Rate Limits zu respektieren
        
    Returns:
    --------
    dict: Entity-Details oder None bei Fehlern
    """
    wikidata_api_url = "https://www.wikidata.org/w/api.php"
    
    # Parameter für die Wikidata-Entitätsabfrage
    params = {
        "action": "wbgetentities",
        "format": "json",
        "ids": entity_id,
        "languages": language,
        "props": "labels|descriptions|claims|sitelinks"
    }
    
    try:
        # Kurze Verzögerung vor der Anfrage, um Rate Limits zu respektieren
        time.sleep(rate_limit_delay)
        
        response = requests.get(wikidata_api_url, params=params, timeout=15)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Fehler beim Abrufen von Wikidata-Entity '{entity_id}': {str(e)}")
        
        # Bei Rate-Limit-Fehlern längere Pause und erneuter Versuch
        if hasattr(e, 'response') and e.response and e.response.status_code == 429:
            wait_time = 5  # 5 Sekunden bei Rate-Limit-Fehler
            logger.warning(f"Rate Limit erreicht. Warte {wait_time} Sekunden und versuche erneut...")
            time.sleep(wait_time)
            
            try:
                response = requests.get(wikidata_api_url, params=params, timeout=15)
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as retry_e:
                logger.error(f"Erneuter Fehler bei der Wikidata API Anfrage: {str(retry_e)}")
                
        return None


def extract_semantic_context(entity_details):
    """
    Extrahiert semantischen Kontext aus Wikidata-Entitätsdetails.
    
    Parameters:
    -----------
    entity_details: dict
        Die Wikidata-Entitätsdetails
        
    Returns:
    --------
    dict: Ein Dictionary mit semantischen Kontextinformationen
    """
    context = {
        "id": None,
        "label": None,
        "description": None,
        "instance_of": [],
        "subclass_of": [],
        "related_entities": [],
        "wikipedia_url": None
    }
    
    if not entity_details or "entities" not in entity_details:
        return context
    
    # Nehmen wir die erste Entität (es sollte nur eine sein)
    entity_id = list(entity_details["entities"].keys())[0]
    entity = entity_details["entities"][entity_id]
    
    # ID und Label
    context["id"] = entity_id
    if "labels" in entity and "de" in entity["labels"]:
        context["label"] = entity["labels"]["de"]["value"]
    
    # Beschreibung
    if "descriptions" in entity and "de" in entity["descriptions"]:
        context["description"] = entity["descriptions"]["de"]["value"]
    
    # Instance of (P31) - wovon ist es eine Instanz
    if "claims" in entity and "P31" in entity["claims"]:
        for claim in entity["claims"]["P31"]:
            if "mainsnak" in claim and "datavalue" in claim["mainsnak"]:
                instance_id = claim["mainsnak"]["datavalue"]["value"]["id"]
                context["instance_of"].append(instance_id)
    
    # Subclass of (P279) - Unterklasse von
    if "claims" in entity and "P279" in entity["claims"]:
        for claim in entity["claims"]["P279"]:
            if "mainsnak" in claim and "datavalue" in claim["mainsnak"]:
                subclass_id = claim["mainsnak"]["datavalue"]["value"]["id"]
                context["subclass_of"].append(subclass_id)
    
    # Verwandte Entitäten basierend auf anderen Properties sammeln
    # Wir könnten hier noch weitere spezifische Properties hinzufügen
    important_properties = ["P361", "P527", "P1889"]  # Teil von, hat Teil, unterschiedlich von
    for prop in important_properties:
        if "claims" in entity and prop in entity["claims"]:
            for claim in entity["claims"][prop]:
                if "mainsnak" in claim and "datavalue" in claim["mainsnak"]:
                    related_id = claim["mainsnak"]["datavalue"]["value"]["id"]
                    context["related_entities"].append(related_id)
    
    # Wikipedia-URL
    if "sitelinks" in entity and "dewiki" in entity["sitelinks"]:
        wiki_title = entity["sitelinks"]["dewiki"]["title"]
        context["wikipedia_url"] = f"https://de.wikipedia.org/wiki/{wiki_title.replace(' ', '_')}"
    
    return context


def select_best_entity_ai(keyword, candidates, context=None, llm=None, other_keywords=None, prompt_manager=None):
    """
    Verwendet ein LLM, um die beste Wikidata-Entität für ein Keyword auszuwählen.
    
    Parameters:
    -----------
    keyword: str
        Das ursprüngliche Keyword
    candidates: list
        Liste von Wikidata-Entitäten
    context: str, optional
        Dokumentenkontext (z.B. Titel, Abstract)
    llm: OllamaLLM, optional
        Eine initialisierte LLM-Instanz
    other_keywords: list, optional
        Andere Keywords des Dokuments
    prompt_manager: PromptManager, optional
        Eine initialisierte PromptManager-Instanz
        
    Returns:
    --------
    dict: Die beste passende Entität oder None
    """
    if not candidates or len(candidates) == 0:
        return None
    
    try:
        # Formatiere Kandidaten für das LLM
        candidate_texts = []
        for i, candidate in enumerate(candidates):
            label = candidate.get("label", "Kein Label")
            description = candidate.get("description", "Keine Beschreibung")
            entity_id = candidate.get("id", "")
            
            candidate_text = f"{i+1}. \"{label}\" (ID: {entity_id}) - {description}"
            candidate_texts.append(candidate_text)
        
        candidates_str = "\n".join(candidate_texts)
        
        # Erweiterte Kontextinformationen vorbereiten
        context_info = []
        if context:
            context_info.append(f"Dokumentkontext:\n{context}")
            
        # Andere Keywords des Dokuments für besseren Kontext hinzufügen
        if other_keywords and isinstance(other_keywords, list) and other_keywords:
            other_keywords_str = ", ".join([f'"{kw}"' for kw in other_keywords[:10]])
            context_info.append(f"Andere Keywords des Dokuments: {other_keywords_str}")
            if len(other_keywords) > 10:
                context_info.append(f"... und {len(other_keywords)-10} weitere Keywords")
                
        context_info = "\n".join(context_info)
        
        # Verwende den PromptManager, wenn verfügbar
        if prompt_manager:
            # Hole das Template aus dem PromptManager
            template = prompt_manager.prompts.get("wikimedia", {}).get("entity_selection", "")
            
            if not template:
                logger.error("Kein 'wikimedia.entity_selection' Template im PromptManager gefunden!")
                template = ("")
        
        # Erstelle das Prompt
        prompt = PromptTemplate(
            template=template,
            input_variables=["original_keyword", "candidates", "context_info"]
        )
        
        # Erstelle den finalen Prompt-Text
        final_prompt = prompt.format(
            original_keyword=keyword,
            candidates=candidates_str,
            context_info=context_info
        )
        
        # Wenn kein LLM übergeben wurde, erstelle ein Standard-LLM
        if llm is None:
            llm = OllamaLLM(model="llama3")
        
        # Rufe das LLM auf
        result = llm.invoke(final_prompt).strip()
        
        # Extrahiere die Nummer
        match = re.search(r'\d+', result)
        if match:
            index = int(match.group()) - 1  # Konvertiere zu 0-basiertem Index
            if 0 <= index < len(candidates):
                return candidates[index]
            elif index == -1:  # Falls "0" zurückgegeben wurde
                return None
        
        logger.warning(f"LLM-Antwort konnte nicht interpretiert werden: {result}")
        return None
        
    except Exception as e:
        logger.error(f"Fehler bei der LLM-Auswahl: {str(e)}")
        return None


def enrich_keyword_with_wikidata(keyword, context=None, llm=None, other_keywords=None, prompt_manager=None):
    """
    Reichert ein Keyword mit Informationen aus Wikidata an.
    
    Parameters:
    -----------
    keyword: str
        Das zu untersuchende Keyword
    context: str, optional
        Dokumentenkontext (z.B. Titel, Abstract)
    llm: OllamaLLM, optional
        Eine initialisierte LLM-Instanz
    other_keywords: list, optional
        Andere Keywords des Dokuments
    prompt_manager: PromptManager, optional
        Eine initialisierte PromptManager-Instanz
        
    Returns:
    --------
    dict: Angereicherte Keyword-Informationen oder None bei Fehlern
    """
    # Normalisiere das Keyword und prüfe auf Lemmafizierung
    original_keyword = keyword
    normalized_keyword = get_normalized_keyword(keyword)
    
    # Effizientere Prüfung der Lemmafizierung
    basic_normalized = keyword.lower().strip()
    lemmatized = normalized_keyword != basic_normalized and len(normalized_keyword.split()) == 1
    
    # Die Lemmafizierung wird nur für bessere Suchbarkeit verwendet
    # Wir speichern sie, ohne sie direkt mit Wikidata-Entitäten zu verknüpfen
    search_options = []
    
    # Immer primär mit dem Original-Keyword suchen
    search_options.append((keyword, "Original"))
    
    # Wenn lemmafiziert, als alternative Suchform anbieten
    if lemmatized:
        search_options.append((normalized_keyword, "Lemmatisiert"))
        logger.info(f"ℹ Lemmafizierung angewendet: '{keyword}' → '{normalized_keyword}'")
    
    # Versuche alle Such-Optionen nacheinander
    found_results = False
    used_search_term = None
    
    for search_term, search_type in search_options:
        logger.info(f"Frage Wikidata mit {search_type}-Form ab: '{search_term}'")
        search_result = get_wikidata_entities(search_term, rate_limit_delay=1.0)
        
        if search_result and "search" in search_result and len(search_result["search"]) > 0:
            found_results = True
            used_search_term = search_term
            break
    
    if not found_results:
        logger.info(f"⚠ Keine Wikidata-Entitäten für Keyword '{keyword}' gefunden")
        return None
    
    # Extrahiere die gefundenen Entitäten
    entities = search_result["search"]
    total = len(entities)
    logger.info(f"ℹ Wikidata API: {total} Treffer für '{used_search_term}' gefunden")
    
    # Zeige die ersten Kandidaten zur besseren Nachvollziehbarkeit
    for i, entity in enumerate(entities[:3]):  # Zeige maximal 3 Beispiele
        label = entity.get("label", "Kein Label")
        description = entity.get("description", "Keine Beschreibung")
        logger.info(f"   Kandidat {i+1}: {label} - {description}")
        
    if len(entities) > 3:
        logger.info(f"   ... und {len(entities)-3} weitere Kandidaten")
    
    # Wähle die beste Entität mit KI aus - nutze das Original-Keyword für KI-Kontext
    logger.info(f"ℹ Starte KI-Auswahl für '{keyword}' mit {len(entities)} Kandidaten...")
    best_entity = select_best_entity_ai(keyword, entities, context, llm, other_keywords, prompt_manager)
    
    if not best_entity:
        logger.info(f"⚠ KI hat keine passende Entität für '{keyword}' gefunden")
        return None
    
    # Mit der ausgewählten Entität fortfahren
    entity_id = best_entity.get("id", "")
    entity_label = best_entity.get("label", "unbekannt")
    entity_description = best_entity.get("description", "")
    
    logger.info(f"✓ KI-Auswahl für '{keyword}': \"{entity_label}\" (ID: {entity_id})")
    
    # Holen detaillierte Informationen zur ausgewählten Entität
    entity_details = get_entity_details(entity_id, rate_limit_delay=1.0)
    
    # Extrahiere semantischen Kontext
    semantic_context = extract_semantic_context(entity_details)
    
    # Erstelle ein erweitertes Keyword-Sample
    # Hier trennen wir klar die sprachliche Normalisierung (lemmafizierung)
    # von der kontextabhängigen Zuordnung zu Wikidata
    keyword_sample = {
        'raw_keyword': original_keyword,
        'normalized_keyword': normalized_keyword,
        'lemmatized': lemmatized,
        'lemmatized_form': normalized_keyword if lemmatized else None,
        'is_wikidata': True,
        'wikidata_id': entity_id,
        'wikidata_label': entity_label,
        'wikidata_description': entity_description,
        'instance_of': semantic_context["instance_of"],
        'subclass_of': semantic_context["subclass_of"],
        'related_entities': semantic_context["related_entities"],
        'wikipedia_url': semantic_context["wikipedia_url"],
        'query_term': used_search_term,  # Term, mit dem gesucht wurde
        'document_context_id': context.split('\n')[0] if context and '\n' in context else None  # Speichere Dokument-ID
    }
    
    # Logge erweiterte Informationen
    logger.info(f"   + Wikidata-Beschreibung: {entity_description}")
    if semantic_context["instance_of"]:
        logger.info(f"   + Instanz von: {', '.join(semantic_context['instance_of'])}")
    if semantic_context["wikipedia_url"]:
        logger.info(f"   + Wikipedia-URL: {semantic_context['wikipedia_url']}")
    
    return keyword_sample


class WikidataKeywordCheck(TaskWithInputFileMonitor):
    def __init__(self, config_stage, config_global):
        super().__init__(config_stage, config_global)
        
        # Setup zentrale Logging-Konfiguration
        self.logger_configurator = setup_stage_logging(config_global)
        
        # Initialisiere spaCy für die Lemmafizierung (lazy loading)
        # Dies lädt das Modell noch nicht, sondern bereitet nur vor
        try:
            # Lade das deutsche spaCy-Modell, wenn es vorhanden ist
            # Nur ein Test, ob es funktioniert - wird später lazy geladen
            if get_spacy_model() is not None:
                logger.info("SpaCy für Lemmafizierung ist verfügbar")
        except Exception as e:
            logger.error(f"Fehler bei der Initialisierung von spaCy: {str(e)}")
            logger.warning("Die Lemmafizierung wird nicht verfügbar sein!")
        
        # Initialisiere Ollama LLM für KI-basierte Entitätsauswahl
        try:
            self.llm = OllamaLLM(model="gemma3:27b", temperature=0.0)
            logger.info("Ollama LLM für Entitätsauswahl erfolgreich initialisiert")
        except Exception as e:
            logger.error(f"Fehler bei der Initialisierung des Ollama LLM: {str(e)}")
            logger.warning("Fallback auf einfache Entitätsauswahl (erstes Ergebnis)")
            self.llm = None
        
        stage_param = config_stage['parameters']
        self.file_name_inputs = Path(config_global['raw_data_folder']) / stage_param['file_name_input']
        self.file_name_output = Path(config_global['raw_data_folder']) / stage_param['file_name_output']
        self.file_folder = Path(config_global['file_folder'])
        self.processed_data_folder = config_global['processed_data_folder']
        self.keyword_list_path = Path(config_global['processed_data_folder']) / "wikimedia_keyword_list"
        
        # Konfigurationsoptionen für das LLM
        self.use_llm = stage_param.get('use_llm', True)  # Standard: LLM verwenden wenn verfügbar
        self.llm_model = stage_param.get('llm_model', "gemma3:27b")  # Standard-Modell
        
        # Rate-Limit-Konfiguration für Wikidata API
        self.rate_limit_delay = stage_param.get('rate_limit_delay', 1.0)  # Verzögerung in Sekunden zwischen API-Anfragen
        
        # Initialisiere PromptManager, falls prompts_file_name konfiguriert ist
        self.prompt_manager = None
        if 'prompts_file_name' in stage_param:
            try:
                prompts_path = stage_param['prompts_file_name']
                self.prompt_manager = PromptManager(prompts_path)
                logger.info(f"PromptManager erfolgreich mit {prompts_path} initialisiert")
            except Exception as e:
                logger.error(f"Fehler bei der Initialisierung des PromptManager: {str(e)}")
                logger.warning("Verwende Standard-Templates für Entitätsauswahl")

    def execute_task(self):
        # Logging wird jetzt zentral konfiguriert - keine lokalen Einstellungen mehr nötig
        
        df_metadata = pd.read_pickle(self.file_name_inputs)
        keyword_columns = [col for col in df_metadata.columns if 'keyword' in col]
        keyword_columns = ['ai:keywords_ext', 'ai:keywords_gen', 'ai:keywords_dnb']

        # Lade vorhandene Keyword-Dateien, falls vorhanden
        if Path(self.file_name_output).exists():
            df_checkedKeywords = pd.read_pickle(self.file_name_output)
            logger.info(f"Vorhandene Keyword-Daten geladen: {len(df_checkedKeywords)} Einträge")
        else:
            # Erstelle einen leeren DataFrame mit denselben Indizes wie df_metadata
            df_checkedKeywords = pd.DataFrame(index=df_metadata.index)
            logger.info("Keine vorhandenen Keyword-Daten gefunden, beginne mit leerer Liste")

        if Path(self.keyword_list_path.with_suffix('.p')).exists():
            df_keywords = pd.read_pickle(self.keyword_list_path.with_suffix('.p'))
            logger.info(f"Vorhandene Keyword-Liste geladen: {len(df_keywords)} einzigartige Keywords")
        else:
            df_keywords = pd.DataFrame()
            logger.info("Keine vorhandene Keyword-Liste gefunden, beginne mit leerer Liste")
            
        # Bestimme, welche Indizes bereits verarbeitet wurden und übersprungen werden können
        already_processed_indices = []
        if 'pipe:ID' in df_checkedKeywords.columns and 'keywords' in df_checkedKeywords.columns:
            # Sichere Überprüfung für komplexe Objekte wie Listen/Arrays
            already_processed_indices = []
            for idx, row in df_checkedKeywords.iterrows():
                # Überprüfe jede Zeile einzeln mit try/except, um Fehler zu vermeiden
                try:
                    pipe_id = row.get('pipe:ID')
                    keywords = row.get('keywords')
                    # Prüfen, ob valide Werte vorhanden sind
                    if (pipe_id is not None and keywords is not None and 
                        not pd.isna(pipe_id) and isinstance(keywords, list) and len(keywords) > 0):
                        already_processed_indices.append(idx)
                except Exception as e:
                    logger.warning(f"Fehler beim Prüfen von Index {idx}: {str(e)}")
                    
            if already_processed_indices:
                logger.info(f"Gefunden: {len(already_processed_indices)} bereits verarbeitete Datensätze, die übersprungen werden")
                logger.info(f"Beispiel für verarbeitete Indizes: {already_processed_indices[:5]}" 
                           f"{' ...' if len(already_processed_indices) > 5 else ''}")

        # Speichere den Fortschritt in einer Status-Datei
        progress_file = Path(self.processed_data_folder) / "wikimedia_processing_progress.json"
        
        # Verarbeite jede Zeile in den Metadaten, überspringe bereits verarbeitete
        for index, row in tqdm(df_metadata.iterrows(), desc="Enriching keywords with Wikidata"):
            # Überspringe bereits verarbeitete Datensätze, wenn sie in der df_checkedKeywords sind
            # und einen gültigen keywords-Eintrag haben
            if index in already_processed_indices:
                logger.info(f"Überspringe bereits verarbeiteten Datensatz: ID={row.get('pipe:ID', 'unbekannt')}, Index={index}")
                continue
                
            all_keywords_list = []
            
            # Log für Datensatz-Start
            logger.info(f"========== Verarbeite Datensatz: ID={row.get('pipe:ID', 'unbekannt')}, Index={index} ==========")
            
            # Speichere aktuellen Fortschritt
            try:
                # Sicherstellen, dass index serialisierbar ist
                index_value = int(index) if isinstance(index, (int, float, str)) else str(index)
                
                # Erstelle ein sauberes Dictionary für JSON
                progress_data = {
                    "last_processed_index": index_value,
                    "last_processed_id": str(row.get('pipe:ID', 'unbekannt')),
                    "timestamp": str(pd.Timestamp.now()),
                    "total_records": len(df_metadata),
                    "processed_records": len(already_processed_indices) + 1,  # +1 für den aktuellen
                    "remaining_records": len(df_metadata) - len(already_processed_indices) - 1
                }
                
                with open(progress_file, 'w', encoding='utf-8') as f:
                    json.dump(progress_data, f, ensure_ascii=False, indent=2)
                    
                logger.info(f"Fortschritt gespeichert: {progress_data['processed_records']}/{progress_data['total_records']} "
                           f"({progress_data['processed_records']/progress_data['total_records']*100:.1f}%)")
            except Exception as e:
                logger.error(f"Fehler beim Speichern des Fortschritts: {str(e)}")
                
            # Sammle alle Keywords aus den verschiedenen Spalten
            for col in keyword_columns:
                if col in row and row[col] is not None:
                    if isinstance(row[col], list):
                        logger.info(f"Gefundene Keywords in Spalte '{col}' (als Liste): {', '.join(str(k) for k in row[col])}")
                        all_keywords_list.extend(row[col])
                    elif isinstance(row[col], str) and row[col].strip():
                        # Prüfen, ob der String mehrere durch Komma getrennte Keywords enthält
                        if ',' in row[col]:
                            # Kommagetrennter String - teile auf und bereinige
                            split_keywords = [k.strip() for k in row[col].split(',') if k.strip()]
                            logger.info(f"Gefundene kommagetrennte Keywords in Spalte '{col}': {', '.join(split_keywords)}")
                            all_keywords_list.extend(split_keywords)
                        else:
                            # Einzelnes Keyword
                            logger.info(f"Gefundenes einzelnes Keyword in Spalte '{col}': {row[col].strip()}")
                            all_keywords_list.append(row[col].strip())

            logger.info(f"Insgesamt {len(all_keywords_list)} Keywords für Datensatz gefunden")
            
            keyword_list = []
            
            # Erstelle eine Liste aller anderen Keywords als Kontext für KI-Entscheidungen
            all_other_keywords = all_keywords_list.copy()
            
            for keyword in all_keywords_list:
                logger.info(f"------ Verarbeite Keyword: '{keyword}' ------")
                
                # Kontext für bessere KI-Entscheidungen sammeln
                document_context = f"Document ID: {row['pipe:ID']}"
                
                # Füge weitere Metadaten als Kontext hinzu, falls verfügbar
                context_fields = ['title', 'description', 'abstract', 'subject']
                for field in context_fields:
                    if field in row and isinstance(row[field], str) and row[field]:
                        document_context += f"\n{field.capitalize()}: {row[field]}"
                
                # Alle anderen Keywords des Dokuments als Kontext für bessere KI-Auswahl hinzufügen
                other_keywords = [kw for kw in all_keywords_list if kw != keyword]
                if other_keywords:
                    logger.info(f"Verwende {len(other_keywords)} andere Keywords des Dokuments als Kontext für KI-Auswahl")
                
                # Direkte Abfrage mit dem Original-Keyword
                logger.info(f"Frage Wikidata für Keyword '{keyword}' ab...")
                keyword_sample = enrich_keyword_with_wikidata(
                    keyword, 
                    document_context, 
                    self.llm if self.use_llm else None,
                    other_keywords,
                    self.prompt_manager
                )
                
                # Wenn keine Ergebnisse, speichere trotzdem Basisinformationen
                if not keyword_sample:
                    keyword_sample = {
                        'raw_keyword': keyword,
                        'normalized_keyword': get_normalized_keyword(keyword),
                        'is_wikidata': False
                    }
                    logger.info(f"✗ Keine Wikidata-Entität gefunden für '{keyword}'")
                
                logger.info(f"------ Ende Keyword '{keyword}' ------")
                keyword_sample['count'] = 0
                
                # Normalisierte Form und Wikidata-ID für Deduplizierung nutzen
                normalized_keyword = keyword_sample['normalized_keyword']
                wikidata_id = keyword_sample.get('wikidata_id')
                
                if df_keywords.shape[0] == 0:
                    # Verwende pd.DataFrame.from_dict mit orient='index' und transpose für ein einzelnes Dict
                    df_keywords = pd.DataFrame.from_dict([keyword_sample])
                else:
                    # Prüfen, ob ein identisches Keyword bereits existiert
                    is_duplicate = False
                    existing_idx = None
                    
                    if wikidata_id:
                        # Bei Wikidata-Entitäten speichern wir alle kontextspezifischen Zuordnungen
                        # Wir prüfen nur, ob dieses Keyword mit exakt derselben Wikidata-ID bereits existiert
                        found_exact_duplicate = False
                        
                        if 'wikidata_id' in df_keywords.columns and 'raw_keyword' in df_keywords.columns:
                            # Suche nach identischen (Keyword, Wikidata-ID)-Paaren
                            for idx, (raw_kw, entity_id) in enumerate(zip(df_keywords['raw_keyword'].values, df_keywords['wikidata_id'].values)):
                                if raw_kw == keyword and entity_id == wikidata_id:
                                    # Exakt dasselbe Keyword mit derselben Wikidata-ID
                                    is_duplicate = True
                                    existing_idx = idx
                                    found_exact_duplicate = True
                                    logger.debug(f"Exaktes Duplikat gefunden: '{keyword}' mit Wikidata-ID {wikidata_id}")
                                    break
                            
                        # Wenn kein exaktes Duplikat gefunden wurde, behandeln wir dies als neuen Eintrag
                        # So können wir kontextbezogene Zuordnungen für dasselbe Keyword speichern
                        if not found_exact_duplicate:
                            # Prüfen, ob das normalisierte Keyword bereits mit anderen Wikidata-IDs existiert
                            # (nur für Informationszwecke, wir fügen trotzdem einen neuen Eintrag hinzu)
                            if 'normalized_keyword' in df_keywords.columns and 'wikidata_id' in df_keywords.columns:
                                similar_forms = []
                                for idx, (norm_kw, entity_id) in enumerate(zip(df_keywords['normalized_keyword'].values, df_keywords['wikidata_id'].values)):
                                    if norm_kw == normalized_keyword and entity_id != wikidata_id:
                                        wikidata_label = df_keywords.iloc[idx].get('wikidata_label', 'unbekannt')
                                        similar_forms.append((entity_id, wikidata_label))
                                
                                if similar_forms:
                                    current_label = keyword_sample.get('wikidata_label', 'unbekannt')
                                    logger.info(f"ℹ Hinweis: Das Keyword '{keyword}' (normalisiert: '{normalized_keyword}') hat bereits andere kontextabhängige Bedeutungen:")
                                    for other_id, other_label in similar_forms:
                                        logger.info(f"   → Alternative Bedeutung: '{other_label}' (ID: {other_id})")
                                    logger.info(f"   → Im aktuellen Kontext: '{current_label}' (ID: {wikidata_id})")
                                    logger.info(f"   → Alle Bedeutungsvarianten werden beibehalten, da sie kontextabhängig sind")
                    else:
                        # Bei Nicht-Wikidata-Keywords: Prüfe, ob das exakte Keyword bereits existiert
                        # Wir nutzen die normalisierte Form, um ähnliche Keywords zu identifizieren
                        if 'raw_keyword' in df_keywords.columns:
                            # Prüfe auf exakte Übereinstimmungen
                            exact_matches = [i for i, kw in enumerate(df_keywords['raw_keyword'].values) if kw == keyword]
                            if exact_matches:
                                is_duplicate = True
                                existing_idx = exact_matches[0]
                                logger.debug(f"Exaktes Keyword-Duplikat gefunden: '{keyword}'")
                        
                        # Wenn kein exaktes Match gefunden wurde, suche nach normalisierten Formen
                        if not is_duplicate:
                            # Verwende die bereits gespeicherten normalisierten Formen, wenn vorhanden
                            if 'normalized_keyword' in df_keywords.columns:
                                existing_normalized_forms = df_keywords['normalized_keyword'].values
                            else:
                                # Fallback: Berechne normalisierte Formen
                                existing_normalized_forms = [get_normalized_keyword(kw) for kw in df_keywords['raw_keyword'].values]
                                
                            similar_matches = [i for i, form in enumerate(existing_normalized_forms) if form == normalized_keyword]
                            
                            if similar_matches:
                                # Bei ähnlichen Formen (z.B. durch Lemmafizierung) können wir diese als Duplikate behandeln
                                is_duplicate = True
                                existing_idx = similar_matches[0]
                                matched_keyword = df_keywords.iloc[existing_idx]['raw_keyword']
                                logger.info(f"Ähnliches Keyword gefunden: '{keyword}' ähnelt '{matched_keyword}' (beide normalisiert zu '{normalized_keyword}')")
                                # Wenn durch Lemmafizierung ähnlich, speichern wir die Verbindung
                    
                    if not is_duplicate:
                        # Kein Duplikat gefunden - füge das neue Keyword hinzu
                        if wikidata_id:
                            wikidata_label = keyword_sample.get('wikidata_label', '')
                            # Wir speichern kontextabhängige Zuordnungen separat
                            logger.info(f"✚ Neues Keyword: '{keyword}' mit kontextspezifischer Wikidata-Zuordnung '{wikidata_label}' (ID: {wikidata_id})")
                            
                            logger.info(f"   Die Zuordnung basiert auf dem aktuellen Dokumentkontext")
                            
                            # Wir prüfen trotzdem auf ähnliche lemmafizierte Formen (nur zur Information)
                            if 'normalized_keyword' in df_keywords.columns and 'raw_keyword' in df_keywords.columns:
                                similar_words = []
                                for idx, (norm_kw, raw_kw) in enumerate(zip(df_keywords['normalized_keyword'].values, df_keywords['raw_keyword'].values)):
                                    if norm_kw == normalized_keyword and raw_kw != keyword:
                                        similar_words.append(raw_kw)
                                
                                if similar_words:
                                    logger.info(f"   Hinweis: Ähnliche Wortformen existieren bereits (durch Lemmafizierung): {', '.join(similar_words)}")
                                    logger.info(f"   Diese werden durch Lemmafizierung als verwandt erkannt")
                                    
                                # Prüfe auf andere semantische Bedeutungen (andere Wikidata-IDs)
                                other_meanings = []
                                for idx, (norm_kw, entity_id) in enumerate(zip(df_keywords['normalized_keyword'].values, df_keywords['wikidata_id'].values)):
                                    if norm_kw == normalized_keyword and entity_id != wikidata_id:
                                        other_label = df_keywords.iloc[idx].get('wikidata_label', 'unbekannt')
                                        other_meanings.append((entity_id, other_label))
                                        
                                if other_meanings:
                                    logger.info(f"   Hinweis: Das Keyword hat in anderen Kontexten andere Bedeutungen:")
                                    for other_id, other_label in other_meanings:
                                        logger.info(f"   → Alternative Bedeutung: '{other_label}' (ID: {other_id})")
                        else:
                            logger.info(f"✚ Neues Keyword: '{keyword}' (ohne Wikidata-Verknüpfung) wird zur Liste hinzugefügt")
                            
                        # Füge das neue Keyword/semantische Konzept hinzu
                        df_keywords = pd.concat([df_keywords, pd.DataFrame.from_dict([keyword_sample])], ignore_index=True)
                    else:
                        # Exaktes Duplikat gefunden - Zähler erhöhen
                        existing_count = df_keywords.at[existing_idx, 'count']
                        df_keywords.at[existing_idx, 'count'] += 1
                        existing_original = df_keywords.iloc[existing_idx]['raw_keyword']
                        
                        if wikidata_id:
                            wikidata_label = df_keywords.iloc[existing_idx].get('wikidata_label', '')
                            logger.info(f"⟳ Semantisches Duplikat gefunden: '{keyword}' → '{wikidata_label}' (ID: {wikidata_id}) entspricht '{existing_original}' (bereits {existing_count}x gezählt)")
                        else:
                            logger.info(f"⟳ Textduplikat gefunden: '{keyword}' entspricht '{existing_original}' (bereits {existing_count}x gezählt)")
                            
                        # Speichere das aktuelle Dokument als Referenz, wo dieses Keyword verwendet wird
                        # Dies erweitert die Kontextinformationen für spätere Analysen
                        if 'context_references' not in df_keywords.columns:
                            df_keywords['context_references'] = None
                            for i in range(len(df_keywords)):
                                df_keywords.at[i, 'context_references'] = []
                                
                        current_refs = df_keywords.at[existing_idx, 'context_references']
                        if current_refs is None:
                            current_refs = []
                        
                        # Füge aktuelle Dokument-ID hinzu, wenn noch nicht vorhanden
                        doc_id = row.get('pipe:ID', 'unbekannt')
                        if doc_id not in current_refs:
                            current_refs.append(doc_id)
                            df_keywords.at[existing_idx, 'context_references'] = current_refs

                keyword_list.append(keyword_sample)
                
            # Speichere die Keyword-Liste für diese Zeile
            # Stelle sicher, dass wir die DataFrame-Spalten korrekt anlegen, falls sie noch nicht existieren
            if 'pipe:ID' not in df_checkedKeywords.columns:
                df_checkedKeywords['pipe:ID'] = None
            if 'keywords' not in df_checkedKeywords.columns:
                # Erstellen mit Objekt-Datentyp, um komplexe Listen zu speichern
                df_checkedKeywords['keywords'] = None
                
            # Verwende at für einzelne Zellenzuweisung, da wir einen komplexen Objekt (Liste) zuweisen
            df_checkedKeywords.at[index, 'pipe:ID'] = row['pipe:ID']
            df_checkedKeywords.at[index, 'keywords'] = keyword_list
            
            # Zusammenfassung für diesen Datensatz
            wikidata_keywords = [k for k in keyword_list if k.get('is_wikidata', False)]
            non_wikidata_keywords = [k for k in keyword_list if not k.get('is_wikidata', False)]
            
            # Analyse semantischer Mehrdeutigkeiten
            ambiguous_keywords = []
            for k in wikidata_keywords:
                raw_kw = k['raw_keyword']
                # Verwende die bereits berechnete normalisierte Form, wenn verfügbar
                norm_kw = k.get('normalized_keyword', get_normalized_keyword(raw_kw))
                current_id = k.get('wikidata_id')
                
                # Prüfe, ob dieses normalisierte Keyword mit anderen Wikidata-IDs existiert
                if 'wikidata_id' in df_keywords.columns:
                    matching_rows = df_keywords[
                        df_keywords.apply(
                            lambda row: get_normalized_keyword(row['raw_keyword']) == norm_kw and 
                                        row.get('wikidata_id') != current_id and 
                                        row.get('wikidata_id') is not None,
                            axis=1
                        )
                    ]
                    
                    if not matching_rows.empty:
                        # Fand alternative semantische Bedeutungen für dieses Keyword
                        ambiguous_info = {
                            'raw_keyword': raw_kw,
                            'current_meaning': {
                                'id': current_id,
                                'label': k.get('wikidata_label', 'unbekannt')
                            },
                            'alternative_meanings': []
                        }
                        
                        for _, row in matching_rows.iterrows():
                            ambiguous_info['alternative_meanings'].append({
                                'id': row.get('wikidata_id'),
                                'label': row.get('wikidata_label', 'unbekannt')
                            })
                        
                        ambiguous_keywords.append(ambiguous_info)
            
            logger.info(f"========== Zusammenfassung für Datensatz ID={row.get('pipe:ID', 'unbekannt')} ==========")
            logger.info(f"Insgesamt {len(keyword_list)} Keywords verarbeitet:")
            logger.info(f"✓ {len(wikidata_keywords)} Keywords erfolgreich mit Wikidata verknüpft")
            logger.info(f"✗ {len(non_wikidata_keywords)} Keywords ohne Wikidata-Verknüpfung")
            
            if non_wikidata_keywords:
                logger.info(f"Nicht verknüpfte Keywords: {', '.join([k['raw_keyword'] for k in non_wikidata_keywords])}")
                
            # Ausgabe semantisch mehrdeutiger Keywords
            if ambiguous_keywords:
                logger.info(f"⚠ {len(ambiguous_keywords)} semantisch mehrdeutige Keywords gefunden:")
                for amb in ambiguous_keywords:
                    current = amb['current_meaning']
                    logger.info(f"   • '{amb['raw_keyword']}': In diesem Kontext → '{current['label']}' (ID: {current['id']})")
                    for alt in amb['alternative_meanings']:
                        logger.info(f"     Alternativ auch → '{alt['label']}' (ID: {alt['id']})")
                        
            logger.info(f"===============================================================")

            # Speichere die Ergebnisse
            df_checkedKeywords.to_pickle(self.file_name_output)
            df_keywords.to_pickle(self.keyword_list_path.with_suffix('.p'))
            df_keywords.to_csv(self.keyword_list_path.with_suffix('.csv'), sep=';')
            
            # Visualisieren der semantischen Beziehungen als JSON-Datei für spätere Graphvisualisierung
            # Erstelle Unterordner für Wikimedia-Relationen, falls noch nicht vorhanden
            wikimedia_relations_dir = Path(self.processed_data_folder) / "Wikimedia_relations"
            wikimedia_relations_dir.mkdir(exist_ok=True, parents=True)
            
            semantic_relations_path = wikimedia_relations_dir / "semantic_relation"
            semantic_data = {
                "document_id": row.get('pipe:ID', 'unbekannt'),
                "keywords": [],
                "relations": []
            }
            
            # Füge alle Keywords mit ihren semantischen Verknüpfungen hinzu
            for kw in keyword_list:
                if kw.get('is_wikidata', False):
                    semantic_data["keywords"].append({
                        "id": kw.get('wikidata_id'),
                        "label": kw.get('wikidata_label', kw.get('raw_keyword')),
                        "description": kw.get('wikidata_description', '')
                    })
                    
                    # Füge Instanz-von-Beziehungen hinzu
                    for instance in kw.get('instance_of', []):
                        semantic_data["relations"].append({
                            "source": kw.get('wikidata_id'),
                            "target": instance,
                            "type": "instance_of"
                        })
                    
                    # Füge Unterklasse-von-Beziehungen hinzu
                    for subclass in kw.get('subclass_of', []):
                        semantic_data["relations"].append({
                            "source": kw.get('wikidata_id'),
                            "target": subclass,
                            "type": "subclass_of"
                        })
                    
                    # Füge verwandte Entitäten hinzu
                    for related in kw.get('related_entities', []):
                        semantic_data["relations"].append({
                            "source": kw.get('wikidata_id'),
                            "target": related,
                            "type": "related"
                        })
            
            # Speichere die semantischen Beziehungen
            json_file_path = f"{semantic_relations_path}_{row.get('pipe:ID', 'unknown')}.json"
            with open(json_file_path, 'w', encoding='utf-8') as f:
                json.dump(semantic_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Semantische Relationen gespeichert unter: {json_file_path}")
