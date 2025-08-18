import pandas as pd
from pathlib import Path
from tqdm import tqdm
import requests
import logging
import json
import re
import sys

# Logger konfigurieren
logger = logging.getLogger(__name__)

# Ollama Integration für KI-basierte Schlagwortauswahl
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate

from pipeline.taskfactory import TaskWithInputFileMonitor

# Import zentrale Logging-Konfiguration
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))
from pipeline_logging import setup_stage_logging

def get_normalized_keyword(word):
    """
    Einfache Funktion, die ein Schlagwort normalisiert (für die Deduplizierung)
    
    Parameters:
    -----------
    word: str or any
        Das zu normalisierende Wort, kann auch numerisch sein
        
    Returns:
    --------
    str: Normalisierte Form des Keywords
    """
    # Prüfen, ob es sich um ein str-Objekt handelt
    if not isinstance(word, str):
        # Konvertiere in String
        word = str(word)
    
    # Grundlegende Normalisierung: lowercase und Leerzeichen trimmen
    return word.lower().strip()


def get_wikidata_entities(keyword, language="de", limit=10):
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
        response = requests.get(wikidata_api_url, params=params, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Fehler bei der Wikidata API Anfrage für '{keyword}': {str(e)}")
        return None


def get_entity_details(entity_id, language="de"):
    """
    Ruft detaillierte Informationen zu einer Wikidata-Entität ab.
    
    Parameters:
    -----------
    entity_id: str
        Die Wikidata-Entity-ID (z.B. "Q42")
    language: str, optional
        Die Sprache für die Ergebnisse (Standard: "de")
        
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
        response = requests.get(wikidata_api_url, params=params, timeout=15)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Fehler beim Abrufen von Wikidata-Entity '{entity_id}': {str(e)}")
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


def select_best_entity_ai(keyword, candidates, context=None, llm=None, other_keywords=None):
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
        
        # Prompt-Vorlage
        template = """Du bist ein Experte für Wikimedia-Wissensgraphen und semantische Verknüpfungen.
        
Originales Keyword: \"{original_keyword}\"

Kandidaten aus der Wikidata-Datenbank:
{candidates}

{context_info}

Aufgabe: Wähle den Kandidaten aus, der semantisch am besten zum originalen Keyword passt.
Beachte dabei folgende Kriterien:
1. Semantische Bedeutung und Relevanz zum Thema des Dokuments
2. Präzision und Spezifität des Konzepts
3. Einordnung in den thematischen Kontext der anderen Keywords
4. Bei mehrdeutigen Begriffen wähle die im Kontext wahrscheinlichste Bedeutung

Gib nur die Nummer des besten Kandidaten zurück. Falls kein Kandidat gut passt, antworte mit \"0\".
Antwort: """
        
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


def enrich_keyword_with_wikidata(keyword, context=None, llm=None, other_keywords=None):
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
        
    Returns:
    --------
    dict: Angereicherte Keyword-Informationen oder None bei Fehlern
    """
    # Suche nach Wikidata-Entitäten für das Keyword
    search_result = get_wikidata_entities(keyword)
    if not search_result or "search" not in search_result or len(search_result["search"]) == 0:
        logger.info(f"⚠ Keine Wikidata-Entitäten für '{keyword}' gefunden")
        return None
    
    # Extrahiere die gefundenen Entitäten
    entities = search_result["search"]
    total = len(entities)
    logger.info(f"ℹ Wikidata API: {total} Treffer für '{keyword}' gefunden")
    
    # Zeige die ersten Kandidaten zur besseren Nachvollziehbarkeit
    for i, entity in enumerate(entities[:3]):  # Zeige maximal 3 Beispiele
        label = entity.get("label", "Kein Label")
        description = entity.get("description", "Keine Beschreibung")
        logger.info(f"   Kandidat {i+1}: {label} - {description}")
        
    if len(entities) > 3:
        logger.info(f"   ... und {len(entities)-3} weitere Kandidaten")
    
    # Wähle die beste Entität mit KI aus
    logger.info(f"ℹ Starte KI-Auswahl für '{keyword}' mit {len(entities)} Kandidaten...")
    best_entity = select_best_entity_ai(keyword, entities, context, llm, other_keywords)
    
    if not best_entity:
        logger.info(f"⚠ KI hat keine passende Entität für '{keyword}' gefunden")
        return None
    
    # Mit der ausgewählten Entität fortfahren
    entity_id = best_entity.get("id", "")
    entity_label = best_entity.get("label", "unbekannt")
    entity_description = best_entity.get("description", "")
    
    logger.info(f"✓ KI-Auswahl für '{keyword}': \"{entity_label}\" (ID: {entity_id})")
    
    # Holen detaillierte Informationen zur ausgewählten Entität
    entity_details = get_entity_details(entity_id)
    
    # Extrahiere semantischen Kontext
    semantic_context = extract_semantic_context(entity_details)
    
    # Erstelle ein erweitertes Keyword-Sample
    keyword_sample = {
        'raw_keyword': keyword,
        'normalized_keyword': get_normalized_keyword(keyword),
        'is_wikidata': True,
        'wikidata_id': entity_id,
        'wikidata_label': entity_label,
        'wikidata_description': entity_description,
        'instance_of': semantic_context["instance_of"],
        'subclass_of': semantic_context["subclass_of"],
        'related_entities': semantic_context["related_entities"],
        'wikipedia_url': semantic_context["wikipedia_url"],
        'query_term': keyword  # Speichern, mit welchem Begriff wir gesucht haben
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

    def execute_task(self):
        # Logging wird jetzt zentral konfiguriert - keine lokalen Einstellungen mehr nötig
        
        df_metadata = pd.read_pickle(self.file_name_inputs)
        keyword_columns = [col for col in df_metadata.columns if 'keyword' in col]
        keyword_columns = ['ai:keywords_ext', 'ai:keywords_gen', 'ai:keywords_dnb']

        if Path(self.file_name_output).exists():
            df_checkedKeywords = pd.read_pickle(self.file_name_output)
        else:
            # Erstelle einen leeren DataFrame mit denselben Indizes wie df_metadata
            df_checkedKeywords = pd.DataFrame(index=df_metadata.index)

        if Path(self.keyword_list_path.with_suffix('.p')).exists():
            df_keywords = pd.read_pickle(self.keyword_list_path.with_suffix('.p'))
        else:
            df_keywords = pd.DataFrame()

        # Verarbeite jede Zeile in den Metadaten
        for index, row in tqdm(df_metadata.iterrows(), desc="Enriching keywords with Wikidata"):
            all_keywords_list = []
            
            # Log für Datensatz-Start
            logger.info(f"========== Verarbeite Datensatz: ID={row.get('pipe:ID', 'unbekannt')} ==========")
            
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
                    other_keywords
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
                    df_keywords = pd.DataFrame(keyword_sample, index=[0])
                else:
                    # Prüfen, ob ein identisches Keyword bereits existiert
                    is_duplicate = False
                    existing_idx = None
                    
                    # Erstelle eine Liste der normalisierten Formen
                    existing_normalized_forms = [get_normalized_keyword(kw) for kw in df_keywords['raw_keyword'].values]
                    
                    # Finde potentielle Duplikate basierend auf normalisiertem Keyword
                    potential_duplicates = [i for i, form in enumerate(existing_normalized_forms) if form == normalized_keyword]
                    
                    if potential_duplicates:
                        # Bei Wikidata-Entitäten auch die Wikidata-ID vergleichen
                        if wikidata_id:
                            for idx in potential_duplicates:
                                if 'wikidata_id' in df_keywords.columns and df_keywords.iloc[idx].get('wikidata_id') == wikidata_id:
                                    # Exaktes Duplikat (gleiche normalisierte Form UND gleiche Wikidata-ID)
                                    is_duplicate = True
                                    existing_idx = idx
                                    break
                        else:
                            # Bei Nicht-Wikidata-Keywords reicht die normalisierte Form
                            is_duplicate = True
                            existing_idx = potential_duplicates[0]
                    
                    if not is_duplicate:
                        # Kein Duplikat gefunden - neues einzigartiges Keyword
                        if wikidata_id:
                            wikidata_label = keyword_sample.get('wikidata_label', '')
                            logger.info(f"✚ Neues einzigartiges Keyword: '{keyword}' mit Wikidata-Label '{wikidata_label}' wird zur Gesamtliste hinzugefügt")
                        else:
                            logger.info(f"✚ Neues einzigartiges Keyword: '{keyword}' wird zur Gesamtliste hinzugefügt")
                        df_keywords = pd.concat([df_keywords, pd.DataFrame(keyword_sample, index=[0])], ignore_index=True)
                    else:
                        # Duplikat gefunden - Zähler erhöhen
                        existing_count = df_keywords.at[existing_idx, 'count']
                        df_keywords.at[existing_idx, 'count'] += 1
                        existing_original = df_keywords.iloc[existing_idx]['raw_keyword']
                        if wikidata_id:
                            wikidata_label = df_keywords.iloc[existing_idx].get('wikidata_label', '')
                            logger.info(f"⟳ Duplikat gefunden: '{keyword}' → '{wikidata_label}' entspricht '{existing_original}' (bereits {existing_count}x gezählt)")
                        else:
                            logger.info(f"⟳ Duplikat gefunden: '{keyword}' entspricht '{existing_original}' (bereits {existing_count}x gezählt)")

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
            
            logger.info(f"========== Zusammenfassung für Datensatz ID={row.get('pipe:ID', 'unbekannt')} ==========")
            logger.info(f"Insgesamt {len(keyword_list)} Keywords verarbeitet:")
            logger.info(f"✓ {len(wikidata_keywords)} Keywords erfolgreich mit Wikidata verknüpft")
            logger.info(f"✗ {len(non_wikidata_keywords)} Keywords ohne Wikidata-Verknüpfung")
            if non_wikidata_keywords:
                logger.info(f"Nicht verknüpfte Keywords: {', '.join([k['raw_keyword'] for k in non_wikidata_keywords])}")
            logger.info(f"===============================================================")

            # Speichere die Ergebnisse
            df_checkedKeywords.to_pickle(self.file_name_output)
            df_keywords.to_pickle(self.keyword_list_path.with_suffix('.p'))
            df_keywords.to_csv(self.keyword_list_path.with_suffix('.csv'), sep=';')
            
            # Visualisieren der semantischen Beziehungen als JSON-Datei für spätere Graphvisualisierung
            semantic_relations_path = Path(self.processed_data_folder) / "wikimedia_semantic_relations"
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
            with open(f"{semantic_relations_path}_{row.get('pipe:ID', 'unknown')}.json", 'w', encoding='utf-8') as f:
                json.dump(semantic_data, f, ensure_ascii=False, indent=2)
