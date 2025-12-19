import pandas as pd
from pathlib import Path
from tqdm import tqdm
import requests
import logging
import json
import spacy
import re
import sys
import pickle

# Logger konfigurieren
logger = logging.getLogger(__name__)

# Ollama Integration für KI-basierte Schlagwortauswahl
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate

from pipeline.taskfactory import TaskWithInputFileMonitor

# Import zentrale Logging-Konfiguration
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))
from pipeline_logging import setup_stage_logging

def get_normalized_keyword(word, nlp=None):
    """
    Einfache Funktion, die ein Schlagwort normalisiert (für die Deduplizierung)
    Da wir nun KI für die Auswahl verwenden, reicht eine einfache Normalisierung
    
    Parameters:
    -----------
    word: str or any
        Das zu normalisierende Wort, kann auch numerisch sein
    nlp: spacy.Language, optional
        SpaCy-Sprachmodell (nicht mehr benötigt)
    """
    # Prüfen, ob es sich um ein str-Objekt handelt
    if not isinstance(word, str):
        # Konvertiere in String
        word = str(word)
    
    # Grundlegende Normalisierung: lowercase und Leerzeichen trimmen
    return word.lower().strip()


def select_best_keyword_match(original_keyword, candidates):
    """
    Wählt den besten Treffer aus den Kandidaten für das ursprüngliche Keyword basierend auf Textähnlichkeit.
    
    Args:
        original_keyword (str): Das ursprüngliche Keyword
        candidates (list): Liste von Kandidaten-Dictionaries
        
    Returns:
        dict: Der beste übereinstimmende Kandidat oder None
    """
    if not candidates:
        return None
        
    original_norm = get_normalized_keyword(original_keyword)
    best_candidate = None
    best_score = -1
    
    for candidate in candidates:
        preferred_name = candidate.get('preferredName', '')
        if not preferred_name:
            continue
        
        # Berechne Ähnlichkeitsscore mit einer verbesserten Methode
        original_words = set(original_norm.lower().split())
        candidate_words = set(get_normalized_keyword(preferred_name).split())
        
        # Verhältnis gemeinsamer Wörter zur Gesamtzahl der Wörter
        common_words = original_words.intersection(candidate_words)
        all_words = original_words.union(candidate_words)
        
        if not all_words:  # Vermeidet Division durch Null
            score = 0
        else:
            score = len(common_words) / len(all_words)
        
        # Einfache Teilstring-Übereinstimmung berücksichtigen
        if original_norm in get_normalized_keyword(preferred_name) or get_normalized_keyword(preferred_name) in original_norm:
            score += 0.3  # Bonus für Teilstring-Übereinstimmungen
            
        if score > best_score:
            best_score = score
            best_candidate = candidate
    
    # Akzeptiere nur Kandidaten mit Mindestähnlichkeit
    if best_score < 0.1:  # Schwellenwert kann angepasst werden
        return None
        
    return best_candidate


def get_lobid_response(word, size=20):
    """
    Ruft die Lobid API auf, um Schlagwörter zu finden.
    
    Parameters:
    -----------
    word: str
        Das zu suchende Schlagwort
    size: int, optional
        Anzahl der zurückzugebenden Ergebnisse (Standard: 20)
        
    Returns:
    --------
    response: requests.Response
        Die Antwort der Lobid API
    """
    # Standard-Filter für Schlagworte verwenden
    filter_param = 'type:SubjectHeading OR type:SubjectHeadingSensoStricto'
    
    params = {
        'q': word,
        'filter': filter_param,
        'format': 'json',
        'size': size  # Erhöhen, um mehr Kandidaten für die KI-Auswahl zu erhalten
    }
    try:
        response = requests.get('https://lobid.org/gnd/search', params=params, timeout=10)
        response.raise_for_status()  # Wirft eine Exception für HTTP-Fehler
        return response
    except requests.exceptions.RequestException as e:
        logger.error(f"Fehler bei der Lobid API Anfrage für '{word}': {str(e)}")
        return None

def select_best_keyword_match_ai(original_keyword, candidates, context=None, llm=None, other_keywords=None):
    """
    Verwendet ein LLM, um den besten Treffer aus den Kandidaten für das ursprüngliche Keyword zu wählen.
    
    Args:
        original_keyword (str): Das ursprüngliche Keyword
        candidates (list): Liste von Kandidaten-Dictionaries
        context (str, optional): Kontext (z.B. Dokumenttitel, Abstract)
        llm: Ein Langchain-LLM-Objekt
        other_keywords (list, optional): Andere Keywords des Dokuments
        
    Returns:
        dict: Der beste übereinstimmende Kandidat oder None
    """
    if not candidates:
        return None
    
    try:
        # Formatiere Kandidaten für das LLM
        candidate_texts = []
        for i, candidate in enumerate(candidates):
            preferred_name = candidate.get('preferredName', '')
            gnd_id = candidate.get('gndIdentifier', '')
            type_info = candidate.get('type', [])
            type_str = ", ".join(type_info) if type_info else "Unbekannt"
            
            candidate_text = f"{i+1}. \"{preferred_name}\" (GND-ID: {gnd_id}, Typ: {type_str})"
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
        template = """Du bist ein Experte für Bibliothekskataloge und deutsche Schlagwortnormdatei (GND).
        
Originales Keyword: \"{original_keyword}\"

Kandidaten aus der GND-Datenbank:
{candidates}

{context_info}

Aufgabe: Wähle den Kandidaten aus, der am besten zum originalen Keyword passt. 
Beachte dabei folgende Kriterien:
1. Semantische Übereinstimmung ist wichtiger als exakte Wortübereinstimmung
2. Bei Institutionen, Organisationen und Universitäten ist der vollständige, offizielle Name zu bevorzugen
3. Berücksichtige den Typ des Eintrags (Person, Ort, Organisation, etc.)
4. Nutze andere Keywords des Dokuments als Kontext, um die semantische Bedeutung des aktuellen Keywords besser zu verstehen

Gib nur die Nummer des besten Kandidaten zurück. Falls kein Kandidat gut passt, antworte mit \"0\".
Antwort: """
        
        # Erstelle das Prompt
        prompt = PromptTemplate(
            template=template,
            input_variables=["original_keyword", "candidates", "context_info"]
        )
        
        # Erstelle den finalen Prompt-Text
        final_prompt = prompt.format(
            original_keyword=original_keyword,
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
        
        # Fallback zur String-Ähnlichkeit, wenn die LLM-Antwort nicht verwendbar ist
        logger.warning(f"LLM-Antwort konnte nicht interpretiert werden: {result}. Verwende String-Ähnlichkeit.")
        return select_best_keyword_match(original_keyword, candidates)
        
    except Exception as e:
        logger.error(f"Fehler bei der LLM-Auswahl: {str(e)}")
        # Fallback zur einfachen String-Ähnlichkeit
        return select_best_keyword_match(original_keyword, candidates)


def receive_lobid_keywords(word, context=None, llm=None, other_keywords=None):
    """
    Ruft Schlagwörter von Lobid ab und wählt das beste Schlagwort mit KI aus
    
    Parameters:
    -----------
    word: str
        Das zu suchende Schlagwort
    context: str, optional
        Kontext für die KI-Entscheidung (z.B. Dokumenttitel, Abstract)
    llm: OllamaLLM, optional
        Eine initialisierte LLM-Instanz, die verwendet werden soll (falls vorhanden)
    other_keywords: list, optional
        Liste anderer Keywords aus demselben Dokument für zusätzlichen Kontext
    entity_type: str, optional
        Typ der Entität, nach dem gefiltert werden soll (z.B. "CorporateBody" für Organisationen)
        
    Returns:
    --------
    dict: Informationen zum besten passenden Schlagwort oder None
    """
    # Keine automatische Institution-Erkennung mehr, wird in einem neuen Anlauf besser implementiert
    response = get_lobid_response(word)
    if not response:
        logger.debug(f"⚠ Keine Antwort von Lobid API für '{word}'")
        return None

    try:
        data = response.json()
    except ValueError as e:
        logger.error(f"Fehler beim Parsen der JSON-Antwort für '{word}': {str(e)}")
        return None

    if data.get('member') and len(data['member']) > 0:
        # Alle Mitglieder sammeln für KI-Auswahl
        candidates = data['member']
        total = data.get('totalItems', 0)
        logger.debug(f"ℹ Lobid API: {total} Treffer für '{word}' gefunden")

        # Zeige die ersten Kandidaten zur besseren Nachvollziehbarkeit
        for i, candidate in enumerate(candidates[:3]):  # Zeige maximal 3 Beispiele
            preferred_name = candidate.get('preferredName', 'Kein Name')
            types = candidate.get('type', [])
            type_info = ", ".join(types) if types else "Unbekannt"
            logger.debug(f"   Kandidat {i+1}: {preferred_name} (Typ: {type_info})")

        if len(candidates) > 3:
            logger.debug(f"   ... und {len(candidates)-3} weitere Kandidaten")

        logger.debug(f"ℹ Starte KI-Auswahl für '{word}' mit {len(candidates)} Kandidaten...")

        # Bestes Schlagwort mit KI auswählen (mit Kontext und anderen Keywords, falls vorhanden)
        best_match = select_best_keyword_match_ai(word, candidates, context, llm, other_keywords)

        if not best_match:
            logger.debug(f"⚠ KI hat keinen passenden Kandidaten für '{word}' gefunden")
            return None

        # Mit dem ausgewählten Schlagwort fortfahren
        preferred_name = best_match.get('preferredName', 'unbekannt')
        gnd_id = best_match.get('id', '')
        logger.debug(f"✓ KI-Auswahl für '{word}': \"{preferred_name}\" (ID: {gnd_id})")
            
        lobid = {
            'totalItems': data.get('totalItems', 0),
            'gnd_link': gnd_id,
            'query_term': word  # Speichern, mit welchem Begriff wir gesucht haben
        }
        
        # Zusätzliche Informationen aus dem besten Treffer extrahieren
        if 'sameAs' in best_match and best_match['sameAs']:
            lobid['sameAs_link'] = best_match['sameAs'][0]['id']
            
        # Dewey Decimal Classification, falls vorhanden
        for dewey_level in ['relatedDdcWithDegreeOfDeterminacy2', 'relatedDdcWithDegreeOfDeterminacy3']:
            if dewey_level in best_match and best_match[dewey_level]:
                ddc_id = best_match[dewey_level][0]['id']
                # Format: ddc_2, ddc_3 (kurze Bezeichner für bessere DataFrame-Handhabung)
                short_level = f"ddc_{dewey_level[-1]}"
                lobid[short_level] = ddc_id
                logger.debug(f"DDC {dewey_level[-1]}: {ddc_id}")

        # Alle bevorzugten Namen speichern
        preferred_names = [entry.get('preferredName', '') for entry in data['member']
                          if 'preferredName' in entry and entry['preferredName'] != "None"]
        if preferred_names:
            lobid['preferredNames'] = str(preferred_names)

        # Informationen zum ausgewählten Schlagwort für Debugging und zum Speichern im DataFrame
        if 'preferredName' in best_match:
            lobid['selected_name'] = best_match['preferredName']
            logger.debug(f"GND-konformes Schlagwort: {best_match['preferredName']}")

        return lobid
    else:
        logger.debug(f"⚠ Keine Ergebnisse von Lobid für Schlagwort '{word}'")

        # Bei langen Keywords versuchen wir, den Grund zu analysieren
        if len(word) > 50:
            logger.debug(f"  Hinweis: Keyword ist sehr lang ({len(word)} Zeichen), eventuell handelt es sich um mehrere kombinierte Keywords")
            # Optional: Versuch, das lange Keyword aufzuteilen
            if ',' in word:
                parts = [p.strip() for p in word.split(',')]
                logger.debug(f"  Das Keyword könnte in {len(parts)} Teile aufgeteilt werden: {', '.join(parts[:3])}...")
                logger.debug(f"  Tipp: Diese Keywords sollten separat in die Datenquelle eingetragen werden")
        return None


class GNDKeywordCheck(TaskWithInputFileMonitor):
    def __init__(self, config_stage, config_global):
        super().__init__(config_stage, config_global)
        
        # Setup zentrale Logging-Konfiguration
        self.logger_configurator = setup_stage_logging(config_global)

        stage_param = config_stage['parameters']

        # Konfiguriere LLM-Nutzung
        self.use_llm = stage_param.get('use_llm', True)
        self.llm_model = stage_param.get('llm_model', 'gemma3:27b')

        # Initialisiere Ollama LLM für KI-basierte Schlagwortauswahl
        self.llm = None
        if self.use_llm:
            try:
                self.llm = OllamaLLM(model=self.llm_model, temperature=0.0)
                # Test ob das Modell wirklich erreichbar ist
                test_response = self.llm.invoke("Test")
                logger.info(f"✓ Ollama LLM ({self.llm_model}) für Schlagwortauswahl erfolgreich initialisiert und getestet")
            except Exception as e:
                logger.error(f"✗ FEHLER: Ollama LLM ({self.llm_model}) nicht erreichbar!")
                logger.error(f"   Details: {str(e)}")
                logger.error(f"   Bitte stellen Sie sicher, dass:")
                logger.error(f"   1. Ollama läuft (ollama serve)")
                logger.error(f"   2. Das Modell '{self.llm_model}' installiert ist (ollama pull {self.llm_model})")
                logger.error(f"   3. Das Modell erreichbar ist (ollama run {self.llm_model})")
                logger.error("")
                logger.error("   Pipeline wird ABGEBROCHEN, da use_llm=True in der Konfiguration gesetzt ist.")
                raise RuntimeError(f"Ollama LLM ({self.llm_model}) ist nicht erreichbar. Pipeline abgebrochen.") from e
        self.file_name_inputs =  Path(config_global['raw_data_folder']) / stage_param['file_name_input']
        self.file_name_output =  Path(config_global['raw_data_folder']) / stage_param['file_name_output']
        self.file_folder = Path(config_global['file_folder'])
        self.processed_data_folder = config_global['processed_data_folder']
        self.keyword_list_path = Path(config_global['processed_data_folder']) / "keyword_list"

    def execute_task(self):
        # Logging wird jetzt zentral konfiguriert - keine lokalen Einstellungen mehr nötig

        df_metadata = pd.read_pickle(self.file_name_inputs)
        keyword_columns = [col for col in df_metadata.columns if 'keyword' in col]
        keyword_columns = ['ai:keywords_ext', 'ai:keywords_gen', 'ai:keywords_dnb']

        if Path(self.file_name_output).exists():
            try:
                df_checkedKeywords = pd.read_pickle(self.file_name_output)
                logger.info(f"Loaded existing keyword checks: {len(df_checkedKeywords)} records")
            except (EOFError, pickle.UnpicklingError, Exception) as e:
                logger.warning(f"Could not load existing keyword checks (corrupted file): {e}")
                logger.warning("Starting with empty keyword checks - all documents will be processed")
                # Backup corrupted file
                import shutil
                backup_path = self.file_name_output.with_suffix('.p.corrupted')
                shutil.move(str(self.file_name_output), str(backup_path))
                logger.info(f"Corrupted file moved to: {backup_path}")
                df_checkedKeywords = pd.DataFrame(index=df_metadata.index)
        else:
            # Erstelle einen leeren DataFrame mit denselben Indizes wie df_metadata
            df_checkedKeywords = pd.DataFrame(index=df_metadata.index)

        if Path(self.keyword_list_path.with_suffix('.p')).exists():
            df_keywords = pd.read_pickle(self.keyword_list_path.with_suffix('.p'))
        else:
            df_keywords = pd.DataFrame()

        # Verarbeite jede Zeile in den Metadaten
        for index, row in tqdm(df_metadata.iterrows(), desc="Checking keywords"):
            all_keywords_list = []
            
            # Log für Datensatz-Start
            logger.info(f"========== Verarbeite Datensatz: ID={row.get('pipe:ID', 'unbekannt')} ==========")

            # Sammle alle Keywords aus den verschiedenen Spalten
            for col in keyword_columns:
                if col in row and row[col] is not None:
                    if isinstance(row[col], list):
                        logger.debug(f"Gefundene Keywords in Spalte '{col}' (als Liste): {', '.join(str(k) for k in row[col])}")
                        all_keywords_list.extend(row[col])
                    elif isinstance(row[col], str) and row[col].strip():
                        # Prüfen, ob der String mehrere durch Komma getrennte Keywords enthält
                        if ',' in row[col]:
                            # Kommagetrennter String - teile auf und bereinige
                            split_keywords = [k.strip() for k in row[col].split(',') if k.strip()]
                            logger.debug(f"Gefundene kommagetrennte Keywords in Spalte '{col}': {', '.join(split_keywords)}")
                            all_keywords_list.extend(split_keywords)
                        else:
                            # Einzelnes Keyword
                            logger.debug(f"Gefundenes einzelnes Keyword in Spalte '{col}': {row[col].strip()}")
                            all_keywords_list.append(row[col].strip())

            logger.debug(f"Insgesamt {len(all_keywords_list)} Keywords für Datensatz gefunden")

            keyword_list = []

            # Erstelle eine Liste aller anderen Keywords als Kontext für KI-Entscheidungen
            all_other_keywords = all_keywords_list.copy()

            for keyword in all_keywords_list:
                logger.debug(f"------ Verarbeite Keyword: '{keyword}' ------")

                keyword_sample = {}
                keyword_sample['raw_keyword'] = keyword

                # Normalisiere das Keyword für Cache-Lookup
                normalized = get_normalized_keyword(keyword)
                keyword_sample['normalized_keyword'] = normalized

                # CACHE-LOOKUP: Prüfe, ob das Keyword bereits verarbeitet wurde
                lobid = None
                if not df_keywords.empty:
                    cached_entries = df_keywords[df_keywords['normalized_keyword'] == normalized]
                    if not cached_entries.empty:
                        # Keyword im Cache gefunden!
                        cached_entry = cached_entries.iloc[0]
                        logger.debug(f"✓ Keyword '{keyword}' aus Cache geladen (spart Lobid-API + LLM-Call)")

                        # Rekonstruiere lobid-Dictionary aus Cache
                        if cached_entry.get('is_gnd', False):
                            lobid = {
                                'selected_name': cached_entry.get('gnd_preferred_name'),
                                'gnd_link': cached_entry.get('gnd_link'),
                                'totalItems': cached_entry.get('totalItems', 0)
                            }
                            # Optionale Felder
                            for key in ['sameAs_link', 'ddc_2', 'ddc_3', 'preferredNames']:
                                if key in cached_entry and pd.notna(cached_entry[key]):
                                    lobid[key] = cached_entry[key]

                            logger.debug(f"   → Cache-Hit: '{cached_entry.get('gnd_preferred_name')}' (ID: {cached_entry.get('gnd_link')})")
                        else:
                            # Im Cache, aber kein GND-Treffer gefunden
                            logger.debug(f"   → Cache-Hit: Kein GND-Eintrag für '{keyword}' (bereits geprüft)")
                            lobid = None  # Explizit None, um anzuzeigen "bereits versucht, kein Treffer"

                # Nur wenn NICHT im Cache: Lobid abfragen
                if lobid is None and (df_keywords.empty or cached_entries.empty):
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
                        logger.debug(f"Verwende {len(other_keywords)} andere Keywords des Dokuments als Kontext für KI-Auswahl")

                    # Direkte Abfrage mit dem Original-Keyword
                    logger.debug(f"Frage Lobid für Original-Keyword '{keyword}' ab...")
                    lobid = receive_lobid_keywords(keyword, document_context, self.llm if self.use_llm else None, other_keywords)

                    # Wenn keine Ergebnisse, versuche mit normalisierter Form
                    if not lobid:
                        if normalized != keyword:
                            logger.debug(f"Keine Treffer für Original-Keyword. Versuche mit normalisierter Form: '{normalized}'")
                            lobid = receive_lobid_keywords(normalized, document_context, self.llm if self.use_llm else None)
                            if lobid:
                                # Logging wenn die normalisierte Form erfolgreich war
                                selected_name = lobid.get('selected_name', 'unknown')
                                logger.debug(f"✓ Erfolg mit normalisierter Form: '{keyword}' -> '{normalized}' -> gewählt: '{selected_name}'")

                # keyword_sample['normalized_keyword'] ist bereits oben gesetzt (Zeile 426)

                if lobid:
                    keyword_sample['is_gnd'] = True
                    selected_name = lobid.get('selected_name', 'unbekannt')
                    gnd_link = lobid.get('gnd_link', 'keine ID')
                    logger.debug(f"✓ Erfolgreich GND-Eintrag gefunden: '{selected_name}' (ID: {gnd_link})")

                    # GND-konforme Schlagworte speichern
                    if 'selected_name' in lobid:
                        keyword_sample['gnd_preferred_name'] = lobid['selected_name']
                        logger.debug(f"   + GND-konformes Schlagwort: {lobid['selected_name']}")

                    # DDC-URLs speichern
                    for dewey_level in ['ddc_2', 'ddc_3']:
                        if dewey_level in lobid:
                            keyword_sample[f'ddc_{dewey_level[-1]}_url'] = lobid[dewey_level]
                            logger.debug(f"   + DDC-{dewey_level[-1]}-URL: {lobid[dewey_level]}")

                    lobbid_keys = ['gnd_link', 'sameAs_link', 'ddc_2', 'ddc_3', 'preferredNames', 'totalItems']
                    for key in lobbid_keys:
                        if key in lobid.keys():
                            keyword_sample[key] = lobid[key]
                            if key not in ['gnd_link', 'totalItems']:
                                logger.debug(f"   + Zusatzinfo: {key}={lobid[key]}")
                else:
                    keyword_sample['is_gnd'] = False
                    logger.debug(f"✗ Kein passender GND-Eintrag gefunden für '{keyword}'")

                logger.debug(f"------ Ende Keyword '{keyword}' ------")
                keyword_sample['count'] = 0
                # Normalisierte Form für Deduplizierung nutzen
                normalized_keyword = get_normalized_keyword(keyword)
                gnd_name = keyword_sample.get('gnd_preferred_name', None)
                
                if df_keywords.shape[0] == 0:
                    df_keywords = pd.DataFrame(keyword_sample, index=[0])
                else:
                    # Prüfen, ob ein identisches Keyword (normalisiert + GND-Name) bereits existiert
                    is_duplicate = False
                    existing_idx = None
                    
                    # Erstelle eine Liste der normalisierten Formen
                    existing_normalized_forms = [get_normalized_keyword(kw) for kw in df_keywords['raw_keyword'].values]
                    
                    # Finde potentielle Duplikate basierend auf normalisiertem Keyword
                    potential_duplicates = [i for i, form in enumerate(existing_normalized_forms) if form == normalized_keyword]
                    
                    if potential_duplicates:
                        # Bei GND-Schlagworten auch den GND-Namen vergleichen
                        if gnd_name:
                            for idx in potential_duplicates:
                                if 'gnd_preferred_name' in df_keywords.columns and df_keywords.iloc[idx].get('gnd_preferred_name') == gnd_name:
                                    # Exaktes Duplikat (gleiche normalisierte Form UND gleicher GND-Name)
                                    is_duplicate = True
                                    existing_idx = idx
                                    break
                        else:
                            # Bei Nicht-GND-Keywords reicht die normalisierte Form
                            is_duplicate = True
                            existing_idx = potential_duplicates[0]
                    
                    if not is_duplicate:
                        # Kein Duplikat gefunden - neues einzigartiges Keyword
                        if gnd_name:
                            logger.debug(f"✚ Neues einzigartiges Keyword: '{keyword}' mit GND-Name '{gnd_name}' wird zur Gesamtliste hinzugefügt")
                        else:
                            logger.debug(f"✚ Neues einzigartiges Keyword: '{keyword}' wird zur Gesamtliste hinzugefügt")
                        df_keywords = pd.concat([df_keywords, pd.DataFrame(keyword_sample, index=[0])], ignore_index=True)
                    else:
                        # Duplikat gefunden - Zähler erhöhen
                        existing_count = df_keywords.at[existing_idx, 'count']
                        df_keywords.at[existing_idx, 'count'] += 1
                        existing_original = df_keywords.iloc[existing_idx]['raw_keyword']
                        if gnd_name:
                            logger.debug(f"⟳ Duplikat gefunden: '{keyword}' → '{gnd_name}' entspricht '{existing_original}' (bereits {existing_count}x gezählt)")
                        else:
                            logger.debug(f"⟳ Duplikat gefunden: '{keyword}' entspricht '{existing_original}' (bereits {existing_count}x gezählt)")

                keyword_list.append(keyword_sample)
                
            # Speichere die Keyword-Liste für diese Zeile
            # Stelle sicher, dass wir die DataFrame-Spalten korrekt anlegen, falls sie noch nicht existieren
            if 'pipe:ID' not in df_checkedKeywords.columns:
                df_checkedKeywords['pipe:ID'] = None
            if 'keywords' not in df_checkedKeywords.columns:
                # Erstellen mit Objekt-Datentyp, um komplexe Listen zu speichern
                df_checkedKeywords['keywords'] = None
                
            # Verwende iat für einzelne Zellenzuweisung, da wir einen komplexen Objekt (Liste) zuweisen
            # Das ist effizienter und verhindert den "equal len keys" Fehler
            df_checkedKeywords.at[index, 'pipe:ID'] = row['pipe:ID']
            df_checkedKeywords.at[index, 'keywords'] = keyword_list
            
            # Zusammenfassung für diesen Datensatz
            gnd_keywords = [k for k in keyword_list if k.get('is_gnd', False)]
            non_gnd_keywords = [k for k in keyword_list if not k.get('is_gnd', False)]
            
            logger.info(f"========== Zusammenfassung für Datensatz ID={row.get('pipe:ID', 'unbekannt')} ==========")
            logger.info(f"Insgesamt {len(keyword_list)} Keywords verarbeitet:")
            logger.info(f"✓ {len(gnd_keywords)} Keywords erfolgreich mit GND abgeglichen")
            logger.info(f"✗ {len(non_gnd_keywords)} Keywords ohne GND-Entsprechung")
            if non_gnd_keywords:
                logger.info(f"Nicht erkannte Keywords: {', '.join([k['raw_keyword'] for k in non_gnd_keywords])}")
            logger.info(f"===============================================================")

            # Speichere die Ergebnisse
            df_checkedKeywords.to_pickle(self.file_name_output)
            df_keywords.to_pickle(self.keyword_list_path.with_suffix('.p'))
            df_keywords.to_csv(self.keyword_list_path.with_suffix('.csv'), sep=';')

        