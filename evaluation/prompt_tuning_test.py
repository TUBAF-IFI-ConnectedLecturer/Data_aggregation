"""
Einzeldokument-Prompt-Test gegen ChromaDB + Ollama.
Testet Prompt-Varianten direkt, ohne die Pipeline zu starten.

Nutzung:
  python prompt_tuning_test.py --doc 9aG6x_KvBAzw --field author
  python prompt_tuning_test.py --doc 10940K0WUPIio --field type
  python prompt_tuning_test.py --doc 5p5gzlF3-MOg --field title
  python prompt_tuning_test.py --all-regressions
"""

import sys
sys.path.insert(0, '/media/sz/Data/Veits_pdfs/Data_aggregation/cl-pipeline/stages/general/')

import argparse
import chromadb
from langchain_ollama import OllamaLLM, OllamaEmbeddings
import yaml
import pandas as pd

# Konfiguration
CHROMA_PATH = '/media/sz/Data/Connected_Lecturers/Opal_md_comparison/processed/chroma_db'
COLLECTION = 'opal_md_comparison'
BASE_URL = 'http://localhost:11434'
EMBEDDING_MODEL = 'jina/jina-embeddings-v2-base-de'
LLM_MODEL = 'llama3.3:70b'
PROMPTS_FILE = '/media/sz/Data/Veits_pdfs/Data_aggregation/cl-pipeline/run/pipelines/opal/prompts/prompts.yaml'
OLD_META = '/media/sz/Data/Connected_Lecturers/Opal_md_comparison/raw/OPAL_ai_meta_OLD.p'
NEW_META = '/media/sz/Data/Connected_Lecturers/Opal_md_comparison/raw/OPAL_ai_meta_md.p'

SYSTEM_TEMPLATE = "### System: You are an assistant trained to support the bibliographic cataloging and indexing of academic and scientific documents. Your task is to extract structured metadata that supports library cataloging, such as author names, titles, classification codes, and keywords. Only use information explicitly available in the document. If no relevant information is found, return an empty value. Do not generate or assume content. Avoid any explanations or filler text — output must be concise and structured. All answers must be in German, unless explicitly requested otherwise. ### Context: {context} ### User: {question} ### Response:"

# Bekannte Regressionsfaelle
AUTHOR_REGRESSIONS = [
    '9aG6x_KvBAzw',  # Frank Babick verloren
    '9h_vYcI6Oin4',  # Jan Schneider verloren
    '47iFnpWswSsM',  # Thomas Koehler verloren
    '139eWuqTXgyCs', # Prof. Dr.-Ing. Wolfgang Schufft verloren
    '6CR8GpkWka1Q',  # Peter Kiessling verloren
    '4jPJTeIaaz3Y',  # Madlen Rentsch verloren
    '7rkjI9ssjrhE',  # Dipl.-Kfm. Joerg Faik verloren
]

TYPE_REGRESSIONS = [
    '10940K0WUPIio', # Lecture slides statt Vorlesungsfolien
    '12Pt-Pn39TrU8', # Lecture script statt Skript
    '6Z4w4nrMk_dA',  # Freitext statt Protokoll
    '30NMZYwmoC3A',  # Lecture slides statt Skript
    '5p5gzlF3-MOg',  # Lecture script statt Skript
]

TITLE_REGRESSIONS = [
    '5p5gzlF3-MOg',  # Titel verloren (V1 Regression)
    '6pE_uG1IzFNE',  # Titel verloren (V1 Regression)
    '14dTEbX1jY5I8', # Titel verloren (V1 Regression)
    '3NCKVBpFS-Cc',  # Titel verloren (V1 Regression)
    '38uPV4VdYK1s',  # Titel verloren (V1 Regression)
    # V2 Regressions: Titel in V1 vorhanden, V2 "Kein Titel gefunden"
    '11O6toNBMgZY0', # TW: 2. Werkstoffe -> "Kein Titel gefunden"
    '15JN3wYhf9MGo', # # Networking -> "Kein Titel gefunden"
    '47iFnpWswSsM',  # ## VL Bildungstechnologie II -> "Kein Titel gefunden"
    '4QZsL6c8nCxk',  # # Geometrie - Sommer 2021 -> "Kein Titel gefunden"
    '6Z4w4nrMk_dA',  # **Seminar Soziale...** -> "Kein Titel gefunden"
]


def get_context_for_doc(doc_id, field):
    """Holt relevante Chunks aus ChromaDB fuer ein Dokument."""
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_collection(COLLECTION)

    # Alle Chunks fuer dieses Dokument
    all_docs = collection.get(include=['documents', 'metadatas'])
    file = None
    chunks = []

    for i, meta in enumerate(all_docs['metadatas']):
        fname = meta.get('filename', '')
        if fname.startswith(doc_id):
            if file is None:
                file = fname
            page = meta.get('page', 99)
            chunks.append((page, all_docs['documents'][i]))

    if not file:
        return None, None, 0

    # Sortieren nach Seite
    chunks.sort(key=lambda x: x[0])

    # Fuer author/title/affiliation: erste Seiten (erweitert auf 0-3 fuer Markdown)
    if field in ['author', 'title', 'affiliation']:
        filtered = [(p, t) for p, t in chunks if p <= 3]
        if not filtered:
            filtered = chunks[:8]
        context_chunks = filtered[:8]
    else:
        context_chunks = chunks[:10]

    context = "\n\n".join(text for _, text in context_chunks)
    return file, context, len(chunks)


def test_prompt(doc_id, field, prompt_text, label=""):
    """Testet einen Prompt gegen ein Dokument."""
    file, context, n_chunks = get_context_for_doc(doc_id, field)
    if not file:
        print(f"  [{label}] Dokument {doc_id} nicht in ChromaDB gefunden")
        return ""

    query = prompt_text.replace("{file}", file)

    # System-Template mit Kontext fuellen
    full_prompt = SYSTEM_TEMPLATE.replace("{context}", context).replace("{question}", query)

    print(f"  [{label}] Chunks: {n_chunks}, Kontext: {len(context)} Zeichen")
    try:
        llm = OllamaLLM(model=LLM_MODEL, temperature=0, base_url=BASE_URL)
        answer = llm.invoke(full_prompt)
        if len(answer) > 300:
            answer = answer[:300] + "..."
        print(f"  [{label}] Antwort: {answer}")
        return answer
    except Exception as e:
        print(f"  [{label}] FEHLER: {e}")
        return ""


def load_reference_values():
    """Laedt alte und neue Referenzwerte."""
    old = pd.read_pickle(OLD_META).set_index('pipe:ID')
    new = pd.read_pickle(NEW_META).set_index('pipe:ID')
    return old, new


def test_type_prompts():
    """Testet den neuen deutschen Typ-Prompt gegen Regressionsfaelle."""
    print("\n" + "=" * 70)
    print("  TYP-PROMPT TUNING")
    print("=" * 70)

    old_prompt = 'You are given a document {file}. Determine what type of material it is. Choose exactly one of the following categories (English label): "Exercise sheet": a list of tasks or problems, typically used in class or for homework. "Lecture slides": visual slides used in lectures or presentations. "Lecture script": structured written notes for a lecture, often textbook-like. "Scientific paper": academic or peer-reviewed research article. "Book": a full-length or chapter of a published book. "Seminar paper": a short academic paper submitted for a seminar (e.g., 5–15 pages). "Bachelor thesis": final thesis for a bachelor\'s degree. "Master thesis": final thesis for a master\'s degree. "Doctoral dissertation": dissertation written to obtain a doctoral degree. "Documentation": user, technical, or project documentation. "Tutorial": instructional guide or how-to with practical steps. "Presentation": general presentation document, not necessarily lecture-related. "Poster": academic or scientific poster used at a conference. "Protocol": a record of an experiment, meeting, or session. "Other": if no category fits clearly. If you are not sure or cannot determine the type confidently, return an empty string. Otherwise, return only the English category name listed above — no explanation or additional text. The output must be in German.'

    new_prompt = 'Bestimme den Dokumenttyp von {file}. Waehle genau eine der folgenden Kategorien: "Uebungsblatt": Aufgaben oder Problemstellungen fuer Uebungen oder Hausaufgaben. "Vorlesungsfolien": Folien fuer Vorlesungen oder Praesentationen. "Vorlesungsskript": Strukturierte schriftliche Vorlesungsnotizen, lehrbuchaehnlich. "Wissenschaftliche Arbeit": Peer-reviewed Forschungsartikel. "Buch": Vollstaendiges Buch oder Buchkapitel. "Seminararbeit": Kurze akademische Arbeit fuer ein Seminar (5-15 Seiten). "Bachelorarbeit": Abschlussarbeit fuer den Bachelor. "Masterarbeit": Abschlussarbeit fuer den Master. "Dissertation": Doktorarbeit. "Dokumentation": Technische, Nutzer- oder Projektdokumentation. "Tutorial": Anleitung mit praktischen Schritten. "Praesentation": Allgemeine Praesentation. "Poster": Wissenschaftliches Konferenzposter. "Protokoll": Versuchs-, Sitzungs- oder Veranstaltungsprotokoll. "Sonstiges": Falls keine Kategorie klar passt. Gib NUR den Kategorienamen zurueck — keine Erklaerung, kein zusaetzlicher Text. Falls unsicher, gib einen leeren String zurueck.'

    old, new = load_reference_values()

    for doc_id in TYPE_REGRESSIONS:
        print(f"\n--- {doc_id} ---")
        old_val = str(old.loc[doc_id, 'ai:type']) if doc_id in old.index else "N/A"
        new_val = str(new.loc[doc_id, 'ai:type']) if doc_id in new.index else "N/A"
        print(f"  Referenz ALT: {old_val}")
        print(f"  Referenz NEU (V1): {new_val}")

        test_prompt(doc_id, 'type', old_prompt, "ALTER Prompt")
        test_prompt(doc_id, 'type', new_prompt, "NEUER Prompt")


def test_author_prompts():
    """Testet den neuen Autor-Prompt gegen Regressionsfaelle."""
    print("\n" + "=" * 70)
    print("  AUTOR-PROMPT TUNING")
    print("=" * 70)

    old_prompt = "Extract ONLY the document authors of {file} (NOT cited authors, references, or bibliography entries). Look specifically in: title page, document header, author block directly below the title. IGNORE: reference lists, citations in text, bibliography sections, footnotes, acknowledgments, 'References', 'Literaturverzeichnis'. If you see reference markers like [1], (2023), or 'et al.', those are NOT document authors. CRITICAL: DO NOT extract institutional names as authors. NEVER return the following as authors: University/Hochschule/Universität names (e.g., 'TU Dresden', 'HTW Dresden', 'Universität Leipzig', 'TU Bergakademie Freiberg', 'Hochschule Mittweida'), Institute/Department names (e.g., 'Medienzentrum', 'Institut für Informatik', 'Fakultät'), Organization abbreviations (e.g., 'TUD', 'HTWK', 'BA Sachsen'), Email domains or website names. Only extract PERSON names (first name and/or last name). If you see only an institution name but no person name, return an empty string. Output the name(s) only as comma-separated list — no introductions, explanations, or phrases. If no clear document authors found on the first pages, return an empty string. Your answer must be in German."

    new_prompt = "Extrahiere NUR die Dokumentautoren von {file} (NICHT zitierte Autoren, Referenzen oder Literaturverzeichniseintraege). HINWEIS: Der Dokumentinhalt liegt im Markdown-Format vor. Ignoriere Markdown-Formatierungszeichen wie #, **, *, |, > bei der Suche nach Autorennamen. Autorennamen koennen in Ueberschriften (# Name), Fettschrift (**Name**) oder Tabellen (| Name |) erscheinen. Suche speziell auf: Titelseite, Dokumentkopf, Autorenblock direkt unter dem Titel. IGNORIERE: Literaturverzeichnisse, Zitate im Text, Fussnoten, Danksagungen. KRITISCH: Extrahiere KEINE Institutionsnamen als Autoren. Gib NIEMALS folgendes als Autoren zurueck: Universitaets-/Hochschulnamen (z.B. 'TU Dresden', 'HTW Dresden'), Institut-/Abteilungsnamen (z.B. 'Medienzentrum', 'Fakultaet'), Organisationsabkuerzungen (z.B. 'TUD', 'HTWK'). Extrahiere nur PERSONENNAMEN (Vorname und/oder Nachname). Gib die Namen als kommaseparierte Liste aus — keine Einleitungen oder Erklaerungen. Falls keine Dokumentautoren auf den ersten Seiten erkennbar sind, gib einen leeren String zurueck."

    old, new = load_reference_values()

    for doc_id in AUTHOR_REGRESSIONS:
        print(f"\n--- {doc_id} ---")
        old_val = str(old.loc[doc_id, 'ai:author']) if doc_id in old.index else "N/A"
        new_val = str(new.loc[doc_id, 'ai:author']) if doc_id in new.index else "N/A"
        print(f"  Referenz ALT: {old_val}")
        print(f"  Referenz NEU (V1): {new_val}")

        test_prompt(doc_id, 'author', old_prompt, "ALTER Prompt")
        test_prompt(doc_id, 'author', new_prompt, "NEUER Prompt")


def test_title_prompts():
    """Testet den neuen Titel-Prompt gegen Regressionsfaelle."""
    print("\n" + "=" * 70)
    print("  TITEL-PROMPT TUNING")
    print("=" * 70)

    old_prompt = 'Extract the document title of {file} from the title page or header. This is typically the largest or most prominent text on the first page, positioned at the top. Look at the first page only. IGNORE: section headings, chapter titles, or content from later pages. If no title is found on the first page, return an empty string — do not write "unknown" or anything else. Return only the title — no introductory phrases or explanations. Your answer should be in German.'

    new_prompt = 'Extrahiere den Dokumenttitel von {file}. HINWEIS: Der Dokumentinhalt liegt im Markdown-Format vor. Der Titel ist FAST IMMER als erste Ueberschrift (# oder ##) oder als erster fettgedruckter Text (**...**) in den ersten Zeilen des Dokuments vorhanden. Suche gezielt nach: 1) Markdown-Ueberschriften (# Titel oder ## Titel), 2) Fettgedrucktem Text am Dokumentanfang (**Titel**), 3) Der ersten prominenten Textzeile. Schau nur auf den Beginn des Dokuments. IGNORIERE: Abschnittsueberschriften aus spaeteren Teilen, Kapitelbezeichnungen. WICHTIG: Gib NUR den reinen Titeltext zurueck, ohne Markdown-Zeichen (#, **, _). Schreibe NIEMALS Formulierungen wie \"Kein Titel gefunden\" oder \"Titel nicht erkennbar\". Falls wirklich kein Titel erkennbar ist, gib einen komplett leeren String zurueck. Gib NICHT den Dateinamen oder \"System:\" als Titel zurueck. Die Antwort muss auf Deutsch sein.'

    old, new = load_reference_values()

    for doc_id in TITLE_REGRESSIONS:
        print(f"\n--- {doc_id} ---")
        old_val = str(old.loc[doc_id, 'ai:title']) if doc_id in old.index else "N/A"
        new_val = str(new.loc[doc_id, 'ai:title']) if doc_id in new.index else "N/A"
        print(f"  Referenz ALT: {old_val}")
        print(f"  Referenz NEU (V1): {new_val}")

        test_prompt(doc_id, 'title', old_prompt, "ALTER Prompt")
        test_prompt(doc_id, 'title', new_prompt, "NEUER Prompt")


def main():
    parser = argparse.ArgumentParser(description='Prompt-Tuning Einzeltests')
    parser.add_argument('--field', choices=['type', 'author', 'title', 'all'], default='all',
                        help='Welches Feld testen')
    parser.add_argument('--all-regressions', action='store_true',
                        help='Alle bekannten Regressionsfaelle testen')
    args = parser.parse_args()

    if args.field in ['type', 'all']:
        test_type_prompts()
    if args.field in ['author', 'all']:
        test_author_prompts()
    if args.field in ['title', 'all']:
        test_title_prompts()


if __name__ == '__main__':
    main()
