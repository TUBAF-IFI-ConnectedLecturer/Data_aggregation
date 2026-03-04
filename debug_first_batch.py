"""
Diagnose-Skript: Identifiziert die ersten Dokumente im Embedding-Batch
und testet sie einzeln gegen Ollama, um das NaN-Problem zu lokalisieren.

Nutzung auf dem Remote-Rechner:
  python debug_first_batch.py /pfad/zum/raw_data_folder /pfad/zum/content_folder

Beispiel:
  python debug_first_batch.py \
    /home/crosslab/Desktop/CL/liascript/raw \
    /home/crosslab/Desktop/CL/liascript/raw/content
"""

import sys
import pandas as pd
from pathlib import Path
import re
import math

# --- Gleiche Funktionen wie in ai_embeddings.py ---

def clean_text_content(text):
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'(?i)(confidential|intern|seite \d+|page \d+|www\.[\w\.]+)', '', text)
    text = re.sub(r'[^\w\s\.,;:!?\(\)\[\]\{\}äöüÄÖÜß#*|>\-]', '', text)
    return text.strip()


def is_useful_chunk(text, min_length=100, min_words=15):
    if len(text) < min_length:
        return False
    if len(text.split()) < min_words:
        return False
    if not re.search(r'[A-ZÄÖÜ][^.!?]+[.!?]', text):
        return False
    return True


def has_nan_or_invalid(text):
    """Prüft ob Text problematische Muster enthält."""
    issues = []
    if not text or not text.strip():
        issues.append("LEER")
    if len(text) > 10000:
        issues.append(f"SEHR_LANG({len(text)})")
    # Prüfe auf hohen Anteil Nicht-Text-Zeichen
    alpha_ratio = sum(c.isalpha() for c in text) / max(len(text), 1)
    if alpha_ratio < 0.3:
        issues.append(f"WENIG_BUCHSTABEN({alpha_ratio:.1%})")
    return issues


def main():
    if len(sys.argv) < 3:
        print("Nutzung: python debug_first_batch.py <raw_data_folder> <content_folder>")
        print("Beispiel: python debug_first_batch.py /home/crosslab/Desktop/CL/liascript/raw /home/crosslab/Desktop/CL/liascript/raw/content")
        sys.exit(1)

    raw_data_folder = Path(sys.argv[1])
    content_folder = Path(sys.argv[2])
    pickle_file = raw_data_folder / "LiaScript_files_validated.p"

    print(f"Lade {pickle_file} ...")
    df = pd.read_pickle(pickle_file)
    print(f"  Gesamt: {len(df)} Dateien")

    if 'pipe:is_valid_liascript' in df.columns:
        df = df[df['pipe:is_valid_liascript'] == True]
        print(f"  Nach Validierung: {len(df)} Dateien")

    df = df[df['pipe:file_type'].isin(['md'])]
    print(f"  Nach Typ-Filter (md): {len(df)} Dateien")

    # Splitter vorbereiten
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
    except ImportError:
        print("\nERROR: langchain_text_splitters nicht installiert.")
        print("Zeige stattdessen die ersten 20 Dateien:")
        for i, (idx, row) in enumerate(df.iterrows()):
            if i >= 20:
                break
            fid = row['pipe:ID']
            md_path = content_folder / f"{fid}.md"
            exists = md_path.exists()
            print(f"  {i}: {fid}.md (exists={exists})")
        sys.exit(0)

    headers_to_split_on = [("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")]
    header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on, strip_headers=False)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, chunk_overlap=150, length_function=len,
        separators=["\n\n", "\n", ". ", ", ", " ", ""]
    )

    print(f"\n--- Erste Batch-Dokumente (BATCH_SIZE=16) ---\n")

    batch_docs = []
    files_checked = 0

    for idx, row in df.iterrows():
        source_filename = row['pipe:ID'] + "." + row['pipe:file_type']
        md_path = content_folder / (row['pipe:ID'] + ".md")

        if not md_path.exists():
            continue

        files_checked += 1
        try:
            with open(str(md_path), 'r', encoding='utf-8') as f:
                md_content = f.read()
        except Exception as e:
            continue

        if not md_content.strip():
            continue

        header_docs = header_splitter.split_text(md_content)
        for doc in header_docs:
            if len(doc.page_content) > text_splitter._chunk_size:
                sub_chunks = text_splitter.split_text(doc.page_content)
                for chunk_text in sub_chunks:
                    chunk_text = clean_text_content(chunk_text)
                    if is_useful_chunk(chunk_text):
                        batch_docs.append((source_filename, chunk_text, idx))
            else:
                cleaned = clean_text_content(doc.page_content)
                if is_useful_chunk(cleaned):
                    batch_docs.append((source_filename, cleaned, idx))

        if len(batch_docs) >= 16:
            break

    print(f"Dateien geprüft bis Batch voll: {files_checked}")
    print(f"Chunks im ersten Batch: {len(batch_docs)}\n")

    for i, (filename, text, idx) in enumerate(batch_docs[:16]):
        issues = has_nan_or_invalid(text)
        issue_str = f" *** PROBLEMATISCH: {', '.join(issues)}" if issues else ""
        print(f"[{i}] Datei: {filename} (idx={idx})")
        print(f"    Länge: {len(text)} Zeichen, {len(text.split())} Wörter{issue_str}")
        print(f"    Preview: {repr(text[:150])}")
        print()

    # Versuche einzeln zu embedden, falls Ollama verfügbar
    print("\n--- Einzeln-Embedding-Test ---\n")
    try:
        from langchain_ollama import OllamaEmbeddings
        embeddings = OllamaEmbeddings(
            base_url="http://localhost:11434",
            model="jina/jina-embeddings-v2-base-de",
        )

        for i, (filename, text, idx) in enumerate(batch_docs[:16]):
            try:
                result = embeddings.embed_documents([text])
                # Prüfe auf NaN
                has_nan = any(math.isnan(v) for v in result[0])
                status = "NaN GEFUNDEN!" if has_nan else "OK"
                print(f"[{i}] {filename}: {status}")
                if has_nan:
                    print(f"    -> Dieser Text verursacht NaN!")
                    print(f"    -> Inhalt: {repr(text[:300])}")
                    print(f"    -> Voller Text in: {content_folder / filename}")
            except Exception as e:
                print(f"[{i}] {filename}: FEHLER - {e}")

    except ImportError:
        print("langchain_ollama nicht verfügbar - Embedding-Test übersprungen.")
    except Exception as e:
        print(f"Embedding-Test fehlgeschlagen: {e}")


if __name__ == "__main__":
    main()
