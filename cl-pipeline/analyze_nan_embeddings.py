"""
Analyze documents that produce NaN embeddings with qwen3-embedding.
Reads the validated LiaScript files and tests each problematic document
to identify what causes the NaN error.

Usage:
    python analyze_nan_embeddings.py [pickle_path] [content_folder]

Defaults:
    pickle_path:    /home/crosslab/Desktop/CL/liascript/raw/LiaScript_files_validated.p
    content_folder: /home/crosslab/Desktop/CL/liascript/raw/content
"""

import pandas as pd
import sys
from pathlib import Path

try:
    from langchain_ollama import OllamaEmbeddings
except ImportError:
    from langchain_community.embeddings import OllamaEmbeddings

# Configuration
PICKLE_PATH = sys.argv[1] if len(sys.argv) > 1 else "/home/crosslab/Desktop/CL/liascript/raw/LiaScript_files_validated.p"
CONTENT_FOLDER = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("/home/crosslab/Desktop/CL/liascript/raw/content")
BASE_URL = "http://localhost:11434"
MODEL = "qwen3-embedding"

print(f"Pickle:  {PICKLE_PATH}")
print(f"Content: {CONTENT_FOLDER}")
print(f"Model:   {MODEL}")
print()

# Load data
df = pd.read_pickle(PICKLE_PATH)
if 'pipe:is_valid_liascript' in df.columns:
    df = df[df['pipe:is_valid_liascript'] == True]
df = df[df['pipe:file_type'] == 'md'].reset_index(drop=True)
print(f"Total valid MD files: {len(df)}")

# Setup embeddings
embeddings = OllamaEmbeddings(base_url=BASE_URL, model=MODEL)

# Warmup
print("Warming up model...")
try:
    embeddings.embed_query("warmup test")
    print("Model ready.\n")
except Exception as e:
    print(f"Warmup failed: {e}")
    sys.exit(1)

# Test problematic range
print("=" * 80)
print("Testing documents in range 515-545 (where NaN errors occurred)")
print("=" * 80)

failed_docs = []

for idx in range(515, 546):
    if idx >= len(df):
        break
    row = df.iloc[idx]
    doc_id = row['pipe:ID']
    content_file = CONTENT_FOLDER / f"{doc_id}.md"

    user = row.get('user', '?')
    repo = row.get('repo', '?')

    if not content_file.exists():
        print(f"  [{idx}] {doc_id} - FILE NOT FOUND")
        continue

    with open(content_file, 'r', errors='replace') as f:
        text = f.read()

    # Detect characteristics
    has_non_ascii = any(ord(c) > 127 for c in text[:2000])
    has_cjk = any('\u4e00' <= c <= '\u9fff' for c in text[:2000])
    has_arabic = any('\u0600' <= c <= '\u06ff' for c in text[:2000])
    has_cyrillic = any('\u0400' <= c <= '\u04ff' for c in text[:2000])

    # Try embedding first 500 chars
    try:
        embeddings.embed_query(text[:500])
        status = "OK"
    except Exception as e:
        status = f"FAILED: {e}"
        failed_docs.append(idx)

    lang_info = []
    if has_cjk: lang_info.append("CJK")
    if has_arabic: lang_info.append("Arabic")
    if has_cyrillic: lang_info.append("Cyrillic")
    if has_non_ascii and not lang_info: lang_info.append("non-ASCII")
    lang_str = f" [{', '.join(lang_info)}]" if lang_info else ""

    print(f"  [{idx}] {doc_id} ({user}/{repo}){lang_str} len={len(text)} - {status}")

print()

# Deep analysis of failed documents
if failed_docs:
    print("=" * 80)
    print(f"Deep analysis of {len(failed_docs)} failed documents")
    print("=" * 80)

    for idx in failed_docs:
        row = df.iloc[idx]
        doc_id = row['pipe:ID']
        content_file = CONTENT_FOLDER / f"{doc_id}.md"

        with open(content_file, 'r', errors='replace') as f:
            text = f.read()

        print(f"\n--- [{idx}] {doc_id} (len={len(text)}) ---")
        print(f"First 300 chars: {repr(text[:300])}")

        # Test different chunk sizes to find where it breaks
        for chunk_size in [100, 200, 500, 1000, 2000]:
            chunk = text[:chunk_size]
            try:
                embeddings.embed_query(chunk)
                result = "OK"
            except:
                result = "FAIL"
            print(f"  First {chunk_size} chars: {result}")

        # Test if it's specific characters
        # Try with only ASCII
        ascii_text = text[:500].encode('ascii', errors='ignore').decode('ascii')
        try:
            embeddings.embed_query(ascii_text)
            print(f"  ASCII-only (500 chars): OK -> non-ASCII chars cause the issue")
        except:
            print(f"  ASCII-only (500 chars): FAIL -> not a character encoding issue")

else:
    print("No failures detected in this run!")
    print("The NaN issue may be sporadic/timing-related rather than content-related.")

print("\nDone.")
