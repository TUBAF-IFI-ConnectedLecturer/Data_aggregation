"""
Diagnose NaN embedding failures using real pipeline documents.
Simulates the exact pipeline behavior: splits markdown files into chunks
and sends all chunks of a file as a batch (like embed_documents does).

Usage:
    python diagnose_nan_embeddings.py [pickle_path] [content_folder] [base_url]

Defaults:
    pickle_path:    /home/crosslab/Desktop/CL/liascript/raw/LiaScript_files_validated.p
    content_folder: /home/crosslab/Desktop/CL/liascript/raw/content
    base_url:       http://localhost:11434
"""

import sys
import time
import json
import requests

BASE_URL = sys.argv[3] if len(sys.argv) > 3 else (sys.argv[1] if len(sys.argv) == 2 else "http://localhost:11434")
MODEL = "qwen3-embedding"

# Check if we have real data or just run with synthetic batches
USE_REAL_DATA = len(sys.argv) > 2 or len(sys.argv) == 1

if USE_REAL_DATA:
    try:
        import pandas as pd
        from pathlib import Path
        from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter

        PICKLE_PATH = sys.argv[1] if len(sys.argv) > 2 else "/home/crosslab/Desktop/CL/liascript/raw/LiaScript_files_validated.p"
        CONTENT_FOLDER = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("/home/crosslab/Desktop/CL/liascript/raw/content")

        if not Path(PICKLE_PATH).exists():
            print(f"Pickle not found at {PICKLE_PATH}, falling back to synthetic data")
            USE_REAL_DATA = False
    except ImportError:
        print("pandas/langchain not available, falling back to synthetic data")
        USE_REAL_DATA = False


def get_ollama_model_info():
    try:
        r = requests.get(f"{BASE_URL}/api/ps", timeout=5)
        return r.json()
    except:
        return None


def embed_batch(texts):
    """Send batch embedding request (like embed_documents does)."""
    try:
        r = requests.post(
            f"{BASE_URL}/api/embed",
            json={"model": MODEL, "input": texts},
            timeout=60
        )
        if r.status_code == 200:
            return True, None, len(texts)
        else:
            return False, f"Status {r.status_code}: {r.text[:200]}", len(texts)
    except Exception as e:
        return False, str(e), len(texts)


def embed_single(text):
    """Send single embedding request."""
    try:
        r = requests.post(
            f"{BASE_URL}/api/embed",
            json={"model": MODEL, "input": text},
            timeout=30
        )
        if r.status_code == 200:
            return True, None
        else:
            return False, f"Status {r.status_code}: {r.text[:200]}"
    except Exception as e:
        return False, str(e)


print(f"Diagnosing NaN embedding failures (batch mode)")
print(f"Model:    {MODEL}")
print(f"Base URL: {BASE_URL}")
print(f"Data:     {'Real documents' if USE_REAL_DATA else 'Synthetic batches'}")
print()

# Initial state
model_info = get_ollama_model_info()
if model_info:
    print(f"Ollama models: {json.dumps(model_info, indent=2)[:500]}")
    print()

# Warmup
print("Warming up model...")
success, error = embed_single("warmup test")
if not success:
    print(f"Warmup failed: {error}")
    sys.exit(1)
print("Model ready.\n")

first_failure = None
consecutive_failures = 0
total_failures = 0
total_files = 0

if USE_REAL_DATA:
    # Load real data
    df = pd.read_pickle(PICKLE_PATH)
    if 'pipe:is_valid_liascript' in df.columns:
        df = df[df['pipe:is_valid_liascript'] == True]
    df = df[df['pipe:file_type'] == 'md']
    print(f"Loaded {len(df)} valid MD files")
    print()

    # Setup splitters (same as pipeline)
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")],
        strip_headers=False,
    )
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, chunk_overlap=150, length_function=len,
        separators=["\n\n", "\n", ". ", ", ", " ", ""]
    )

    print(f"{'File':>5} | {'Chunks':>6} | {'Status':>6} | {'Time':>7} | {'TotalChars':>10} | Notes")
    print("-" * 90)

    for file_num, (idx, row) in enumerate(df.iterrows()):
        doc_id = row['pipe:ID']
        content_file = CONTENT_FOLDER / f"{doc_id}.md"

        if not content_file.exists():
            continue

        with open(content_file, 'r', errors='replace') as f:
            text = f.read()

        # Split like pipeline does
        md_splits = header_splitter.split_text(text)
        chunks = []
        for doc in md_splits:
            sub_chunks = text_splitter.split_text(doc.page_content)
            chunks.extend(sub_chunks)

        if not chunks:
            continue

        total_files += 1
        total_chars = sum(len(c) for c in chunks)

        # Send as batch (like embed_documents)
        t0 = time.time()
        success, error, n = embed_batch(chunks)
        elapsed = time.time() - t0

        notes = ""
        if success:
            consecutive_failures = 0
            status = "OK"
        else:
            total_failures += 1
            consecutive_failures += 1
            status = "FAIL"
            if first_failure is None:
                first_failure = total_files
                notes = f"<-- FIRST FAILURE (idx={idx}): {error[:60]}"
            elif consecutive_failures == 3:
                notes = "<-- 3 consecutive failures"

                # Diagnose: try chunks individually
                print(f"\n  Diagnosing failed file {doc_id} ({len(chunks)} chunks):")
                for ci, chunk in enumerate(chunks):
                    s, e = embed_single(chunk)
                    cstatus = "OK" if s else "FAIL"
                    print(f"    Chunk {ci}: {cstatus} (len={len(chunk)}, preview={repr(chunk[:60])})")
                print()

        if not success or total_files % 20 == 0 or total_files <= 5:
            print(f"{total_files:>5} | {len(chunks):>6} | {status:>6} | {elapsed:>6.2f}s | {total_chars:>10} | {notes}")

        if consecutive_failures >= 20:
            print(f"\n20 consecutive failures - stopping.")
            break

        if total_files >= 500:
            print(f"\nReached 500 files limit.")
            break

else:
    # Synthetic batch tests with increasing batch sizes
    print("Testing with synthetic batches of increasing size")
    print(f"{'Test':>5} | {'Chunks':>6} | {'Status':>6} | {'Time':>7} | {'TotalChars':>10} | Notes")
    print("-" * 90)

    base_text = "Dies ist ein Beispieltext für die Einbettung von Dokumenten in einem Vektorraum. " * 5

    for batch_size in [1, 2, 4, 8, 16, 32, 64]:
        for repeat in range(5):
            total_files += 1
            chunks = [base_text + f" Variante {i} Batch {repeat}" for i in range(batch_size)]
            total_chars = sum(len(c) for c in chunks)

            t0 = time.time()
            success, error, n = embed_batch(chunks)
            elapsed = time.time() - t0

            status = "OK" if success else "FAIL"
            notes = ""
            if not success:
                total_failures += 1
                consecutive_failures += 1
                if first_failure is None:
                    first_failure = total_files
                    notes = f"<-- FIRST FAILURE: {error[:60]}"
            else:
                consecutive_failures = 0

            print(f"{total_files:>5} | {batch_size:>6} | {status:>6} | {elapsed:>6.2f}s | {total_chars:>10} | {notes}")

print()
print("=" * 80)
print(f"Summary:")
print(f"  Total files tested: {total_files}")
print(f"  Failures:           {total_failures}")
if first_failure is not None:
    print(f"  First failure:      file #{first_failure}")
    print(f"  Failure rate:       {total_failures/total_files*100:.1f}%")
else:
    print("  No failures!")

print("\nDone.")
