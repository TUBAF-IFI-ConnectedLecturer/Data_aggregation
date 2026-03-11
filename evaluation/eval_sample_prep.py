#!/usr/bin/env python3
"""
Evaluation Sample Preparation Script

Selects 15 diverse documents from the OPAL ChromaDB collection and creates
an output file with document content (first 3 chunks) and pipeline metadata
for evaluation purposes.

Selection strategy:
- Cover all document types (ai:type) present in the corpus
- Include documents with varying chunk counts (short, medium, long)
- Include different file types (pdf, docx, pptx, xlsx)
"""

import sys
import os
import pickle
from collections import Counter

# Add module paths needed for pickle deserialization
sys.path.insert(0, '/media/sz/Data/Veits_pdfs/Data_aggregation/cl-pipeline/stages/general')
sys.path.insert(0, '/media/sz/Data/Veits_pdfs/Data_aggregation/cl-pipeline/stages/opal')

import chromadb
import pandas as pd

# ============================================================
# Configuration
# ============================================================
CHROMA_DB_PATH = '/media/sz/Data/Connected_Lecturers/Opal_md_comparison/processed/chroma_db'
COLLECTION_NAME = 'opal_md_comparison'
PICKLE_PATH = '/media/sz/Data/Connected_Lecturers/Opal_md_comparison/raw/OPAL_ai_meta_md_V3.p'
OUTPUT_PATH = '/media/sz/Data/Veits_pdfs/Data_aggregation/eval_documents.txt'
NUM_DOCS = 15
MAX_CHUNKS_SHOWN = 3
MAX_CHARS_PER_CHUNK = 2000

# ============================================================
# Step 1: Load ChromaDB data
# ============================================================
print("Loading ChromaDB collection...")
client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
collection = client.get_collection(COLLECTION_NAME)

# Get ALL documents with their content and metadata
all_data = collection.get(include=['metadatas', 'documents'])
total_chunks = len(all_data['ids'])
print(f"  Total chunks in collection: {total_chunks}")

# Build a mapping: filename -> list of (page, chunk_id, document_text)
doc_chunks = {}
for idx, (chunk_id, meta, doc_text) in enumerate(zip(
    all_data['ids'], all_data['metadatas'], all_data['documents']
)):
    fn = meta['filename']
    page = meta.get('page', 0)
    if fn not in doc_chunks:
        doc_chunks[fn] = []
    doc_chunks[fn].append({
        'chunk_id': chunk_id,
        'page': page,
        'text': doc_text if doc_text else ''
    })

# Sort chunks within each document by page number, then by chunk_id
for fn in doc_chunks:
    doc_chunks[fn].sort(key=lambda c: (c['page'], c['chunk_id']))

print(f"  Unique documents: {len(doc_chunks)}")

# Chunk count distribution
chunk_counts = {fn: len(chunks) for fn, chunks in doc_chunks.items()}
print(f"  Chunk count range: {min(chunk_counts.values())} - {max(chunk_counts.values())}")

# ============================================================
# Step 2: Load V3 metadata from pickle
# ============================================================
print("\nLoading V3 metadata from pickle...")
metadata_df = None
try:
    with open(PICKLE_PATH, 'rb') as f:
        metadata_df = pickle.load(f)
    print(f"  Loaded DataFrame with shape: {metadata_df.shape}")
    print(f"  Columns: {list(metadata_df.columns)}")
except Exception as e:
    print(f"  WARNING: Failed to load pickle: {e}")
    print("  Metadata will not be included in output.")

# Build a lookup: doc_id (without extension) -> metadata row
meta_lookup = {}
if metadata_df is not None:
    for _, row in metadata_df.iterrows():
        doc_id = row['pipe:ID']
        meta_lookup[doc_id] = row

# ============================================================
# Step 3: Select 15 diverse documents
# ============================================================
print("\nSelecting 15 diverse documents...")

# First, get type info for each document from metadata
doc_info = {}
for fn, chunks in doc_chunks.items():
    doc_id = fn.rsplit('.', 1)[0]  # Remove extension
    file_ext = fn.rsplit('.', 1)[1] if '.' in fn else ''
    n_chunks = len(chunks)
    doc_type = ''
    if doc_id in meta_lookup:
        doc_type = meta_lookup[doc_id].get('ai:type', '')
    doc_info[fn] = {
        'doc_id': doc_id,
        'file_ext': file_ext,
        'n_chunks': n_chunks,
        'doc_type': doc_type
    }

# Group by type
type_groups = {}
for fn, info in doc_info.items():
    t = info['doc_type'] if info['doc_type'] else 'Unknown'
    if t not in type_groups:
        type_groups[t] = []
    type_groups[t].append(fn)

print(f"  Document types found: {list(type_groups.keys())}")

# Selection strategy:
# 1. Ensure every document type is represented (pick 1 from each type)
# 2. For types with many documents, pick additional ones to reach 15
# 3. Prefer diversity in chunk count (short/medium/long)

selected = []
used_types = set()

# Sort types by count (ascending) so rare types get picked first
type_order = sorted(type_groups.keys(), key=lambda t: len(type_groups[t]))

# Round 1: Pick one from each type, preferring median chunk count
for doc_type in type_order:
    if len(selected) >= NUM_DOCS:
        break
    fns = type_groups[doc_type]
    # Sort by chunk count and pick the median one
    fns_sorted = sorted(fns, key=lambda fn: doc_info[fn]['n_chunks'])
    median_idx = len(fns_sorted) // 2
    chosen = fns_sorted[median_idx]
    selected.append(chosen)
    used_types.add(doc_type)

print(f"  After round 1 (one per type): {len(selected)} selected")

# Round 2: If we still need more, pick additional documents to maximize
# diversity in chunk counts. Pick from the largest type groups first.
if len(selected) < NUM_DOCS:
    remaining_pool = []
    for fn in doc_chunks:
        if fn not in selected:
            remaining_pool.append(fn)

    # Sort remaining by chunk count descending to get long documents
    remaining_pool.sort(key=lambda fn: doc_info[fn]['n_chunks'], reverse=True)

    for fn in remaining_pool:
        if len(selected) >= NUM_DOCS:
            break
        # Prefer documents that add chunk-count diversity
        selected.append(fn)

print(f"  After round 2: {len(selected)} selected")

# Final sort by chunk count for nice output ordering
selected.sort(key=lambda fn: doc_info[fn]['n_chunks'])

# Print selection summary
print("\n  Selected documents:")
print(f"  {'Filename':<30} {'Type':<25} {'Chunks':>6} {'Ext':>5}")
print(f"  {'-'*30} {'-'*25} {'-'*6} {'-'*5}")
for fn in selected:
    info = doc_info[fn]
    print(f"  {fn:<30} {info['doc_type']:<25} {info['n_chunks']:>6} {info['file_ext']:>5}")

# ============================================================
# Step 4: Generate output file
# ============================================================
print(f"\nWriting output to: {OUTPUT_PATH}")

with open(OUTPUT_PATH, 'w', encoding='utf-8') as out:
    out.write("=" * 80 + "\n")
    out.write("EVALUATION SAMPLE DOCUMENTS\n")
    out.write(f"Generated by eval_sample_prep.py\n")
    out.write(f"Total documents in corpus: {len(doc_chunks)}\n")
    out.write(f"Selected for evaluation: {len(selected)}\n")
    out.write("=" * 80 + "\n\n")

    for doc_idx, fn in enumerate(selected, 1):
        info = doc_info[fn]
        doc_id = info['doc_id']
        chunks = doc_chunks[fn]

        out.write("=" * 80 + "\n")
        out.write(f"DOCUMENT {doc_idx}/{len(selected)}\n")
        out.write(f"Document ID: {doc_id}\n")
        out.write(f"Filename: {fn}\n")
        out.write(f"File type: {info['file_ext']}\n")
        out.write(f"Number of chunks: {info['n_chunks']}\n")
        out.write("=" * 80 + "\n\n")

        # --- Document Content (first 3 chunks) ---
        out.write("-" * 60 + "\n")
        out.write("DOCUMENT CONTENT (first 3 chunks)\n")
        out.write("-" * 60 + "\n\n")

        chunks_to_show = chunks[:MAX_CHUNKS_SHOWN]
        for ci, chunk in enumerate(chunks_to_show, 1):
            text = chunk['text']
            truncated = len(text) > MAX_CHARS_PER_CHUNK
            display_text = text[:MAX_CHARS_PER_CHUNK]

            out.write(f"--- Chunk {ci} (ID: {chunk['chunk_id']}, Page: {chunk['page']}) ---\n")
            out.write(display_text)
            if truncated:
                out.write(f"\n[... TRUNCATED at {MAX_CHARS_PER_CHUNK} chars, full chunk: {len(text)} chars ...]\n")
            out.write("\n\n")

        if len(chunks) > MAX_CHUNKS_SHOWN:
            out.write(f"[{len(chunks) - MAX_CHUNKS_SHOWN} additional chunks not shown]\n\n")

        # --- Pipeline Metadata ---
        out.write("-" * 60 + "\n")
        out.write("PIPELINE METADATA (V3)\n")
        out.write("-" * 60 + "\n\n")

        if doc_id in meta_lookup:
            row = meta_lookup[doc_id]
            metadata_fields = [
                ('ai:title', 'Title'),
                ('ai:author', 'Author'),
                ('ai:type', 'Type'),
                ('ai:keywords_ext', 'Keywords (extracted)'),
                ('ai:keywords_dnb', 'Keywords (DNB)'),
                ('ai:dewey', 'Dewey Classification'),
                ('ai:summary', 'Summary'),
            ]
            for field_key, field_label in metadata_fields:
                value = row.get(field_key, 'N/A')
                if value is None:
                    value = 'N/A'
                # Format dewey specially (it's a list of dicts)
                if field_key == 'ai:dewey' and isinstance(value, list):
                    dewey_strs = []
                    for d in value:
                        notation = d.get('notation', '')
                        label = d.get('label', '')
                        score = d.get('score', '')
                        dewey_strs.append(f"{notation} - {label} (score: {score})")
                    value = '; '.join(dewey_strs)
                out.write(f"  {field_label}: {value}\n")
        else:
            out.write("  [No metadata found for this document in V3 pickle]\n")

        out.write("\n\n")

print(f"\nDone! Output written to: {OUTPUT_PATH}")
print(f"File size: {os.path.getsize(OUTPUT_PATH):,} bytes")
