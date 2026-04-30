# Test-Pipeline für Prompt-Verbesserungen

## 📋 Übersicht

Dieses Test-Setup ermöglicht schnelles Testen der verbesserten Prompts auf 50 Dokumenten statt aller 1010.

## 🎯 Ziel

Validierung der **verbesserten Titel-Extraktion** (und anderer Prompt-Verbesserungen) bevor die gesamte Pipeline neu läuft.

**Erwartete Verbesserung:**
- Titel-Übereinstimmung: Von **6.4%** auf **>30%**

## 📁 Dateien

```
config/
├── test_improved.yaml      # Test-Config mit verbesserten Prompts
└── test.yaml              # Alte Test-Config (Backup)

scripts/
└── create_test_subset.py   # Erstellt 50-Dokument-Subset

/data_pipeline/local_pdfs/raw/
├── LOCAL_files_base_test.p      # 50 Test-Dokumente
└── LOCAL_ai_meta_test.p         # Output (nach Pipeline-Run)
```

## 🚀 Test-Pipeline ausführen

### Schritt 1: Test-Subset erstellen (bereits erledigt ✅)

```bash
cd cl-pipeline
pipenv run python3 run/pipelines/local_pdfs/scripts/create_test_subset.py
```

**Output:** `LOCAL_files_base_test.p` mit 50 zufällig ausgewählten Dokumenten

### Schritt 2: Test-Pipeline starten

```bash
cd cl-pipeline/run
pipenv run python3 run_pipeline.py -c pipelines/local_pdfs/config/test_improved.yaml
```

**Laufzeit:** ~35-60 Minuten für 50 Dokumente

**Stages:**
1. ✓ Extract PDF metadata (~10s) - schnell, gecacht
2. ✓ Extract content (~2-3 min) - gecacht wenn schon gelaufen
3. ✓ Filter by content (~10s)
4. ✓ Generate embeddings (~5-10 min) - gecacht wenn schon gelaufen
5. **Extract metadata with AI (~15-25 min)** ← WICHTIG: Neue Prompts!
6. **GND keyword check (~10-20 min)** ← Mit Cache-Optimierung

### Schritt 3: Ergebnisse analysieren

```bash
cd cl-pipeline
pipenv run python3 << 'EOF'
import pandas as pd
from pathlib import Path
import difflib

# Lade Test-Ergebnisse
base_path = Path('/media/sz/Data/Veits_pdfs/data_pipeline/local_pdfs/raw')
df_base = pd.read_pickle(base_path / 'LOCAL_files_base_test.p')
df_ai = pd.read_pickle(base_path / 'LOCAL_ai_meta_test.p')

# Merge
df = df_base.merge(df_ai, on='pipe:ID')

print("=== TITEL-VERGLEICH (Test-Set: 50 Dokumente) ===\n")

exact = 0
partial = 0
failed = 0
examples = []

for idx, row in df.iterrows():
    bib_title = str(row['bibtex:title']).lower().replace('{', '').replace('}', '')
    ai_title = str(row['ai:title']).lower()

    if bib_title == 'nan' or ai_title == 'nan':
        continue

    similarity = difflib.SequenceMatcher(None, bib_title, ai_title).ratio()

    if bib_title == ai_title:
        exact += 1
    elif similarity > 0.7:
        partial += 1
    else:
        failed += 1
        if len(examples) < 5:
            examples.append((row['pipe:ID'], similarity, bib_title[:60], ai_title[:60]))

total = len(df)
print(f"Gesamt: {total} Dokumente")
print(f"✓ Exakte Übereinstimmung: {exact} ({exact/total*100:.1f}%)")
print(f"~ Teilweise (>70%): {partial} ({partial/total*100:.1f}%)")
print(f"✗ Fehlgeschlagen (<70%): {failed} ({failed/total*100:.1f}%)")

if examples:
    print("\n=== Beispiele fehlgeschlagener Extraktionen ===")
    for doc_id, sim, bib, ai in examples:
        print(f"\nID {doc_id} ({sim:.1%} Ähnlichkeit)")
        print(f"  BibTeX: {bib}...")
        print(f"  LLM:    {ai}...")

# Vergleich mit Baseline
print("\n=== Vergleich mit Vollständigem Datensatz ===")
print("Baseline (965 Dokumente, alter Prompt):")
print("  Exakt: 6.4%, Teilweise: 16.8%, Fehlgeschlagen: 67.4%")
print(f"\nTest-Set (50 Dokumente, neuer Prompt):")
print(f"  Exakt: {exact/total*100:.1f}%, Teilweise: {partial/total*100:.1f}%, Fehlgeschlagen: {failed/total*100:.1f}%")

EOF
```

## 📊 Was wird getestet?

### Stage 5: AI Metadata Extraction

**Neue Prompts** (aus `prompts_scientific_papers.yaml`):

**Titel-Prompt (verbessert):**
```
"Extract ONLY the MAIN document title from the FIRST page.
Rules:
1) Look for the LARGEST, most prominent text at the TOP of page 1
2) IGNORE: section headings, referenced titles in lists/agendas
3) Return ONLY ONE title as plain text
4) NO lists like '1. Title A 2. Title B'
..."
```

**Vorher vs. Nachher:**
- Vorher: "Extract the document title from the title page..."
- Problem: LLM extrahierte oft Listen von Titeln aus Referenzen
- Jetzt: Explizite IGNORE-Liste und Format-Vorgaben

### Stage 6: GND Keyword Check

**Optimierungen:**
- ✅ Cache-Lookup vor Lobid-API (spart 92% der Calls)
- ✅ Nur neue Keywords werden validiert

## 🎯 Erfolgskriterien

**Minimum:**
- Titel-Exakt: **>20%** (aktuell 6.4%)
- Titel-Teilweise: **>30%** (aktuell 16.8%)

**Ziel:**
- Titel-Exakt: **>30%**
- Titel-Teilweise: **>40%**

**Optimal:**
- Titel-Exakt: **>50%**
- Titel-Teilweise: **>30%**

## 🔄 Nach erfolgreichen Test

Wenn die Ergebnisse gut sind:

### Option 1: Volle Pipeline neu laufen lassen

```bash
cd cl-pipeline/run

# Stage 5 mit force_run: True
# Editiere full.yaml: Zeile 104 ändern zu force_run: True
pipenv run python3 run_pipeline.py -c pipelines/local_pdfs/config/full.yaml
```

**Dauer:** ~8-12 Stunden für 965 Dokumente

### Option 2: Nur fehlgeschlagene Dokumente neu verarbeiten

Erstelle eine Whitelist der Dokumente mit schlechter Übereinstimmung und verarbeite nur diese neu.

## 📝 Konfigurationsunterschiede: test_improved.yaml vs. full.yaml

| Parameter | test_improved.yaml | full.yaml | Grund |
|-----------|-------------------|-----------|-------|
| `file_name_input` | `LOCAL_files_base_test.p` | `LOCAL_files_base.p` | Test-Subset |
| `collection_name` | `local_pdfs_test_collection` | `local_pdfs_collection` | Separate ChromaDB |
| `chroma_db_folder` | `chroma_db_local_test` | `chroma_db_local` | Separate DB |
| `force_run` (Stage 5) | `True` | `False` | Immer neu für Test |
| `force_processing` | Titel, Autor, Keywords | `[]` | Teste alle relevanten Felder |

## 🐛 Troubleshooting

### Fehler: "No module named 'checkAuthorNames'"

```bash
cd cl-pipeline/run
pipenv run python3 run_pipeline.py -c pipelines/local_pdfs/config/test_improved.yaml
```

Wichtig: Von `cl-pipeline/run` Verzeichnis ausführen!

### Fehler: "Ollama connection refused"

```bash
# Prüfe SSH-Tunnel
ps aux | grep "ssh.*11434"

# Falls nicht aktiv:
ssh -L 11434:localhost:11434 dgx.ollama.extern
```

### ChromaDB-Fehler

```bash
# Lösche Test-ChromaDB und starte neu
rm -rf /media/sz/Data/Veits_pdfs/data_pipeline/local_pdfs/processed/chroma_db_local_test
```

## 📊 Erwartete Output-Dateien

Nach erfolgreichem Run:

```
/media/sz/Data/Veits_pdfs/data_pipeline/local_pdfs/raw/
├── LOCAL_files_base_test.p           (Input: 50 Dokumente)
├── LOCAL_files_meta_test.p           (Stage 1 Output)
├── LOCAL_content_test.p              (Stage 2 Output)
├── LOCAL_files_filtered_test.p       (Stage 3 Output)
├── LOCAL_embeddings_files_test.p     (Stage 4 Output)
├── LOCAL_ai_meta_test.p              (Stage 5 Output) ← WICHTIG!
└── LOCAL_checked_keywords_test.p     (Stage 6 Output)
```

---

**Erstellt:** 2025-12-07
**Zweck:** Schnelles Testen von Prompt-Verbesserungen
**Nächster Schritt:** Test-Pipeline starten und Ergebnisse analysieren
