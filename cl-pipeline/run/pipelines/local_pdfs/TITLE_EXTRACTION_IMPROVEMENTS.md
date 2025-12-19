# Verbesserungsvorschläge: Titel-Extraktion

## 📊 Aktueller Status (Problematisch!)

### Messungen
- **Exakte Übereinstimmung**: 62/965 = **6.4%** ❌
- **Teilweise Übereinstimmung** (>70%): 162/965 = **16.8%**
- **Keine Übereinstimmung** (<70%): 650/965 = **67.4%** ❌

### Hauptproblem
Das LLM extrahiert oft **mehrere Titel** aus Referenzlisten, Agenden oder Inhaltsverzeichnissen statt des Haupt-Dokumenttitels.

**Beispiel ID 8008:**
```
BibTeX Titel (korrekt):
  "Satelliten-Konferenz „Wissenschaftsgeleitetes Open-Access-Publizieren""

LLM Titel (falsch):
  "document titles:
   1. Berliner Erklärung
   2. Wege des hochwertigen, transparenten...
   3. Towards Responsible Publishing
   ..."
```

## 🔧 Implementierte Verbesserungen

### 1. Verbesserter Titel-Prompt (ERLEDIGT ✅)

**Datei**: `prompts_scientific_papers.yaml`, Zeile 12

**Vorher:**
```yaml
title: "Extract the document title from the title page or header.
       This is typically the largest text on the first page,
       positioned prominently at the top."
```

**Nachher:**
```yaml
title: "Extract ONLY the MAIN document title from the FIRST page.
       Rules:
       1) Look for the LARGEST, most prominent text at the TOP of page 1
       2) This is usually above or near the author names
       3) IGNORE: section headings, referenced titles in lists/agendas,
                  figure captions, headers/footers, institution names
       4) Return ONLY ONE title as plain text
       5) NO lists like '1. Title A 2. Title B'
       6) NO prefix like 'Document title:'
       7) If title spans multiple lines, combine them

       Example good output:
         'Wissenschaftsgeleitetes Open-Access-Publizieren in Deutschland'

       Example bad output:
         'document titles: 1. Title A 2. Title B'"
```

**Verbesserungen:**
- ✅ Explizite Anweisung: NUR EIN Titel
- ✅ Konkrete IGNORE-Liste (Referenzen, Agenden, etc.)
- ✅ Beispiele für richtig/falsch
- ✅ Verbot von Listen-Formaten
- ✅ Verbot von Präfixen wie "Document title:"

### 2. Retrieval-Strategie (BEREITS IMPLEMENTIERT ✅)

**Datei**: `ai_metadata.py`, Zeile 159-166

Der Code nutzt bereits die **ersten 2 Seiten** für Titelextraktion:

```python
if field_name in ['ai:author', 'ai:title', 'ai:affiliation']:
    # Authors, titles, and affiliations are almost always on first 1-2 pages
    page_filter = [
        {"page": 0},
        {"page": 1},
    ]
    k_chunks = 5  # Limit to 5 chunks from first pages
```

✅ **Gut konfiguriert** - keine Änderung nötig.

## 🎯 Weitere Verbesserungsvorschläge (noch nicht implementiert)

### 3. PDF-Metadaten als Fallback

**Problem**: Viele PDFs haben eingebettete Metadaten im `file:title` Feld.

**Vorschlag**:
```python
def process_title(self, file: str, chain: Any, file_metadata: Dict) -> Dict[str, str]:
    """Extract document title with PDF metadata fallback"""

    # Primär: LLM-Extraktion
    title_query = self.prompt_manager.get_document_prompt("title", file)
    llm_title = self.llm_interface.get_monitored_response(title_query, chain)
    llm_title = self.response_filter.filter_response(llm_title)

    # Fallback: PDF-Metadaten
    pdf_title = file_metadata.get('file:title', '').strip()

    # Validierung: LLM-Titel sollte nicht wie eine Liste aussehen
    if llm_title and not re.match(r'^\d+\.|^document titles?:', llm_title.lower()):
        return {'ai:title': llm_title}
    elif pdf_title:
        logging.warning(f"LLM title suspicious for {file}, using PDF metadata: {pdf_title}")
        return {'ai:title': pdf_title, 'ai:title_source': 'pdf_metadata'}
    else:
        return {'ai:title': llm_title}  # Behalte trotzdem, aber logge Warning
```

**Vorteil**:
- PDF-Metadaten sind oft korrekt (aus Zotero/BibTeX-Import)
- Fallback bei LLM-Fehlern

**Aufwand**: Mittel (ca. 1-2 Stunden)

### 4. Post-Processing Filter

**Problem**: LLM gibt manchmal Listen zurück trotz Anweisung.

**Vorschlag**: Response-Filter erweitern:

```python
class TitleFilter:
    """Filter and validate extracted titles"""

    @staticmethod
    def clean_title(title: str) -> str:
        """Remove common extraction artifacts"""
        # Remove list prefixes
        title = re.sub(r'^(document\s+)?titles?:\s*', '', title, flags=re.IGNORECASE)
        title = re.sub(r'^\d+\.\s*', '', title)  # "1. Title" -> "Title"

        # If multiple lines, take only first non-empty line
        lines = [l.strip() for l in title.split('\n') if l.strip()]
        if lines:
            title = lines[0]

        # Remove trailing dots/commas
        title = title.rstrip('.,')

        return title

    @staticmethod
    def is_valid_title(title: str) -> bool:
        """Check if extracted title looks reasonable"""
        if not title or len(title) < 10:
            return False

        # Suspicious patterns
        suspicious = [
            r'^\d+\.',  # Starts with number
            r'document titles?:',  # Contains "document title:"
            r'\d+\..+\d+\.',  # Multiple numbered items "1. A 2. B"
        ]

        for pattern in suspicious:
            if re.search(pattern, title, re.IGNORECASE):
                return False

        return True
```

**Aufwand**: Gering (ca. 30 Minuten)

### 5. Zwei-Schritt-Extraktion

**Problem**: Komplexe Layouts (Titelseiten mit Logos, Grafiken) verwirren das LLM.

**Vorschlag**:
1. **Schritt 1**: "Welche Texte sind auf Seite 1 vorhanden? Liste alle Textblöcke."
2. **Schritt 2**: "Welcher der folgenden Texte ist der Haupttitel? {text_blocks}"

**Vorteil**:
- LLM sieht explizit alle Optionen
- Kann besser zwischen Titel und anderen Elementen unterscheiden

**Nachteil**:
- Doppelter LLM-Call pro Dokument
- Höhere Kosten/Zeit

**Aufwand**: Hoch (ca. 3-4 Stunden)

### 6. Modell-Wechsel für Titel

**Problem**: llama3:70b könnte für diese präzise Aufgabe zu groß/generisch sein.

**Vorschlag**: Kleineres, spezialisiertes Modell nur für Titelextraktion.

**Config-Änderung**:
```yaml
# Stage 5: AI-based metadata extraction
- name: Extract metadata with AI
  class: AIMetaDataExtraction
  parameters:
    model_name: llama3:70b  # Standard für alle Felder

    # Field-specific models
    field_models:
      ai:title: gemma3:27b      # Kleiner, schneller für einfache Extraktion
      ai:keywords_ext: llama3:70b  # Komplex, braucht großes Modell
```

**Vorteil**:
- Schnellere Verarbeitung für Titel
- Evtl. präziser (kleine Modelle folgen Anweisungen manchmal besser)

**Aufwand**: Mittel (ca. 2 Stunden Implementierung)

### 7. Hybrid-Ansatz: LLM + Heuristik

**Idee**: Kombiniere LLM-Extraktion mit regelbasierten Heuristiken.

```python
def extract_title_hybrid(pdf_path, llm_title, file_metadata):
    """Combine LLM extraction with heuristics"""

    # 1. LLM-Titel
    llm_score = score_title_quality(llm_title)

    # 2. PDF-Metadaten
    pdf_title = file_metadata.get('file:title', '')
    pdf_score = score_title_quality(pdf_title)

    # 3. Heuristik: Größter Text auf Seite 1
    heuristic_title = extract_largest_text_block(pdf_path, page=0)
    heuristic_score = score_title_quality(heuristic_title)

    # Wähle besten Kandidaten
    candidates = [
        (llm_title, llm_score, 'llm'),
        (pdf_title, pdf_score, 'pdf'),
        (heuristic_title, heuristic_score, 'heuristic')
    ]

    best_title, best_score, source = max(candidates, key=lambda x: x[1])

    return {
        'ai:title': best_title,
        'ai:title_source': source,
        'ai:title_confidence': best_score
    }
```

**Aufwand**: Hoch (ca. 4-5 Stunden)

## 📝 Empfohlene Umsetzungsreihenfolge

### Sofort (bereits erledigt):
1. ✅ **Verbesserter Prompt** (Zeile 12 in prompts_scientific_papers.yaml)

### Kurzfristig (nächste Iteration):
2. **Post-Processing Filter** (30 Min) - Größter Impact bei geringem Aufwand
3. **PDF-Metadaten Fallback** (1-2 Std) - Nutzt bereits vorhandene Daten

### Mittelfristig (wenn Zeit):
4. **Modell-Wechsel testen** (2 Std) - Experiment mit gemma3:27b für Titel
5. **Zwei-Schritt-Extraktion** (3-4 Std) - Falls Verbesserung immer noch zu gering

### Langfristig (optional):
6. **Hybrid-Ansatz** (4-5 Std) - Maximale Präzision, aber aufwendig

## 🧪 Test-Plan

Nach Implementierung der Verbesserungen:

```bash
# 1. Erneute Titel-Extraktion mit neuem Prompt
cd /media/sz/Data/Veits_pdfs/Data_aggregation/cl-pipeline/run
pipenv run python3 run_pipeline.py -c pipelines/local_pdfs/config/full.yaml

# 2. Analyse der Verbesserung
pipenv run python3 << 'EOF'
import pandas as pd
from pathlib import Path
import difflib

base_path = Path('/media/sz/Data/Veits_pdfs/data_pipeline/local_pdfs/raw')
df_base = pd.read_pickle(base_path / 'LOCAL_files_base.p')
df_ai = pd.read_pickle(base_path / 'LOCAL_ai_meta.p')
df = df_base.merge(df_ai, on='pipe:ID')

exact = 0
partial = 0
failed = 0

for idx, row in df.iterrows():
    bib = str(row['bibtex:title']).lower().replace('{', '').replace('}', '')
    ai = str(row['ai:title']).lower()

    if bib == ai:
        exact += 1
    elif difflib.SequenceMatcher(None, bib, ai).ratio() > 0.7:
        partial += 1
    else:
        failed += 1

print(f"Exakt: {exact/len(df)*100:.1f}%")
print(f"Teilweise: {partial/len(df)*100:.1f}%")
print(f"Fehlgeschlagen: {failed/len(df)*100:.1f}%")
EOF
```

**Ziel**:
- Exakte Übereinstimmung: >30% (vorher 6.4%)
- Teilweise Übereinstimmung: >40% (vorher 16.8%)
- Fehlgeschlagen: <30% (vorher 67.4%)

## 📌 Hinweise

- Der neue Prompt ist **sofort aktiv**, aber die Daten müssen **neu extrahiert** werden (Stage 5 mit `force_run: True`)
- Für vollständige Re-Extraktion: **~8-12 Stunden** (965 Dokumente × llama3:70b)
- Alternativ: Nur **Teilmenge testen** (z.B. 50 Dokumente) mit `test.yaml` Config

---

**Erstellt**: 2025-12-07
**Status**: Prompt verbessert ✅, weitere Maßnahmen ausstehend
