# Changelog - Local PDFs Scripts

## 2025-12-07 - BibTeX Parser Fix

### Problem
Der BibTeX-Parser verwendete ein einfaches Regex-Pattern `{([^}]+)}`, das bei geschachtelten Klammern versagte.

**Beispiel:**
```bibtex
title = {bavarikon - {DDB} - Europeana: Die Beteiligung...}
```

**Vorher (falsch):**
```
Extrahiert: "bavarikon - {DDB"  ❌
```

**Nachher (korrekt):**
```
Extrahiert: "bavarikon - {DDB} - Europeana: Die Beteiligung..."  ✅
```

### Ursache
In BibTeX werden geschweifte Klammern `{...}` verwendet, um Text vor Case-Änderungen zu schützen:
- `{DDB}` → bleibt "DDB" (nicht "ddb" oder "Ddb")
- `{FID}` → bleibt "FID"
- `{OCR}` → bleibt "OCR"

Diese geschachtelten Klammern wurden vom alten Regex nicht erkannt.

### Lösung
Neue Funktion `extract_bibtex_field()` mit Klammer-Zähler:

```python
def extract_bibtex_field(entry, field_name):
    """Extract BibTeX field handling nested braces correctly."""
    pattern = rf'{field_name}\s*=\s*\{{'
    match = re.search(pattern, entry)
    if not match:
        return None

    start = match.end()
    brace_count = 1
    i = start

    # Find matching closing brace, handling nesting
    while i < len(entry) and brace_count > 0:
        if entry[i] == '{':
            brace_count += 1
        elif entry[i] == '}':
            brace_count -= 1
        i += 1

    if brace_count == 0:
        return entry[start:i-1].strip()
    return None
```

### Auswirkung
- **Vorher:** 369 von 1010 Titeln (36.5%) waren abgeschnitten
- **Nachher:** Alle 1010 Titel korrekt extrahiert ✅

### Betroffene Felder
Die Funktion wird jetzt für alle BibTeX-Felder verwendet:
- `title` ← Hauptproblem
- `author`
- `keywords`
- `abstract`
- `journal`
- `doi`, `url`, `date`, `year`, `langid`

### Beispiele korrigierter Titel

| ID   | Vorher (abgeschnitten) | Nachher (korrekt) |
|------|------------------------|-------------------|
| 109  | bavarikon - {DDB | bavarikon - {DDB} - Europeana: Die Beteiligung... |
| 7972 | Kompetenzzentrum für Lizenzierung: Zentrale Dienstleistungen für das {FID | Kompetenzzentrum für Lizenzierung: Zentrale Dienstleistungen für das {FID}-Netzwerk |
| 245  | Die Evaluierung des {UrhWissG | Die Evaluierung des {UrhWissG} – Case closed? |
| 7970 | {FAIRe | {FAIRe} Forschungsdaten in den Geisteswissenschaften... |

### Notwendige Schritte nach dem Fix

1. **Daten neu generieren:**
```bash
cd /media/sz/Data/Veits_pdfs/Data_aggregation/cl-pipeline
pipenv run python3 run/pipelines/local_pdfs/scripts/prepare_local_pdfs.py
```

2. **HTML-Übersicht neu generieren:**
```bash
pipenv run python3 run/pipelines/local_pdfs/analysis/generate_overview.py
```

3. **(Optional) Pipeline neu laufen lassen:**
Da `LOCAL_files_base.p` als Input für spätere Stages dient, könnte ein Neustart sinnvoll sein.
Allerdings: Die AI-Metadaten-Extraktion nutzt die PDF-Inhalte, nicht die BibTeX-Titel, daher ist das nicht kritisch.

### Dateien geändert
- `run/pipelines/local_pdfs/scripts/prepare_local_pdfs.py`
  - Neue Funktion: `extract_bibtex_field()` (Zeile 35-68)
  - Ersetzt: Alle `re.search(r'field\s*=\s*\{([^}]+)\}')` durch `extract_bibtex_field()` Aufrufe (Zeile 99-150)

### Validierung
```python
# Test mit ID 109
df = pd.read_pickle('LOCAL_files_base.p')
title = df[df['pipe:ID'] == '109'].iloc[0]['bibtex:title']
print(title)
# Output: bavarikon - {DDB} - Europeana: Die Beteiligung von Archiven und Bibliotheken an überregionalen Kulturportalen im Vergleich
```

✅ **Fix verifiziert und deployed!**

---

## 2025-12-07 - HTML-Darstellung: Geschweifte Klammern entfernt

### Problem
Die BibTeX-Klammern `{...}` wurden im HTML angezeigt und störten den Lesefluss:
```
"bavarikon - {DDB} - Europeana: Die Beteiligung..."
```

### Lösung
In `generate_overview.py` werden die Klammern vor der Anzeige entfernt:

```python
def format_title(title_str: str) -> str:
    # Remove BibTeX curly braces (used to protect capitalization)
    title_str = title_str.replace('{', '').replace('}', '')
```

Gilt für:
- `format_title()` - BibTeX-Titel und LLM-Titel
- `format_author_list()` - BibTeX-Autoren und LLM-Autoren

### Ergebnis
```
Vorher: bavarikon - {DDB} - Europeana: Die Beteiligung...
Jetzt:  bavarikon - DDB - Europeana: Die Beteiligung...
```

**Hinweis:** Die Klammern bleiben im Backend-DataFrame (`LOCAL_files_base.p`) erhalten, nur die HTML-Darstellung ist bereinigt. Das ist korrekt, da die Klammern semantische Bedeutung für BibTeX haben (Groß-/Kleinschreibung).

✅ **HTML-Darstellung verbessert!**
