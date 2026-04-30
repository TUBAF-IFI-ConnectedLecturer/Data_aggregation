# Analysis Scripts - Local PDFs Collection

Dieses Verzeichnis enthält Analyse-Tools für die Auswertung der verarbeiteten PDF-Metadaten.

## 📊 HTML-Übersicht generieren

### Script: `generate_overview.py`

Erstellt eine interaktive HTML-Tabelle mit allen verarbeiteten Dokumenten.

#### Verwendung

```bash
cd cl-pipeline
pipenv run python3 run/pipelines/local_pdfs/analysis/generate_overview.py
```

#### Output

Die HTML-Datei wird hier erstellt:
```
/media/sz/Data/Veits_pdfs/data/raw_data/metadata_overview.html
```

#### Was wird angezeigt?

Die HTML-Tabelle enthält für jedes Dokument:

| Spalte | Beschreibung | Quelle |
|--------|--------------|--------|
| **ID** | Dokument-ID | Pipeline |
| **BibTeX Autor** | Autoren aus .bib-Datei | BibTeX-Import |
| **BibTeX Titel** | Titel aus .bib-Datei | BibTeX-Import |
| **LLM Autor** | Von LLM extrahierte Autoren | AI-Metadatenextraktion (Stage 5) |
| **LLM Titel** | Von LLM extrahierter Titel | AI-Metadatenextraktion (Stage 5) |
| **GND-Keywords** | Von Lobid validierte Schlagwörter | GND-Keyword-Check (Stage 6) |
| **PDF** | Link zum Original-PDF | Relativ-Pfad |

#### Features

- ✅ **Interaktive Tabelle** mit Sticky Header
- ✅ **Farbcodierung**:
  - 🟨 Gelb = BibTeX-Metadaten
  - 🟦 Blau = LLM-extrahierte Metadaten
  - 🟩 Grün = GND-validierte Keywords
- ✅ **Klickbare PDF-Links** (funktionieren wenn HTML in `/data/raw_data` liegt)
- ✅ **Responsive Design** für verschiedene Bildschirmgrößen
- ✅ **Keyword-Badges** mit Anzahl-Anzeige

#### Beispiel-Output

```html
ID    | BibTeX Autor      | BibTeX Titel        | LLM Autor         | ...
------|-------------------|---------------------|-------------------|----
7972  | Hillenkötter, K.  | Kompetenzzentrum... | Heinrich, Indra   | ...
```

#### Relative Pfade

Die PDF-Links verwenden relative Pfade von `/data/raw_data` zu `/data_pipeline/`:

```
/media/sz/Data/Veits_pdfs/
├── data/
│   └── raw_data/
│       └── metadata_overview.html  ← HTML-Datei hier
└── data_pipeline/
    └── local_pdfs/
        └── raw/
            └── files/
                └── 7972.pdf         ← PDF-Dateien hier
```

**Relativer Pfad:** `../../data_pipeline/local_pdfs/raw/files/7972.pdf`

## 📋 Datenquellen

Das Script liest folgende Pickle-Dateien:

```
/media/sz/Data/Veits_pdfs/data_pipeline/local_pdfs/raw/
├── LOCAL_files_base.p         # BibTeX-Metadaten + Dateipfade
├── LOCAL_ai_meta.p            # LLM-extrahierte Metadaten
└── LOCAL_checked_keywords.p   # GND-validierte Keywords
```

## 🔧 Anpassungen

### Keyword-Anzahl ändern

In `generate_overview.py`, Zeile ~70:

```python
def format_keywords(keywords: List[str], max_display: int = 10) -> str:
    # Ändere max_display auf gewünschte Anzahl
```

### Zusätzliche Spalten hinzufügen

Verfügbare Felder in den DataFrames:

**BibTeX (`df_base`):**
- `bibtex:author`, `bibtex:title`, `bibtex:year`, `bibtex:doi`
- `bibtex:journal`, `bibtex:abstract`, `bibtex:keywords`

**AI-Metadaten (`df_ai`):**
- `ai:author`, `ai:title`, `ai:affiliation`, `ai:type`
- `ai:keywords_ext`, `ai:keywords_gen`, `ai:keywords_dnb`
- `ai:summary`, `ai:dewey`

**Keywords (`df_keywords`):**
- `keywords` (Liste von Dicts mit `gnd_preferred_name`, `gnd_link`, `is_gnd`, etc.)

## 🎨 Styling anpassen

CSS befindet sich im `<style>`-Block des generierten HTML (Zeile ~115-180).

Beispiel für Farbanpassung:

```css
.bibtex-col {
    background-color: #fff3cd;  /* Gelb → andere Farbe */
}
```

## 🐛 Troubleshooting

### Fehler: "ModuleNotFoundError: No module named 'checkAuthorNames'"

**Ursache:** Script muss vom Pipeline-Root ausgeführt werden.

**Lösung:**
```bash
cd cl-pipeline
pipenv run python3 run/pipelines/local_pdfs/analysis/generate_overview.py
```

### PDF-Links funktionieren nicht

**Ursache:** HTML-Datei nicht in `/data/raw_data`.

**Lösung:** HTML automatisch dort generiert. Falls verschoben, Pfade in Script anpassen (Zeile ~88).

### Leere Keywords

**Ursache:** Stage 6 (GND-Check) noch nicht ausgeführt.

**Lösung:**
```bash
cd run
pipenv run python3 run_pipeline.py -c pipelines/local_pdfs/config/full.yaml
```

## 📞 Support

Bei Fragen oder Problemen wenden Sie sich an das Pipeline-Team.

---

**Erstellt:** 2025-12-07
**Version:** 1.0
**Pipeline:** local_pdfs
