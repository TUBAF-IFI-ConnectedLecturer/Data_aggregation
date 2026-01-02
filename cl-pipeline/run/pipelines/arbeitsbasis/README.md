# Arbeitsbasis Pipeline

Diese Pipeline verarbeitet eine umfangreiche Sammlung wissenschaftlicher Forschungsarbeiten (PDFs) aus 18 verschiedenen Fachzeitschriften und extrahiert umfassende Metadaten mit KI-basierten Techniken.

## Übersicht

Die Arbeitsbasis-Pipeline ist für große, heterogene Multi-Journal-Sammlungen optimiert. Sie verarbeitet 6.732 wissenschaftliche Artikel aus 18 unterschiedlichen wissenschaftlichen Zeitschriften.

### Dokument-Charakteristiken
- **Quellen**: 18 wissenschaftliche Fachzeitschriften
- **Datenmenge**: 6.732 PDFs
- **Typ**: Wissenschaftliche Artikel (Zeitschriftenartikel, Konferenzbeiträge, Forschungsberichte, etc.)
- **Struktur**: Heterogen - verschiedene Journal-Formate und Strukturen

## Unterschied zu local_pdfs Pipeline

Die `arbeitsbasis`- und `local_pdfs`-Pipelines nutzen **die gleichen 6 Verarbeitungsstufen (Stages)**, unterscheiden sich jedoch in folgenden Punkten:

### Hauptunterschiede

| Aspekt | arbeitsbasis | local_pdfs |
|--------|-------------|-----------|
| **Datenmenge** | 6.732 PDFs | 1.010 PDFs |
| **Datenstruktur** | 18 Journal-Ordner mit Unterordnern | Flache Dateisammlung |
| **BibTeX-Quellen** | 18 separate `.bib`-Dateien (eine pro Journal) | 1 zentrale `Artikelbasis.bib` |
| **Journal-Tracking** | `pipe:journal`-Feld trackt Herkunft | Keine Journal-Differenzierung |
| **Prepare-Skript** | `prepare_arbeitsbasis.py` (430 Zeilen) | `prepare_local_pdfs.py` (364 Zeilen) |
| **Verarbeitungsstrategie** | Aggressiv: `force_run: True` | Inkrementell: `force_run: False` |
| **ChromaDB** | `chroma_db_arbeitsbasis` | `chroma_db_local` |
| **Collection Name** | `arbeitsbasis_collection` | `local_pdfs_collection` |

### Technische Unterschiede im Detail

#### 1. Prepare-Skript (`prepare_arbeitsbasis.py`)

Das Prepare-Skript für arbeitsbasis ist komplexer, weil es:

- **18 Journal-Ordner verarbeitet**: Jeder Journal hat eine eigene Ordnerstruktur
- **Journalfilterung unterstützt**: `SELECTED_JOURNALS` kann bestimmte Journals auswählen
- **Multi-BibTeX-Merge**: Kombiniert 18 separate `.bib`-Dateien zu einer einzigen Datenstruktur
- **Journal-Zugehörigkeit trackt**: Fügt `pipe:journal`-Feld zu jedem Dokument hinzu (z.B. "ABI-Technik", "B.I.T.online")

Beispielcode aus `prepare_arbeitsbasis.py` (Zeile 287):
```python
'pipe:journal': journal_name,  # Track which journal this came from
```

#### 2. Konfiguration (`config/full.yaml`)

**Stage 3 (FilterFilesByContent)**:
```yaml
force_run: True  # Filtert immer neu bei Änderungen
```

**Stage 5 (AIMetaDataExtraction)**:
```yaml
force_run: True  # TEMPORÄR auf True gesetzt um neue PDFs zu verarbeiten
skip_if_any_field_exists: true  # Überspringt nur, wenn IRGENDEIN AI-Feld existiert
```

**Bedeutung**: Die arbeitsbasis-Pipeline nutzt eine **aggressive Reprocessing-Strategie**. Wenn sich Konfiguration oder Prompts ändern, werden alle Dokumente neu verarbeitet (außer solche, die bereits mindestens ein AI-Feld haben).

**Vorteil**: Stellt sicher, dass bei Verbesserungen der Prompts/Konfiguration alle Dokumente profitieren
**Nachteil**: Längere Laufzeit bei wiederholten Runs

Im Gegensatz dazu nutzt local_pdfs `force_run: False` für inkrementelle Updates.

## Pipeline Stages

Die Pipeline verwendet die gleichen 6 Stages wie local_pdfs:

### 1. Extract PDF Metadata
Extrahiert eingebettete Metadaten aus PDF-Dateien (Titel, Autor, Erstellungsdatum, etc.)

### 2. Extract Content
Extrahiert Volltext-Inhalt aus PDFs mit PyMuPDF

### 3. Filter Files by Content
Filtert Dateien mit unzureichendem oder qualitativ schlechtem Inhalt aus
- **force_run**: `True` (immer neu filtern bei Änderungen)

### 4. Generate Embeddings
Erstellt Vektor-Embeddings für RAG (Retrieval-Augmented Generation)
- **Chunk size**: 800 Zeichen
- **Chunk overlap**: 150 Zeichen
- **Embedding model**: jina/jina-embeddings-v2-base-de

### 5. AI Metadata Extraction
Extrahiert strukturierte Metadaten mit LLM (llama3.3:70b):
- **Autorennamen**: Extrahiert und validiert mit spezialisiertem LLM-basierten Name Parser
- **Titel**: Dokumententitel
- **Dokumenttyp**: Klassifikation (Artikel, Konferenzbeitrag, Thesis, etc.)
- **Affiliationen**: Institutionelle Zugehörigkeit von der Titelseite
- **Keywords**:
  - `ai:keywords_ext`: Keywords aus dem Dokument (Autor-bereitgestellt oder extrahiert)
  - `ai:keywords_gen`: Generierte beschreibende Keywords für Katalogisierung
  - `ai:keywords_dnb`: Kontrollierte Vokabular-Keywords (GND, DDC)
- **Dewey-Klassifikation**: Bis zu 3 DDC-Klassifikationen
- **Zusammenfassung**: 3-Satz-Zusammenfassung auf Deutsch

#### Verarbeitungsstrategie

**Aggressive Reprocessing-Konfiguration**:
```yaml
force_run: True  # Verarbeitet alle Dokumente neu
skip_if_any_field_exists: true  # Außer wenn bereits AI-Felder vorhanden sind
```

**Bedeutung**: Bei Konfigurationsänderungen werden alle Dokumente reprocessed, die noch keine AI-Metadaten haben.

#### Retrieval-Konfiguration
- **max_retrieval_chunks**: 10 (begrenzt Kontext um Halluzinationen zu reduzieren)
- **retrieval_strategy**: "all" (kann zu "first_and_last_pages" oder "first_pages_only" geändert werden)

### 6. GND Keyword Check
Validiert und reichert Keywords gegen GND (Deutsche Nationalbibliothek) an
- LLM-basiertes semantisches Matching mit Wikidata
- Nutzt Dokumentkontext für Disambiguierung

## Konfigurationsdateien

### `config/full.yaml`
Vollständige Pipeline-Konfiguration für alle 6.732 PDFs
- Collection: `arbeitsbasis_collection`
- ChromaDB: `chroma_db_arbeitsbasis`

## Prompts

### `prompts/prompts_scientific_papers.yaml`
Spezialisierte Prompts optimiert für wissenschaftliche Arbeiten (identisch mit local_pdfs)

## Skripte

### `scripts/prepare_arbeitsbasis.py`
Hauptskript zur Vorbereitung der Multi-Journal-Sammlung

**Besondere Features**:
- Verarbeitet 18 Journal-Ordner
- Merged 18 separate BibTeX-Dateien
- Trackt Journal-Zugehörigkeit in `pipe:journal`
- Unterstützt Journalfilterung via `SELECTED_JOURNALS`

**Datenstruktur**:
```
arbeitsbasis/
├── raw_data/
│   ├── ABI-Technik/
│   │   ├── files/
│   │   │   ├── 1234/
│   │   │   │   └── document.pdf
│   │   │   └── ...
│   │   └── ABI-Technik.bib
│   ├── BIT-online/
│   │   ├── files/
│   │   └── BIT-online.bib
│   └── ... (16 weitere Journals)
```

## Datenpfade

**Input**: `/home/crosslab/Desktop/VeitsPdfs/data_pipeline/arbeitsbasis/raw_data/`
**Output**: `/home/crosslab/Desktop/VeitsPdfs/data_pipeline/arbeitsbasis/processed/`

## Pipeline ausführen

### Full Run (6.732 PDFs)
```bash
cd /media/sz/Data/Veits_pdfs/Data_aggregation/cl-pipeline/run
python -m pipeline.run pipelines/arbeitsbasis/config/full.yaml
```

## Unterstützte Journals

Die Pipeline verarbeitet folgende 18 wissenschaftliche Zeitschriften:

1. ABI-Technik
2. B.I.T.online
3. Bibliothek Forschung und Praxis
4. Bibliotheksdienst
5. LIBREAS
6. o-bib
7. Information - Wissenschaft & Praxis
8. Informationspraxis
9. PERSPEKTIVE BIBLIOTHEK
10. VÖB-Mitteilungen
11. 027.7
12. Medien & Altern
13. Informations- und Medientechnik
14. GMS Medizin-Bibliothek-Information
15. Young Information Scientist
16. BIBLIOTHEK - Forschung und Praxis (alternative)
17. Zeitschrift für Bibliothekswesen und Bibliographie
18. Weitere Fachzeitschriften

Jeder Journal hat seine eigene `.bib`-Datei und Ordnerstruktur.

## Schlüsselfeatures

### Journal-Tracking
Jedes Dokument wird mit seinem Ursprungs-Journal gekennzeichnet:
```python
'pipe:journal': 'ABI-Technik'
```

Dies ermöglicht spätere Analysen und Filterung nach Journal-Herkunft.

### Intelligente Retrieval-Strategien
Drei Strategien zur Auswahl von Dokumentchunks:
1. **all**: Erste N Chunks (Standard, rückwärtskompatibel)
2. **first_and_last_pages**: Chunks von ersten und letzten Seiten (löst "Autor auf letzter Seite"-Problem)
3. **first_pages_only**: Nur erste 1-2 Seiten (optimal für Metadatenextraktion)

### Response-Filterung
Filtert automatisch unerwünschte LLM-Ausgaben:
- Entfernt verbale Phrasen ("Hier ist", "Basierend auf", etc.)
- Filtert kurze unhilfreiche Antworten ("Nein", "Ja", "Ja")
- Bereinigt Formatierungs-Artefakte

### Name Parsing
Spezialisierter LLM-basierter Name Parser (gemma3:27b):
- Behandelt einzelne Wortnamen (behandelt als Nachname)
- Verarbeitet zusammengesetzte Namen (von Goethe, van der Meer)
- Validiert gegen Blacklist
- Extrahiert Titel (Dr., Prof.)

## Verarbeitungsmodus

Die Pipeline nutzt einen hybriden Verarbeitungsansatz:

**Force Processing** (immer überschreiben):
- ai:affiliation
- ai:dewey
- ai:author

**Conditional Processing** (nur wenn leer):
- ai:keywords_gen
- ai:title
- ai:type
- ai:keywords_ext
- ai:keywords_dnb
- ai:summary

**Aggressive Reprocessing-Strategie**:
- `force_run: True` in Stage 5: Verarbeitet alle Dokumente neu bei Konfigurationsänderungen
- `skip_if_any_field_exists: true`: Überspringt nur Dokumente, die bereits mindestens ein AI-Feld haben

## Häufige Probleme

### Niedrige Qualität der Metadaten
**Lösung**: Sicherstellen, dass Prompts korrekt aus `prompts/prompts_scientific_papers.yaml` geladen werden

### Autor-Halluzinationen
**Lösung**:
1. `max_retrieval_chunks` erhöhen, wenn Autoreninfo über Seiten verteilt ist
2. `retrieval_strategy: "first_and_last_pages"` verwenden, wenn Autoren am Ende erscheinen

### Spracherkennung-Fehler
**Prüfen**: `pipe:language`-Feld ist während Metadatenextraktion-Stage gesetzt

## Logging

### Log-Datei-Speicherort

Pipeline-Logs werden automatisch gespeichert in:
- **Full pipeline**: `pipelines/arbeitsbasis/logs/full_<timestamp>.log`

Jeder Pipeline-Run erstellt eine neue timestamped Log-Datei für einfaches Tracking.

### Log-Rotation

- **Max file size**: 10 MB
- **Backup files**: 5 (behält letzte 5 Rotationen)
- Wenn ein Log 10 MB erreicht, wird es rotiert zu `<filename>.log.1`, `<filename>.log.2`, etc.

### Logs anzeigen

```bash
# Neuesten Full Run anzeigen
ls -t pipelines/arbeitsbasis/logs/full_*.log | head -1 | xargs tail -f

# Alle Logs anzeigen
cat pipelines/arbeitsbasis/logs/full_*.log

# Nach Fehlern filtern
grep ERROR pipelines/arbeitsbasis/logs/full_*.log

# Nach spezifischer Stage filtern
grep "AIMetaDataExtraction" pipelines/arbeitsbasis/logs/full_*.log
```

## Vergleich mit local_pdfs Pipeline

### Wann welche Pipeline verwenden?

**Verwende arbeitsbasis, wenn**:
- Du eine große, heterogene Sammlung aus mehreren Quellen hast
- Du Journal-Herkunft tracken musst
- Du bereit bist, längere Laufzeiten für vollständiges Reprocessing zu akzeptieren
- Du separate BibTeX-Dateien pro Quelle hast

**Verwende local_pdfs, wenn**:
- Du eine kleinere, homogene Sammlung hast
- Du schnelle, inkrementelle Updates bevorzugst
- Du zusätzliche Post-Processing-Tools brauchst (Title-Improvement-Scripts)
- Du eine zentrale BibTeX-Datei hast

### Gemeinsame Features

Beide Pipelines teilen:
- Identische 6 Verarbeitungsstufen (Stages)
- Gleiche KI-Modelle (llama3.3:70b, gemma3:27b)
- Gleiches Embedding-Model (jina-embeddings-v2-base-de)
- Identische Prompt-Struktur für wissenschaftliche Arbeiten
- ChromaDB für Vektor-Speicherung
- RAG-basierte Metadatenextraktion

### Unterschiedliche Features

| Feature | arbeitsbasis | local_pdfs |
|---------|-------------|-----------|
| **Prepare-Skript** | Multi-Journal-Processing mit 430 Zeilen | Single-Collection-Processing mit 364 Zeilen |
| **Journal-Tracking** | ✅ `pipe:journal`-Feld | ❌ Nicht vorhanden |
| **Force Run (Stage 5)** | ✅ `True` (aggressiv) | ❌ `False` (inkrementell) |
| **Zusatztools** | ❌ Keine | ✅ 6 Title-Improvement & Export-Scripts |
| **BibTeX-Export** | ❌ Nicht verfügbar | ✅ `bibtex_export.py` |

## Abhängigkeiten

- LangChain
- Ollama (llama3.3:70b, gemma3:27b)
- ChromaDB
- PyMuPDF
- Jina Embeddings (jina-embeddings-v2-base-de)
