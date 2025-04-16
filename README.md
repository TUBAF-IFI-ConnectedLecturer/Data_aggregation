# Pipeline Umsetzung

## Zielstellung

+ automatisiertes "Auffüllen" von Metadaten für existierende OPAL Datensätze
+ Unterstützung bei der Eingabe von Metadaten während der Eingabe durch den Lehrenden
+ Identifikation von ähnlichen Inhalten

## Konzept

Die Umsetzung der einzelnen Aggregationsstufen baut auf der im Ordner `pipeline` vorgesehenen [Implemetierung](https://github.com/TUBAF-IFI-ConnectedLecturer/Data_aggregation/blob/main/pipeline/README.md) auf. 
Diese sind durch die json Konfigurationsfiles, die unter `cl_pipeline\run` für LiaScript und OPAL definiert.

Die Konfigurationsfiles bestehen aus 3 Teilen: Der erste Abschnitt legt globale Parameter zur Datenstruktur fest, die über Variablen (z.B. `RAW`) im weiteren genutzt werden können. 
Der zweite Abschnitt beschreibt die für die Pipeline bereitstehenden Implementierungen der "Stages", die von den pipeline-Klassen abgeleitet werden. 
Den dritten Abschnitt bilden dann die eigentlichen Stages oder Pipelinestufen, die wiederum lokale Variablen umfassen.


```yaml
# Defining data folder stucture
folder_structure:
  #data_root_folder: &BASE ./data
  data_root_folder: &BASE /mnt/9cd5c6a1-07f3-4580-be34-8d8dd9d6fe6d/Connected_Lecturers/Opal
  raw_data_folder: &RAW !join [*BASE, /raw]
  file_folder: &FILE_FOLDER !join [*RAW, /files]
  processed_data_folder: &PREPROCESSED !join [*BASE, /processed]

# Referencing the modules containing modules
stages_module_path:
    - ../src/general/
    - ../src/opal/

# Defining stages and parameters
stages:
  - name: Generate data folder structure
    class: ProvideDataFolders

  - name: Collect raw data from OPAL 
    class: CollectOPALOERdocuments
    parameters:
      repo_file_name: OPAL_repos.p
      file_file_name: &OPALSINGLEFILES OPAL_files.p
      json_file_name: OPAL_raw_data.json
      json_url: https://bildungsportal.sachsen.de/opal/oer/content.json
      overwrite_json: False
...
```

## Verabeitungskette und Pipelinestufen

### Schritt 1: OER Material-Aggregation und Vorverarbeitung

```mermaid
flowchart TD
%%{init:{'flowchart':{'nodeSpacing': 10, 'rankSpacing': 25}}}%%

    classDef green fill:#5bd21c
    classDef yellow fill:#ffd966
    classDef gray fill:#bcbcbc

    
    subgraph BASIC[A.&nbsp;Material‑Identfikation&nbsp;und&nbsp;Aggregations‑Phase]

    OPAL[(OPAL <br> Materialien <br> & Kurse)] 

    subgraph A. Materialidentifikation
    direction LR
    OPAL_QUERY(Abfrage <br>OER Inhalte):::green
    OPAL_QUERY --> |Ganze  Kurse| OPAL_REPOS[ignore]
    OPAL_QUERY --> |Einzelne Dateien| TYPE_FILTER[Extrahiere <br>OPAL Metadaten]:::green
    
    end
    OPAL<--> OPAL_QUERY

    subgraph FILE_AGG["B. "Datenerfassung]
    direction LR
    FILE_DOWNLOAD[Datei<br>Download]:::green --> TEXT_EXTRAKTION[Text-<br>extraktion]:::green --> TEXT_ANALYSIS[Textbasis-<br>analyse]:::green
    FILE_DOWNLOAD --> FILE_METADATA_EXTRACTION[Metadaten<br>extraktion]:::green
    end

    TYPE_FILTER -->  |.pdf, .pptx, ... | FILE_DOWNLOAD

    OPAL_FILES[(Material-<br>dateien)]
    FILE_DOWNLOAD --> OPAL_FILES

    FILE_METADATA[(Datei<br>Metadaten)]
    FILE_METADATA_EXTRACTION --> FILE_METADATA

    OPAL_METADATA[(OPAL<br>Metadaten)]
    TYPE_FILTER --> |.pdf, .pptx, ... | OPAL_METADATA

    OPAL_CONTENT[(Datei<br>Textinhalt)]
    TEXT_EXTRAKTION --> OPAL_CONTENT

    CONTENT_METADATA[(Inhalt<br> Metadaten)]
    TEXT_ANALYSIS --> CONTENT_METADATA

    subgraph FUSION["C. "Fusion & Validierung]
    direction TB
    ABC[Autoren-<br>name]:::yellow
    CDE[Sprache<br>.]:::yellow
    EFG[Erstellungs-<br>datum]:::yellow

    end

    subgraph FILTER["D. Filterung <a href='http://google.com'>[Link]</a>"]
    direction TB
    Sprache:::green
    Textlängen:::green
    Dublikate:::green
    end

    CONTENT_METADATA --> FILTER
    FILE_METADATA --> FUSION
    OPAL_METADATA --> FUSION
    OPAL_CONTENT --> FILTER

    FUSION --> FILTER

    end

    class BASIC, gray
```


## Schritt 2: Metadatenextraktion und -generierung

```mermaid
flowchart 
%%{init:{'flowchart':{'nodeSpacing': 25, 'rankSpacing': 15}}}%%

    classDef green fill:#5bd21c
    classDef yellow fill:#ffd966
    classDef gray fill:#bcbcbc
    classDef white fill:#ffffff,stroke:#ffffff
    
    subgraph BASIC[KI&nbsp;basierte&nbsp;Extraktion&nbsp;der&nbsp;Metadaten]
    OPAL_CONTENT[(Opal<br>Text Inhalte)]
    OPAL_EMBEDDINGS(Embeddings Generation):::green
    OPAL_CONTENT --> OPAL_EMBEDDINGS

    subgraph RAG ["A.&nbsp;Retrieval‑Augmented&nbsp;Generation"]
    VECTOR_DB[(Vektor<br> Datenbank)]
    OPAL_EMBEDDINGS --> VECTOR_DB
    PROMPTS@{ shape: doc, label: "Prompts für <br> Titel<br> Keywords <br> ..." }
    LLM(Lokales LLM):::green
    VECTOR_DB --> LLM
    PROMPTS --> LLM
    end

    subgraph GND ["B. "GND Check]
    AI_METADATA[(AI generierte<br>Metadata)]
    LLM --> AI_METADATA
    GND_CHECK(GND Keyword Check):::green
    end

    subgraph SIMILARITY ["C. "Ähnlichkeitsanalyse]
    KEYWORD_SIM(Keyword basiert):::green
    EMBEDDING_SIM(Embedding basiert):::green
    MINHASH_SIM(MinHash basiert):::green
    RESULT@{ shape: documents, label: "Ähnlichkeitsmatrizen" }
    MINHASH_SIM --> RESULT
    EMBEDDING_SIM--> RESULT
    KEYWORD_SIM--> RESULT
    end
    AI_METADATA <--> GND_CHECK
    OPAL_CONTENT --> MINHASH_SIM
    VECTOR_DB --> EMBEDDING_SIM
    AI_METADATA -->KEYWORD_SIM
    end
    class BASIC, gray


    class Metadatenaggregation,Evaluation gray
```

## Generelle Installation 

+ Installation von `pipenv` als virtuelle Entwicklungsumgebung
+ Ausführen von `pipenv install` im Projektordner


### OPAL

*Vorbereitung*

+ Prüfen der Angaben der Ordnerstruktur und den internen Nutzern in `identification_opal.yaml`

*Ausführung*

``` 
pipenv shell
(pipenv) cd run
(pipenv) python run_pipeline.py -c identification_opal.yaml
```

### LiaScript 

*Vorbereitung*

+ github Account anlegen und in `.env` hinterlegen
+ Prüfen der Angaben der Ordnerstruktur und den internen Nutzern in `identification_liascript.yaml`

*Ausführung*

``` 
pipenv shell
(pipenv) cd run
(pipenv) python run_pipeline.py -c identification_liascript.yaml
```

> Wegen der API Limitierung von Github, kann es sein, dass das Auslesen der Datensätze von Github mehrere Tage dauern!
