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

    classDef green fill:#5bd21c
    classDef yellow fill:#ffd966
    classDef gray fill:#bcbcbc

    
    subgraph BASIC[OPAL Datenaggregation]

    OPAL[(OPAL <br> Materialien <br> & Kurse)] 

    subgraph A. OPAL API
    direction LR
    OPAL_QUERY(Abfrage <br>OER Inhalte):::green
    OPAL_QUERY --> |Ganze  Kurse| OPAL_REPOS[ignore]
    OPAL_QUERY --> |Einzelne Dateien| TYPE_FILTER[Extrahiere <br>OPAL Metadaten]:::green
    
    end
    OPAL<--> OPAL_QUERY

    subgraph FILE_AGG["B. "Dateierfassung]
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
flowchart TD

    classDef green fill:#5bd21c
    classDef yellow fill:#ffd966
    classDef gray fill:#bcbcbc
    classDef white fill:#ffffff,stroke:#ffffff

    subgraph Datenaggregation
    OPAL_FILES[(OPAL Files<br>office,pdf)]
    OPAL_METADATA_FILES[(OPAL Meta<br>office,pdf)]
    LIA_FILES_[(LiaScript<br>Files)]
    LIA_METADATA_FILES[(LiaScript<br>Metadata)]
    end

    class Materialidentifikation, gray

    subgraph Metadatenaggregation
    subgraph OPAL Pipeline
    OPAL_EXTRACTION_TYP_MD(Extraktion Datei-<br> typspezifischer Metadaten):::green
    OPAL_EXTRACTION_LLM_MD(LLM basierte<br>Extraktion Metadaten):::green
    OPAL_EXTRACTION_TYP_MD--> |file:|OPALFILES[(Metadaten<br>Sammlung<br> OPAL)]
    OPAL_EXTRACTION_LLM_MD --> |ai:|OPALFILES
    OPAL_METADATA_FILES --> |opal:|OPALFILES
    end
    subgraph LiaScript Pipeline
    LIA_EXTRACTION_TYP_MD(Extraktion markdown-<br>spezifischer Metadaten):::yellow
    LIA_EXTRACTION_LLM_MD(LLM basierte<br>Extraktion Metadaten):::yellow
    LIA_EXTRACTION_TYP_MD--> |md:|LIAFILES[(Metadaten <br>Sammlung<br>LiaScript)]
    LIA_EXTRACTION_LLM_MD --> |ai:|LIAFILES
    LIA_METADATA_FILES --> |github:|LIAFILES
    end

    subgraph Evaluation
    KREUZVERGLEICH(Kreuzvergleich Autoren):::green 
    KLASSIFIKATION(Normierung der Keywords):::yellow
    PLAUSIBILISIERUNG(Externer Check Autoren)
    BIB_KLASSIFIKATION(Bibliografische Einordung)
    end

    OPALFILES --> Evaluation
    LIAFILES --> Evaluation

    end

    OPAL_FILES --> OPAL_EXTRACTION_TYP_MD
    OPAL_FILES --> OPAL_EXTRACTION_LLM_MD
    LIA_FILES_ --> LIA_EXTRACTION_TYP_MD
    LIA_FILES_ --> LIA_EXTRACTION_LLM_MD

    Evaluation --> METADATA_Analysis(Analyse der Datensätze)
    Evaluation --> METADATA_PROPOSALS(Metadatenvorschläge<br>für Autoren)
    Evaluation --> METADATA_CLASSIFICATION(Materialklassifikation<br>für Autoren)

    class Datenaggregation white
    class Datenaggregation,Metadatenaggregation,Evaluation gray
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
