# Pipeline Umsetzung

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

## Realisierung

### Schritt 1: Datenidentifikation und -aggregation

```mermaid
flowchart TD

    classDef green fill:#5bd21c
    classDef yellow fill:#ffd966
    classDef gray fill:#bcbcbc

    subgraph Datenidentifikation und -aggregation
    subgraph OPAL Pipeline  
    direction TB
    OPAL[(OPAL)] --> OPAL_QUERY(OPAL Query):::green
    OPAL_QUERY --> |Ganze  Dateien| OPAL_REPOS[?]
    OPAL_QUERY --> |Einzelne Dateien| TYPE_FILTER[Extrahiere Dateityp]:::green
    TYPE_FILTER -->  |.pdf, .pptx, ... | OPAL_DOWNLOAD[Opal Download]:::green
    TYPE_FILTER -->  |.pdf, .pptx, ... | EXTRACT_OPAL_META[Opal Metadaten]:::green
    OPAL_DOWNLOAD -->  OPAL_FILES[(OPAL Files\noffice,pdf)]
    OPAL_DOWNLOAD -.->  OPAL_METADATA_FILES[(OPAL Meta\noffice,pdf)]
    EXTRACT_OPAL_META -. opal: .->  OPAL_METADATA_FILES[(OPAL Meta\noffice,pdf)]
    end    

    subgraph LiaScript Pipeline
    direction TB
    LIA_IDENT(Liascript Suche)
    GITHUB[(Github)] --> |Github API| LIA_IDENT:::green
    LIA_REPOS --> |Dateisuche| LIA_FILES
    LIA_IDENT --> |Dateisuche| LIA_FILES[(LiaScript\nDatei Liste)]
    LIA_IDENT --> |Reposuche| LIA_REPOS[(LiaScript\nRepo Liste)]

    LIA_FILES -->  GITHUB_DOWNLOAD(Github Download):::green
    GITHUB_DOWNLOAD -->  LIA_FILES_[(LiaScript\nFiles)]
    LIA_REPOS -.-> FEATURE_EXTRACTION_LIA(Github Metadaten Query)
    FEATURE_EXTRACTION_LIA  -. github: .->  LIA_METADATA_FILES[(LiaScript\nMetadata)]
    LIA_FILES -.-> FEATURE_EXTRACTION_LIA
    FEATURE_EXTRACTION_LIA:::green
    end  
    end

    class Materialidentifikation, gray
```

