# Connected Lecturers Pipeline - Datenaggregation

## Metadaten der OPAL Dateien

Die Erhebung der Metadaten für die OPAL Dateien wird auf drei Ebenen durchgeführt.

| Ebene | Quelle                                                        | Kürzel im Datensatz |
| ----- | ------------------------------------------------------------- | ------------------- |
| 1     | originäre Metainformationen im JSON Datensatz der UB Freiberg | `opal`              |
| 2     | Metadaten/Properties in den Dateien                           | `file`              |
| 3     | mit KI extrahierte Metadaten aus den Inhalten der Dateien     | `ai`                |

Letztendlich entsteht ein Datensatz, der die Metadaten aus allen drei Ebenen vereint und auf ein übergreifendes Schema abbildet [OER-Metadaten](https://dini-ag-kim.github.io/hs-oer-lom-profil/latest/).

## Pipeline Metainformationen für jeden Datensatz  

| Bedeutung          | CL-Naming             |
| ------------------ | --------------------- |
| ID                 | `pipe:ID`             |
| Ordner             | `pipe:file_path`      |
| Dateitype          | `pipe:file_type`      |
| Language           | `pipe:language`       |
| Error              | `pipe:error_download` |
| Datum des Download | `pipe:download_date`  |

## Originäre Metainformationen

| OPAL Label         | OPAL Bedeutung         | CL-Naming               |
| ------------------ | ---------------------- | ----------------------- |
| `filename`         | Dateiname              | `opal:filename`         |
| `license`          | Lizenz                 | `opal:license`          |
| `oer_permalink`    |                        | `opal:oer_permalink`    |
| `title`            | Titel                  | `opal:title`            |
| `comment`          | Beschreibung           | `opal:comment`          |
| `creator`          | Autor                  | `opal:creator`          |
| `publisher`        | Herausgeber            |                         |
| `source`           | Quelle                 |                         |
| `city`             | Ort                    |                         |
| `publicationMonth` | Veröffentlichungsmonat | `opal:publicationMonth` |
| `publicationYear`  | Veröffentlichungsjahr  | `opal:publicationYear`  |
| `pages`            | Seiten                 |                         |
| `language`         | Sprache                | `opal:language`         |
| `url`              | Verlinkung / URL       |                         |
| `act`              | Werk                   |                         |
| `appId`            | Projekt                |                         |
| `category`         | Kategorie              |                         |
| `chapter`          | Kapitel                |                         |
| `duration`         | Dauer                  |                         |
| `mediaType`        | Medientyp              |                         |
| `nav1`             | ?                      |                         |
| `nav2`             | ?                      |                         |
| `nav3`             | ?                      |                         |
| `series`           | Reihe                  |                         |

## Metadaten in den Dateien

Aus den Dateien der Typen `docx`, `pptx`, `xlsx` und `pdf` werden Metadaten extrahiert. 

| Office Dateien   | pdf Dateien    | CL-Naming       |
| ---------------- | -------------- | --------------- |
| `creator`        | `author`       | `file:author`   |
| `title`          | `title`        | `file:title`    |
| `description`    |                |                 |
| `subject`        | `subject`      | `file:subject`  |
| `identifier`     |                |                 |
| `language`       | `language`     | `file:language` |
| `created`        | `creationDate` | `file:created`  |
| `modified`       | `modDate`      | `file:modified` |
| `lastModifiedBy` |                |                 |
| `category`       |                |                 |
| `contentStatus`  |                |                 |
| `version`        |                |                 |
| `revision`       |                |                 |
| `keywords`       |                | `file:keywords` |
| `lastPrinted`    |                |                 |
|                  | `creator`      |                 |
|                  | `producer`     |                 |
|                  | `format`       |                 |


## Metadaten extrahiert mit KI

| CL-Naming     | Promt                                                                                                         |
| ------------- | ------------------------------------------------------------------------------------------------------------- |
| `ai:title`    | `f"Give me a title of the document {file}. Just answer by the title. Please answer in German."`               |
| `ai:author`   | `f"Who is the author of the document {file}. Avoid all additional information, just answer by authors name."` |
| `ai:keywords` | `f"Please extract 5 Keywords from {file}? Just answer by a list separted by commas. Please answer in German.` |

## OER Metadaten Schema

Die Struktur aus [LOM for Higher Education OER Repositories](https://dini-ag-kim.github.io/hs-oer-lom-profil/latest/) wurde hier eingeebnet. Zur Projektlaufzeit wird das finale Schema festgelegt und ein Transformationsskript integriert.

| Benennung            | Bemerkungen                                     | `pipe:`          | `opal:`              | `file:`         | `ai:`         |
| -------------------- | ----------------------------------------------- | ---------------- | -------------------- | --------------- | ------------- |
| `<title>`            |                                                 |                  | `opal:title`         | `file:title`    | `ai:title`    |
| `<language>`         |                                                 | `pipe:language`  | `opal:language`      | `file:language` |               |
| `<description>`      |                                                 |                  | `opal:comment`       | `file:subject`  |               |
| `<keyword>`          |                                                 |                  |                      | `file:keywords` | `ai:keywords` |
| `<aggregationlevel>` | für einzelne, atomare Materialien  (1)          | 1                |                      |                 |               |
| `<format>`           | z. B. application/pdf oder image/png.           | `pipe:file_type` |                      |                 |               |
| `<location>`         | in der Regel ein Uniform Resource Locator (URL) |                  | `opal:oer_permalink` |                 |               |
| `<rights>`           | Lizenzparameter                                 |                  | `opal:license`       |                 |               |
| `<author>`           |                                                 |                  | `opal:creator`       | `file:author`   | `ai:author`   |
| `<date>`             |                                                 |                  |                      | `file:modified` |               |





