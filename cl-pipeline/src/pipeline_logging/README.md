# Zentrale Logging-Konfiguration

## Übersicht

Die zentrale Logging-Konfiguration wurde implementiert, um die über verschiedene Pipeline-Stages verteilten Logging-Einstellungen zu konsolidieren und konfigurierbar zu machen.

## Struktur

```
src/pipeline_logging/
├── __init__.py                 # Paket-Initialisierung
├── logger_configurator.py     # Hauptklasse für Logging-Konfiguration
└── stage_utils.py             # Utility-Funktionen für Stage-Integration
```

**Hinweis**: Das Modul heißt `pipeline_logging` (nicht `logging`) um Konflikte mit dem Python built-in `logging` Modul zu vermeiden.

## Konfiguration in cl_opal.yaml

```yaml
# Global Logging Configuration
logging:
  configure_root_logger: false  # Ob der Root-Logger konfiguriert werden soll
  root_level: "ERROR"          # Level für Root-Logger (falls aktiviert)
  pipeline_level: "INFO"       # Level für Pipeline-eigene Logger
  
  # Externe Logger-Konfiguration (überschreibt Standardwerte)
  external_loggers:
    # HTTP und Netzwerk - auf CRITICAL für weniger Noise
    httpcore: "CRITICAL"
    "httpcore.http11": "CRITICAL"
    httpx: "CRITICAL"
    urllib3: "CRITICAL"
    "urllib3.connectionpool": "CRITICAL"
    requests: "CRITICAL"
    
    # LangChain und AI - auf WARNING für wichtige Meldungen
    langchain: "WARNING"
    langchain_ollama: "WARNING"
    "langchain_text_splitters.base": "ERROR"
    ollama: "WARNING"
    
    # Bildverarbeitung - auf WARNING
    PIL: "WARNING"
    "PIL.PngImagePlugin": "WARNING"
    
    # Unstructured - auf CRITICAL für weniger Noise
    "unstructured.trace": "CRITICAL"
```

## Integration in Stages

### Vorher (verteilt über alle Stages)
```python
# In ai_metadata.py
logging.getLogger("httpcore.http11").setLevel(logging.CRITICAL)
logging.getLogger("langchain_text_splitters.base").setLevel(logging.ERROR)
# ... viele weitere

# In ai_embeddings.py  
logging.getLogger("httpcore.http11").setLevel(logging.CRITICAL)
logging.getLogger("urllib3.connectionpool").setLevel(logging.CRITICAL)
# ... viele weitere

# In metadata.py
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
# ... viele weitere
```

### Nachher (zentral konfiguriert)
```python
# Import hinzufügen
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))
from pipeline_logging import setup_stage_logging

class YourStage(TaskWithInputFileMonitor):
    def __init__(self, config_stage, config_global):
        super().__init__(config_stage, config_global)
        
        # Setup zentrale Logging-Konfiguration
        self.logger_configurator = setup_stage_logging(config_global)
        
        # Rest der Initialisierung...
    
    def execute_task(self):
        # Keine lokalen Logging-Konfigurationen mehr nötig!
        # Alle Logger sind bereits zentral konfiguriert
        pass
```

## Vorteile

### 1. Konsistenz
- Alle Stages verwenden dieselbe Logging-Konfiguration
- Keine Widersprüche zwischen verschiedenen Stages

### 2. Konfigurierbarkeit
- Zentrale Konfiguration über cl_opal.yaml
- Keine Code-Änderungen nötig für Anpassungen
- Verschiedene Level für verschiedene Umgebungen

### 3. Wartbarkeit
- Ein zentraler Ort für alle Logging-Einstellungen
- Einfaches Hinzufügen neuer Logger
- Übersichtliche Struktur

### 4. Flexibilität
- Standard-Konfiguration mit sinnvollen Defaults
- Überschreibung einzelner Logger möglich
- Programmatische Konfiguration weiterhin möglich

## Migrierte Stages

Die folgenden Stages wurden auf die zentrale Logging-Konfiguration umgestellt:

1. **stages/opal/ai_metadata.py** ✅
   - `_setup_logging()` Methode obsolet gemacht
   - Zentrale Konfiguration im `__init__`

2. **stages/opal/ai_embeddings.py** ✅
   - Lokale HTTP-Logger-Konfiguration entfernt
   - Zentrale Integration

3. **stages/opal/metadata.py** ✅
   - Alle lokalen `logging.getLogger().setLevel()` entfernt
   - Zentrale Integration

4. **stages/opal/keywordCheck.py** ✅
   - urllib3-Konfiguration entfernt
   - Zentrale Integration

5. **stages/opal/ai_similarity.py** ✅
   - urllib3.propagate-Konfiguration entfernt
   - Zentrale Integration

6. **stages/general/extractFileContent.py** ✅
   - Root-Logger-Konfiguration entfernt
   - Zentrale Integration

7. **stages/opal/downloadraw.py** ✅
   - Mehrere HTTP-Logger-Konfigurationen entfernt
   - Zentrale Integration

## Verwendung

### Einfache Nutzung in Stages
```python
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))
from pipeline_logging import setup_stage_logging

# In der __init__ Methode Ihrer Stage
self.logger_configurator = setup_stage_logging(config_global)
```

### Erweiterte Nutzung
```python
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))
from pipeline_logging import LoggerConfigurator

# Benutzerdefinierte Konfiguration
custom_config = {
    'logging': {
        'external_loggers': {
            'my_custom_logger': 'DEBUG'
        }
    }
}

configurator = LoggerConfigurator.create_from_config(custom_config)
configurator.setup_all_loggers()

# Status abfragen
status = configurator.get_logger_status()
print(f"Konfigurierte Logger: {len(status)}")
```

## Demonstration

Ein Demo-Script ist verfügbar unter `run/demo_logging.py`, das die verschiedenen Funktionen demonstriert:

```bash
cd cl-pipeline/run
python demo_logging.py
```

## Anpassung der Konfiguration

Zur Anpassung der Logging-Levels bearbeiten Sie die `cl_opal.yaml`:

```yaml
logging:
  external_loggers:
    # Für mehr Debugging-Information
    httpcore: "INFO"        # statt "CRITICAL"
    langchain: "DEBUG"      # statt "WARNING"
    
    # Neue Logger hinzufügen
    my_custom_library: "ERROR"
```

## Kompatibilität

- Die Änderungen sind vollständig rückwärtskompatibel
- Bestehende lokale Logging-Konfigurationen wurden ersetzt, nicht entfernt
- Fallback auf Standard-Konfiguration falls keine YAML-Konfiguration vorhanden ist
