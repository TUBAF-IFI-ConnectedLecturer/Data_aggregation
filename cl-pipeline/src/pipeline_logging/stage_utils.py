"""
Utility-Funktionen für die Integration der zentralen Logging-Konfiguration
in die Pipeline-Stages.
"""

from pathlib import Path
from typing import Dict, Any

from .logger_configurator import LoggerConfigurator


def setup_stage_logging(config_global: Dict[str, Any]) -> LoggerConfigurator:
    """
    Convenience-Funktion zum Setup des Loggings für eine Stage.
    
    Diese Funktion kann von jeder Stage aufgerufen werden, um die zentrale
    Logging-Konfiguration zu aktivieren.
    
    Args:
        config_global: Globale Konfiguration der Pipeline
        
    Returns:
        LoggerConfigurator-Instanz für erweiterte Kontrolle
    """
    # Prüfe ob Logging-Konfiguration vorhanden ist
    if 'logging' not in config_global:
        # Fallback auf Standard-Konfiguration
        config_global['logging'] = {}
    
    configurator = LoggerConfigurator.create_from_config(config_global)
    configurator.setup_all_loggers()
    
    return configurator


def load_logging_config_from_yaml(yaml_path: Path) -> Dict[str, Any]:
    """
    Lädt die Logging-Konfiguration direkt aus einer YAML-Datei.
    
    Args:
        yaml_path: Pfad zur YAML-Konfigurationsdatei
        
    Returns:
        Logging-Konfiguration als Dictionary
    """
    try:
        import yaml
        with open(yaml_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
        return config.get('logging', {})
    except ImportError:
        print("Warning: PyYAML not available. Cannot load YAML configuration.")
        return {}
    except Exception as e:
        print(f"Error loading YAML configuration: {e}")
        return {}


def get_default_logging_setup():
    """
    Gibt die Standard-Logging-Konfiguration zurück.
    
    Returns:
        LoggerConfigurator mit Standard-Konfiguration
    """
    configurator = LoggerConfigurator()
    configurator.setup_all_loggers()
    return configurator
