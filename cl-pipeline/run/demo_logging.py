#!/usr/bin/env python3
"""
Demo-Script zur Demonstration der zentralen Logging-Konfiguration

Dieses Script zeigt, wie die neue zentrale Logging-Konfiguration
verwendet werden kann und demonstriert die verschiedenen Funktionen.
"""

import sys
from pathlib import Path
import yaml

# Pfad zur src-Bibliothek hinzufügen
src_path = Path(__file__).parent.parent / 'src'
sys.path.append(str(src_path))

from logging.logger_configurator import LoggerConfigurator
from logging.stage_utils import setup_stage_logging


def demo_basic_usage():
    """Demonstriert die Grundnutzung der Logging-Konfiguration"""
    print("=== Demo: Grundnutzung ===")
    
    # Erstelle einen Standard-Konfigurator
    configurator = LoggerConfigurator()
    
    # Zeige aktuelle Status
    print("\nStandard-Konfiguration:")
    for logger_name, level in configurator.external_loggers.items():
        print(f"  {logger_name}: {level}")
    
    # Aktiviere die Konfiguration
    configurator.setup_all_loggers()
    print("\nLogging-Konfiguration aktiviert!")
    
    # Zeige aktuellen Status
    status = configurator.get_logger_status()
    print("\nAktueller Logger-Status:")
    for logger_name, level in status.items():
        print(f"  {logger_name}: {level}")


def demo_custom_config():
    """Demonstriert benutzerdefinierte Konfiguration"""
    print("\n\n=== Demo: Benutzerdefinierte Konfiguration ===")
    
    # Benutzerdefinierte Konfiguration
    custom_config = {
        'logging': {
            'configure_root_logger': True,
            'root_level': 'WARNING',
            'pipeline_level': 'DEBUG',
            'external_loggers': {
                'httpcore': 'ERROR',  # Weniger streng als Standard
                'custom_logger': 'INFO'  # Neuer Logger
            }
        }
    }
    
    # Erstelle Konfigurator mit benutzerdefinierten Einstellungen
    configurator = LoggerConfigurator.create_from_config(custom_config)
    
    print("\nBenutzerdefinierte Konfiguration:")
    for logger_name, level in configurator.external_loggers.items():
        print(f"  {logger_name}: {level}")
    
    configurator.setup_all_loggers()
    print("\nBenutzerdefinierte Logging-Konfiguration aktiviert!")


def demo_yaml_config():
    """Demonstriert Nutzung mit YAML-Konfiguration"""
    print("\n\n=== Demo: YAML-Konfiguration ===")
    
    # Lade die echte YAML-Konfiguration
    yaml_path = Path(__file__).parent / 'cl_opal.yaml'
    
    if yaml_path.exists():
        with open(yaml_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
        
        print(f"Lade Konfiguration aus: {yaml_path}")
        
        if 'logging' in config:
            print("\nLogging-Konfiguration gefunden:")
            logging_config = config['logging']
            
            # Zeige relevante Einstellungen
            print(f"  Root Logger konfigurieren: {logging_config.get('configure_root_logger', False)}")
            print(f"  Pipeline Level: {logging_config.get('pipeline_level', 'INFO')}")
            
            external_loggers = logging_config.get('external_loggers', {})
            print(f"  Externe Logger: {len(external_loggers)} konfiguriert")
            
            # Erstelle Konfigurator aus YAML
            configurator = LoggerConfigurator.create_from_config(config)
            configurator.setup_all_loggers()
            print("\nYAML-basierte Logging-Konfiguration aktiviert!")
        else:
            print("Keine Logging-Konfiguration in YAML gefunden.")
    else:
        print(f"YAML-Datei nicht gefunden: {yaml_path}")


def demo_stage_integration():
    """Demonstriert Integration in Pipeline-Stages"""
    print("\n\n=== Demo: Stage-Integration ===")
    
    # Simuliere config_global Dictionary wie in echten Stages
    fake_config_global = {
        'logging': {
            'configure_root_logger': False,
            'pipeline_level': 'INFO',
            'external_loggers': {
                'httpcore': 'CRITICAL',
                'urllib3': 'CRITICAL',
                'langchain': 'WARNING'
            }
        }
    }
    
    print("Simuliere Stage-Setup mit setup_stage_logging()...")
    
    # Nutze die Stage-Utility-Funktion
    configurator = setup_stage_logging(fake_config_global)
    
    print("Stage-Logging erfolgreich konfiguriert!")
    print("\nKonfigurierte Logger:")
    status = configurator.get_logger_status()
    for logger_name, level in list(status.items())[:5]:  # Zeige nur die ersten 5
        print(f"  {logger_name}: {level}")
    
    print(f"  ... und {len(status) - 5} weitere Logger")


if __name__ == "__main__":
    print("=== Zentrale Logging-Konfiguration Demo ===")
    print("Dieses Script demonstriert die neue zentrale Logging-Konfiguration.\n")
    
    try:
        # Führe alle Demos aus
        demo_basic_usage()
        demo_custom_config()
        demo_yaml_config()
        demo_stage_integration()
        
        print("\n\n=== Demo abgeschlossen ===")
        print("Die zentrale Logging-Konfiguration ist nun einsatzbereit!")
        print("\nNächste Schritte:")
        print("1. Integrieren Sie setup_stage_logging(config_global) in Ihre Stages")
        print("2. Entfernen Sie lokale Logging-Konfigurationen")
        print("3. Passen Sie die YAML-Konfiguration nach Bedarf an")
        
    except ImportError as e:
        print(f"Fehler beim Import: {e}")
        print("Stellen Sie sicher, dass alle erforderlichen Module installiert sind.")
    except Exception as e:
        print(f"Unerwarteter Fehler: {e}")
        print("Überprüfen Sie die Konfiguration und Pfade.")
