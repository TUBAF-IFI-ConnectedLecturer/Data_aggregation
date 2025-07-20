"""
Zentrale Logging-Konfiguration für die Pipeline

Diese Klasse übernimmt die Konfiguration aller Logger, die in den verschiedenen 
Pipeline-Stages verwendet werden, um eine konsistente und konfigurierbare
Logging-Konfiguration zu gewährleisten.
"""

import logging
from typing import Dict, Any, Optional


class LoggerConfigurator:
    """
    Zentrale Klasse für die Konfiguration aller Logger in der Pipeline.
    
    Diese Klasse sammelt alle verstreuten Logging-Konfigurationen und
    stellt eine einheitliche Schnittstelle für deren Verwaltung bereit.
    """
    
    # Standard-Konfiguration für externe Bibliotheken
    DEFAULT_EXTERNAL_LOGGERS = {
        # HTTP und Netzwerk
        "httpcore": "CRITICAL",
        "httpcore.http11": "CRITICAL", 
        "httpx": "CRITICAL",
        "urllib3": "CRITICAL",
        "urllib3.connectionpool": "CRITICAL",
        "requests": "CRITICAL",
        
        # LangChain und AI
        "langchain": "WARNING",
        "langchain_ollama": "WARNING", 
        "langchain_text_splitters.base": "ERROR",
        "ollama": "WARNING",
        
        # Bildverarbeitung
        "PIL": "WARNING",
        "PIL.PngImagePlugin": "WARNING",
        
        # Sonstige
        "unstructured.trace": "CRITICAL"
    }
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialisiert den Logger-Konfigurator.
        
        Args:
            config: Optional - Benutzerdefinierte Logging-Konfiguration
        """
        self.config = config or {}
        self.external_loggers = self._merge_external_loggers()
        
    def _merge_external_loggers(self) -> Dict[str, str]:
        """
        Kombiniert Standard- und benutzerdefinierte externe Logger-Konfigurationen.
        
        Returns:
            Dict mit Logger-Namen und Level-Strings
        """
        merged = self.DEFAULT_EXTERNAL_LOGGERS.copy()
        
        # Überschreibe mit benutzerdefinierten Konfigurationen
        if 'external_loggers' in self.config:
            merged.update(self.config['external_loggers'])
            
        return merged
    
    def setup_external_loggers(self):
        """
        Konfiguriert alle externen Logger basierend auf der Konfiguration.
        """
        for logger_name, level_str in self.external_loggers.items():
            logger = logging.getLogger(logger_name)
            level = getattr(logging, level_str.upper())
            
            # Für urllib3 spezielle Behandlung
            if logger_name == "urllib3":
                logger.propagate = False
                
            logger.setLevel(level)
    
    def setup_pipeline_logging(self, level: str = "INFO"):
        """
        Konfiguriert das Haupt-Pipeline-Logging.
        
        Args:
            level: Logging-Level für die Pipeline (default: INFO)
        """
        pipeline_level = self.config.get('pipeline_level', level)
        
        # Root-Logger konfigurieren falls gewünscht
        if self.config.get('configure_root_logger', False):
            root_level = self.config.get('root_level', 'ERROR')
            logging.getLogger().setLevel(getattr(logging, root_level.upper()))
    
    def setup_all_loggers(self):
        """
        Führt die komplette Logging-Konfiguration durch.
        """
        self.setup_external_loggers()
        self.setup_pipeline_logging()
    
    def get_logger_status(self) -> Dict[str, str]:
        """
        Gibt den aktuellen Status aller konfigurierten Logger zurück.
        
        Returns:
            Dict mit Logger-Namen und aktuellen Levels
        """
        status = {}
        for logger_name in self.external_loggers.keys():
            logger = logging.getLogger(logger_name)
            status[logger_name] = logging.getLevelName(logger.getEffectiveLevel())
        return status
    
    @staticmethod
    def create_from_config(config_dict: Dict[str, Any]) -> 'LoggerConfigurator':
        """
        Factory-Methode zum Erstellen eines Konfigurators aus einer Konfiguration.
        
        Args:
            config_dict: Konfigurationsdictionary
            
        Returns:
            LoggerConfigurator-Instanz
        """
        logging_config = config_dict.get('logging', {})
        return LoggerConfigurator(logging_config)


def setup_logging_from_config(config_dict: Dict[str, Any]):
    """
    Convenience-Funktion zum direkten Setup des Loggings aus einer Konfiguration.
    
    Args:
        config_dict: Konfigurationsdictionary mit 'logging'-Sektion
    """
    configurator = LoggerConfigurator.create_from_config(config_dict)
    configurator.setup_all_loggers()
    
    return configurator
