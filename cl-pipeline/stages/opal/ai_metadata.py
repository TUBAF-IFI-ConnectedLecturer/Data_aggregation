"""
AI Metadata Extraction - Modular refactored version
Original inspired by https://medium.com/@onkarmishra/using-langchain-for-question-answering-on-own-data-3af0a82789ed

This refactored version uses a clean modular architecture with specialized processors
for different types of metadata extraction.
"""

import pandas as pd
import logging
from pathlib import Path
from tqdm import tqdm
from collections import Counter
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_chroma import Chroma
import chromadb

from pipeline.taskfactory import TaskWithInputFileMonitor

# Import the new modular components
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent / 'src'))

from ai_metadata_core.utils.prompt_manager import PromptManager
from ai_metadata_core.utils.configuration_manager import ProcessingConfigManager
from ai_metadata_core.utils.llm_interface import LLMInterface
from ai_metadata_core.processors.document_processor import DocumentProcessor
from ai_metadata_core.processors.affiliation_processor import AffiliationProcessor
from ai_metadata_core.processors.keyword_extractor import KeywordExtractor
from ai_metadata_core.processors.dewey_classifier import DeweyClassifier

# Import zentrale Logging-Konfiguration
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))
from pipeline_logging import setup_stage_logging


class AIMetaDataExtraction(TaskWithInputFileMonitor):
    """
    Refactored AI Metadata Extraction with clean modular architecture.
    
    This class orchestrates the various specialized processors for different
    types of metadata extraction while keeping the main logic simple and clean.
    """
    
    def __init__(self, config_stage, config_global):
        super().__init__(config_stage, config_global)
        
        # Setup zentrale Logging-Konfiguration
        self.logger_configurator = setup_stage_logging(config_global)
        
        # Initialize configuration
        stage_param = config_stage['parameters']
        self._setup_paths_and_config(config_global, stage_param)
        
        # Initialize core components
        self._initialize_core_components(stage_param)
        
        # Initialize specialized processors
        self._initialize_processors()
    
    def _setup_paths_and_config(self, config_global, stage_param):
        """Setup file paths and basic configuration"""
        self.file_name_inputs = Path(config_global['raw_data_folder']) / stage_param['file_name_input']
        self.file_name_output = Path(config_global['raw_data_folder']) / stage_param['file_name_output']
        self.file_folder = Path(config_global['file_folder'])
        self.file_types = stage_param['file_types']
        self.processed_data_folder = config_global['processed_data_folder']
        self.chroma_file = Path(config_global['processed_data_folder']) / "chroma_db"
        self.llm_model = stage_param['model_name']
        
        # LLM configuration from config file
        self.llm_config = stage_param.get('llm_config', {})
        self.base_url = self.llm_config.get('base_url', 'http://localhost:11434')
        self.embedding_model = self.llm_config.get('embedding_model', 'jina/jina-embeddings-v2-base-de')
        self.collection_name = self.llm_config.get('collection_name', 'oer_connected_lecturer')
        self.timeout_seconds = self.llm_config.get('timeout_seconds', 240)
    
    def _initialize_core_components(self, stage_param):
        """Initialize core utility components"""
        # Prompt management
        self.prompt_manager = PromptManager(stage_param['prompts_file_name'])
        
        # Processing configuration
        processing_mode = stage_param.get('processing_mode', {})
        self.config_manager = ProcessingConfigManager(processing_mode)
        
        # LLM interface
        self.llm_interface = LLMInterface(timeout_seconds=self.timeout_seconds)
    
    def _initialize_processors(self):
        """Initialize specialized metadata processors"""
        # Document metadata processor
        self.document_processor = DocumentProcessor(self.prompt_manager, self.llm_interface)
        
        # Affiliation processor
        self.affiliation_processor = AffiliationProcessor(self.prompt_manager, self.llm_interface)
        
        # Keywords processor
        self.keyword_extractor = KeywordExtractor(self.prompt_manager, self.llm_interface)
        
        # Dewey classification processor
        dewey_file = getattr(self, 'dewey_classification_file', 'dewey_classification.txt')
        self.dewey_classifier = DeweyClassifier(self.prompt_manager, self.llm_interface, dewey_file)
    
    def _setup_logging(self):
        """Configure logging to reduce noise from dependencies"""
        # Diese Methode ist jetzt obsolet - Logging wird zentral konfiguriert
        # Behalten für Rückwärtskompatibilität, aber keine Funktion mehr
        pass
    
    def _setup_vector_store(self):
        """Setup ChromaDB vector store and embeddings"""
        embeddings = OllamaEmbeddings(
            base_url=self.base_url,
            model=self.embedding_model
        )
        
        vectorstore = Chroma(
            collection_name=self.collection_name,
            embedding_function=embeddings,
            persist_directory=str(self.chroma_file)
        )
        
        chroma_client = chromadb.PersistentClient(path=str(self.chroma_file))
        collection = chroma_client.get_or_create_collection(name=self.collection_name)
        
        return vectorstore, collection
    
    def _create_retrieval_chain(self, vectorstore, file, pages):
        """Create LangChain retrieval chain for a specific file"""
        retriever_with_filter = vectorstore.as_retriever(
            search_kwargs={"filter": {"filename": file}, "k": pages}
        )
        
        prompt = PromptTemplate.from_template(self.prompt_manager.get_system_template())
        
        return self.llm_interface.create_qa_chain(
            retriever_with_filter,
            OllamaLLM(model=self.llm_model, temperature=0),
            prompt
        )
    
    def _process_single_field(self, field_name, file, chain, collection, existing_metadata):
        """Process a single metadata field using appropriate processor"""
        should_process, process_type = self.config_manager.should_process_field(field_name, existing_metadata)
        
        if not should_process:
            return {}
        
        try:
            if field_name == 'ai:author':
                return self.document_processor.process_author(file, chain)
            elif field_name == 'ai:title':
                return self.document_processor.process_title(file, chain)
            elif field_name == 'ai:type':
                return self.document_processor.process_document_type(file, chain)
            elif field_name == 'ai:affiliation':
                return self.affiliation_processor.process_affiliation(file, chain, collection)
            elif field_name == 'ai:keywords_ext':
                return self.keyword_extractor.extract_keywords(file, chain)
            elif field_name == 'ai:keywords_gen':
                return self.keyword_extractor.generate_keywords(file, chain)
            elif field_name == 'ai:keywords_dnb':
                return self.keyword_extractor.extract_controlled_vocabulary_keywords(file, chain)
            elif field_name == 'ai:dewey':
                return self.dewey_classifier.process_dewey_classification(file, chain)
            else:
                logging.warning("Unknown field type: %s", field_name)
                return {}
        
        except Exception as e:
            logging.error("Error processing field %s for file %s: %s", field_name, file, e)
            return {}
    
    def _get_processing_fields(self):
        """Get all fields that can be processed"""
        return (
            self.config_manager.force_processing_fields + 
            self.config_manager.conditional_processing_fields
        )
    
    def _format_output(self, file_name, metadata, existing_metadata):
        """Format output for display"""
        output_lines = [f"File      : {file_name}"]
        
        # Field display mappings
        field_labels = {
            'ai:author': 'Author',
            'ai:affiliation': 'Affiliation', 
            'ai:title': 'Title',
            'ai:type': 'Type',
            'ai:keywords_ext': 'Keywords',
            'ai:keywords_gen': 'Keywords2',
            'ai:keywords_dnb': 'Keywords3',
            'ai:dewey': 'Dewey'
        }
        
        for field_name, label in field_labels.items():
            should_show, process_type = self.config_manager.should_process_field(field_name, existing_metadata)
            
            if field_name in metadata:
                value = metadata[field_name]
                if field_name == 'ai:author' and 'ai:revisedAuthor' in metadata:
                    value += f" / {metadata['ai:revisedAuthor']}"
                output_lines.append(f"{label:<12}: {value}")
        
        return "\n            ".join(output_lines)
    
    def execute_task(self):
        """Main execution method - orchestrates the entire process"""
        self._setup_logging()
        
        # Load input data
        df_files = pd.read_pickle(self.file_name_inputs)
        df_files = df_files[df_files['pipe:file_type'].isin(self.file_types)]
        
        # Load existing metadata
        if Path(self.file_name_output).exists():
            df_metadata = pd.read_pickle(self.file_name_output)
        else:
            df_metadata = pd.DataFrame()
        
        # Setup vector store
        vectorstore, collection = self._setup_vector_store()
        
        # Get file chunk counts
        name_result = collection.get()
        filenames_list = [x["filename"] for x in name_result['metadatas']]
        chunk_counter = Counter(filenames_list)
        
        # Process each file
        processing_fields = self._get_processing_fields()
        
        for _, row in tqdm(df_files.iterrows(), total=df_files.shape[0]):
            if row['pipe:file_type'] not in self.file_types:
                continue
            
            file = f"{row['pipe:ID']}.{row['pipe:file_type']}"
            pages = chunk_counter.get(file, 0)
            
            if pages == 0:
                continue
            
            # Check existing metadata
            existing_metadata = None
            if df_metadata.shape[0] > 0:
                existing_rows = df_metadata[df_metadata['pipe:ID'] == row['pipe:ID']]
                if existing_rows.shape[0] > 0:
                    existing_metadata = existing_rows.iloc[0]
            
            # Check if file should be skipped
            if self.config_manager.should_skip_file(existing_metadata):
                continue
            
            # Initialize metadata structure
            metadata = {
                'pipe:ID': row['pipe:ID'],
                'pipe:file_type': row['pipe:file_type']
            }
            
            # Copy existing metadata
            if existing_metadata is not None:
                for key in existing_metadata.index:
                    if key.startswith('ai:'):
                        metadata[key] = existing_metadata[key]
            
            # Create retrieval chain
            chain = self._create_retrieval_chain(vectorstore, file, pages)
            
            # Process each field
            needs_processing = False
            for field_name in processing_fields:
                field_result = self._process_single_field(
                    field_name, file, chain, collection, existing_metadata
                )
                if field_result:
                    metadata.update(field_result)
                    needs_processing = True
            
            # Skip if no processing was needed
            if not needs_processing:
                continue
            
            # Display results
            output = self._format_output(file, metadata, existing_metadata)
            print(f"\n{output}\n")
            
            # Update DataFrame
            if existing_metadata is not None:
                df_metadata = df_metadata[df_metadata['pipe:ID'] != row['pipe:ID']]
            
            # Ensure ai:dewey is always a list
            if 'ai:dewey' not in metadata:
                metadata['ai:dewey'] = []
            
            df_aux = pd.DataFrame([metadata])
            df_metadata = pd.concat([df_metadata, df_aux], ignore_index=True)
            df_metadata.to_pickle(self.file_name_output)
        
        # Final save
        df_metadata.reset_index(drop=True, inplace=True)
        df_metadata.to_pickle(self.file_name_output)
