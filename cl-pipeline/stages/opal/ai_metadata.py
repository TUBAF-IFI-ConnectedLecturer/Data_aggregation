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
from ai_metadata_core.processors.summary_processor import SummaryProcessor

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

        # Optional: Limit number of chunks retrieved for LLM context
        # Default: None (use all available chunks for backwards compatibility)
        self.max_retrieval_chunks = stage_param.get('max_retrieval_chunks', None)

        # Optional: Retrieval strategy for selecting relevant chunks
        # Options: "all" (default), "first_and_last_pages", "first_pages_only"
        self.retrieval_strategy = stage_param.get('retrieval_strategy', 'all')
    
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
        
        # Summary processor
        self.summary_processor = SummaryProcessor(self.prompt_manager, self.llm_interface)
    
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

    def _apply_retrieval_strategy(self, file, available_chunks, field_name=None):
        """
        Apply retrieval strategy to select which chunks to use.

        Args:
            file: Filename being processed
            available_chunks: Total number of chunks available
            field_name: Optional field name for field-specific strategies

        Returns:
            tuple: (k_chunks, page_filter)
                - k_chunks: Maximum number of chunks to retrieve
                - page_filter: Optional list of page filters for ChromaDB query
        """
        # Field-specific strategy override: author and title should only use first pages
        if field_name in ['ai:author', 'ai:title', 'ai:affiliation']:
            # Authors, titles, and affiliations are almost always on first 1-2 pages
            page_filter = [
                {"page": 0},
                {"page": 1},
            ]
            k_chunks = 5  # Limit to 5 chunks from first pages
            return k_chunks, page_filter

        # Strategy: all (default - backwards compatible)
        if self.retrieval_strategy == "all":
            k_chunks = min(available_chunks, self.max_retrieval_chunks) if self.max_retrieval_chunks else available_chunks
            return k_chunks, None

        # Strategy: first_and_last_pages (for metadata extraction)
        elif self.retrieval_strategy == "first_and_last_pages":
            # Get chunks from page 0 (first) and last page
            # We don't know total pages here, so we'll use collection query
            page_filter = [
                {"page": 0},  # First page
                {"page": 1},  # Second page (for overflow)
            ]
            # Also try to get last pages - we'll rely on k_chunks limit
            k_chunks = self.max_retrieval_chunks if self.max_retrieval_chunks else 10
            return k_chunks, None  # For now, just limit chunks

        # Strategy: first_pages_only (for metadata extraction)
        elif self.retrieval_strategy == "first_pages_only":
            page_filter = [
                {"page": 0},
                {"page": 1},
            ]
            k_chunks = self.max_retrieval_chunks if self.max_retrieval_chunks else 5
            return k_chunks, None  # For now, just limit chunks

        # Fallback to all
        else:
            k_chunks = min(available_chunks, self.max_retrieval_chunks) if self.max_retrieval_chunks else available_chunks
            return k_chunks, None

    def _create_retrieval_chain(self, vectorstore, file, pages, field_name=None):
        """
        Create LangChain retrieval chain for a specific file and field.

        Args:
            vectorstore: ChromaDB vector store
            file: Filename being processed
            pages: Total number of pages/chunks
            field_name: Optional field name for field-specific retrieval strategies

        Returns:
            RetrievalQA chain
        """
        # Apply retrieval strategy with intelligent chunk selection
        k_chunks, page_filter = self._apply_retrieval_strategy(file, pages, field_name)

        # Build search filter - ChromaDB doesn't support $or operator
        # For page filtering, we need to use a different approach
        search_filter = {"filename": file}

        # If we need page filtering for specific fields (author, title, affiliation),
        # we use a custom retriever that filters by page after retrieval
        if page_filter and field_name in ['ai:author', 'ai:title', 'ai:affiliation']:
            # Use custom document retriever with page filtering
            retriever_with_filter = self._create_page_filtered_retriever(
                vectorstore, file, k_chunks, page_filter
            )
        else:
            # Standard retriever without page filtering
            retriever_with_filter = vectorstore.as_retriever(
                search_kwargs={"filter": search_filter, "k": k_chunks}
            )

        prompt = PromptTemplate.from_template(self.prompt_manager.get_system_template())

        return self.llm_interface.create_qa_chain(
            retriever_with_filter,
            OllamaLLM(model=self.llm_model, temperature=0),
            prompt
        )

    def _create_page_filtered_retriever(self, vectorstore, file, k_chunks, page_filter):
        """
        Create a custom retriever that filters documents by page numbers.

        Args:
            vectorstore: ChromaDB vector store
            file: Filename to filter
            k_chunks: Number of chunks to retrieve
            page_filter: List of page dictionaries (e.g., [{"page": 0}, {"page": 1}])

        Returns:
            Custom retriever that only returns chunks from specified pages
        """
        from langchain_core.retrievers import BaseRetriever
        from langchain_core.callbacks import CallbackManagerForRetrieverRun
        from langchain_core.documents import Document
        from typing import List

        class PageFilteredRetriever(BaseRetriever):
            """Custom retriever that filters by page numbers"""

            def __init__(self, vectorstore, filename: str, k: int, pages: list):
                """Initialize the retriever with explicit storage of parameters"""
                super().__init__()
                # Store parameters without using Pydantic fields
                object.__setattr__(self, '_vectorstore', vectorstore)
                object.__setattr__(self, '_filename', filename)
                object.__setattr__(self, '_k', k)
                object.__setattr__(self, '_allowed_pages', set(p["page"] for p in pages))

            def _get_relevant_documents(
                self,
                query: str,
                *,
                run_manager: CallbackManagerForRetrieverRun = None
            ) -> List[Document]:
                """Retrieve documents and filter by page number"""
                # Get more documents than needed, then filter
                search_filter = {"filename": self._filename}
                # Request more documents to account for filtering
                retriever = self._vectorstore.as_retriever(
                    search_kwargs={"filter": search_filter, "k": self._k * 3}
                )

                all_docs = retriever.invoke(query)

                # Filter by page number
                filtered_docs = []
                for doc in all_docs:
                    if "page" in doc.metadata and doc.metadata["page"] in self._allowed_pages:
                        filtered_docs.append(doc)
                        if len(filtered_docs) >= self._k:
                            break

                return filtered_docs[:self._k]

        return PageFilteredRetriever(vectorstore, file, k_chunks, page_filter)
    
    def _process_single_field(self, field_name, file, vectorstore, pages, collection, existing_metadata):
        """
        Process a single metadata field using appropriate processor.

        Args:
            field_name: Name of the field to process
            file: Filename being processed
            vectorstore: ChromaDB vector store
            pages: Total number of pages/chunks
            collection: ChromaDB collection
            existing_metadata: Existing metadata for the file

        Returns:
            Dictionary with processed field data
        """
        should_process, process_type = self.config_manager.should_process_field(field_name, existing_metadata)

        if not should_process:
            return {}

        try:
            # Create field-specific retrieval chain
            chain = self._create_retrieval_chain(vectorstore, file, pages, field_name)

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
            elif field_name == 'ai:summary':
                return self.summary_processor.process_summary(file, chain)
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
            'ai:dewey': 'Dewey',
            'ai:summary': 'Summary'
        }
        
        # Priority fields that should always be shown if available
        priority_fields = ['ai:summary', 'ai:author', 'ai:title']
        
        # Show priority fields first (always display if available)
        for field_name in priority_fields:
            if field_name in metadata and metadata[field_name]:
                label = field_labels[field_name]
                value = metadata[field_name]
                if field_name == 'ai:author' and 'ai:revisedAuthor' in metadata:
                    value += f" / {metadata['ai:revisedAuthor']}"
                # Special formatting for summary (add separator for better readability)
                if field_name == 'ai:summary':
                    output_lines.append(f"{label:<12}: {value}")
                    output_lines.append("-" * 50)  # Visual separator after summary
                else:
                    output_lines.append(f"{label:<12}: {value}")
        
        # Show other fields only if they were processed
        for field_name, label in field_labels.items():
            if field_name not in priority_fields:  # Skip priority fields (already shown)
                if field_name in metadata and metadata[field_name]:
                    value = metadata[field_name]
                    output_lines.append(f"{label:<12}: {value}")
        
        return "\n            ".join(output_lines)
    
    def _safe_load_pickle(self, file_path):
        """Safely load pickle file, handling missing module errors"""
        import pickle
        import sys
        from io import BytesIO

        class SafeUnpickler(pickle.Unpickler):
            """Custom unpickler that handles missing modules gracefully"""
            def find_class(self, module, name):
                try:
                    return super().find_class(module, name)
                except (ModuleNotFoundError, AttributeError):
                    # If module not found, return a dummy class
                    logging.warning(f"Module {module}.{name} not found during unpickling, using placeholder")
                    return type(name, (), {})

        try:
            with open(file_path, 'rb') as f:
                return SafeUnpickler(f).load()
        except Exception as e:
            logging.error(f"Error loading pickle file: {e}")
            raise

    def execute_task(self):
        """Main execution method - orchestrates the entire process"""
        self._setup_logging()

        # Load input data
        df_files = pd.read_pickle(self.file_name_inputs)
        df_files = df_files[df_files['pipe:file_type'].isin(self.file_types)]

        # Load existing metadata
        if Path(self.file_name_output).exists():
            try:
                df_metadata = self._safe_load_pickle(self.file_name_output)
                print(f"✓ Loaded existing metadata: {len(df_metadata)} documents already processed")
            except Exception as e:
                print(f"✗ ERROR: Could not load existing metadata file: {e}")
                print(f"  Starting with empty metadata - all documents will be processed")
                df_metadata = pd.DataFrame()
        else:
            print(f"INFO: No existing metadata file found - starting with empty metadata")
            df_metadata = pd.DataFrame()
        
        # Setup vector store
        vectorstore, collection = self._setup_vector_store()
        
        # Get file chunk counts
        name_result = collection.get()
        filenames_list = [x["filename"] for x in name_result['metadatas']]
        chunk_counter = Counter(filenames_list)
        
        # Process each file
        processing_fields = self._get_processing_fields()

        # Statistics for debugging
        total_files = 0
        skipped_files = 0
        processed_files = 0

        for _, row in tqdm(df_files.iterrows(), total=df_files.shape[0]):
            if row['pipe:file_type'] not in self.file_types:
                continue

            file = f"{row['pipe:ID']}.{row['pipe:file_type']}"
            pages = chunk_counter.get(file, 0)

            if pages == 0:
                continue

            total_files += 1

            # Check existing metadata
            existing_metadata = None
            if df_metadata.shape[0] > 0:
                existing_rows = df_metadata[df_metadata['pipe:ID'] == row['pipe:ID']]
                if existing_rows.shape[0] > 0:
                    existing_metadata = existing_rows.iloc[0]

            # Check if file should be skipped
            should_skip = self.config_manager.should_skip_file(existing_metadata)
            if should_skip:
                skipped_files += 1
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

            # Process each field (chain is now created per-field with field-specific strategies)
            needs_processing = False
            for field_name in processing_fields:
                field_result = self._process_single_field(
                    field_name, file, vectorstore, pages, collection, existing_metadata
                )
                if field_result:
                    metadata.update(field_result)
                    needs_processing = True
            
            # Skip if no processing was needed
            if not needs_processing:
                continue

            processed_files += 1

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

        # Print statistics
        print(f"\n{'='*70}")
        print(f"=== PROCESSING STATISTICS ===")
        print(f"Total files checked: {total_files}")
        print(f"Files skipped (complete): {skipped_files}")
        print(f"Files processed: {processed_files}")
        print(f"Final metadata count: {len(df_metadata)}")
        print(f"{'='*70}\n")
