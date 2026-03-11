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
from ai_metadata_core.processors.education_processor import EducationProcessor

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

        # Batch processing configuration
        # Save results every N documents to reduce I/O overhead and improve crash recovery
        self.batch_size = stage_param.get('batch_size', 50)
    
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

        # Education processor
        self.education_processor = EducationProcessor(self.prompt_manager, self.llm_interface)
    
    def _setup_logging(self):
        """Configure logging to reduce noise from dependencies"""
        # Diese Methode ist jetzt obsolet - Logging wird zentral konfiguriert
        # Behalten für Rückwärtskompatibilität, aber keine Funktion mehr
        pass
    
    def _setup_vector_store(self):
        """Setup ChromaDB vector store and embeddings"""
        embeddings = OllamaEmbeddings(
            base_url=self.base_url,
            model=self.embedding_model,
            num_gpu=0
        )

        # Warmup: ensure embedding model is loaded before processing
        logging.info(f"Warming up embedding model '{self.embedding_model}'...")
        for attempt in range(3):
            try:
                embeddings.embed_query("warmup")
                logging.info("Embedding model ready.")
                break
            except Exception as e:
                logging.warning(f"Warmup attempt {attempt + 1}/3 failed: {e}")
                import time
                time.sleep(5)

        # Warmup: ensure LLM model is loaded before processing
        logging.info(f"Warming up LLM model '{self.llm_model}'...")
        for attempt in range(3):
            try:
                llm = OllamaLLM(model=self.llm_model, temperature=0)
                llm.invoke("warmup")
                logging.info("LLM model ready.")
                break
            except Exception as e:
                logging.warning(f"LLM warmup attempt {attempt + 1}/3 failed: {e}")
                import time
                time.sleep(15)

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
            tuple: (k_chunks, page_filter, use_hybrid)
                - k_chunks: Maximum number of chunks to retrieve
                - page_filter: Optional list of page filters for ChromaDB query
                - use_hybrid: Whether to use hybrid positional+similarity retrieval
        """
        # Field-specific strategy: author/title/affiliation use hybrid retrieval
        # Always include first N chunks by position + additional by similarity
        if field_name in ['ai:author', 'ai:title', 'ai:affiliation']:
            k_chunks = 8
            return k_chunks, None, True

        # Strategy: all (default - backwards compatible)
        if self.retrieval_strategy == "all":
            k_chunks = min(available_chunks, self.max_retrieval_chunks) if self.max_retrieval_chunks else available_chunks
            return k_chunks, None, False

        # Strategy: first_and_last_pages (for metadata extraction)
        elif self.retrieval_strategy == "first_and_last_pages":
            k_chunks = self.max_retrieval_chunks if self.max_retrieval_chunks else 10
            return k_chunks, None, False

        # Strategy: first_pages_only (for metadata extraction)
        elif self.retrieval_strategy == "first_pages_only":
            k_chunks = self.max_retrieval_chunks if self.max_retrieval_chunks else 5
            return k_chunks, None, False

        # Fallback to all
        else:
            k_chunks = min(available_chunks, self.max_retrieval_chunks) if self.max_retrieval_chunks else available_chunks
            return k_chunks, None, False

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
        k_chunks, page_filter, use_hybrid = self._apply_retrieval_strategy(file, pages, field_name)

        search_filter = {"filename": file}

        if use_hybrid and field_name in ['ai:author', 'ai:title', 'ai:affiliation']:
            # Hybrid retrieval: first N chunks by position + additional by similarity
            retriever_with_filter = self._create_hybrid_retriever(
                vectorstore, file, n_positional=3, n_similarity=5
            )
        else:
            # Standard similarity retriever
            retriever_with_filter = vectorstore.as_retriever(
                search_kwargs={"filter": search_filter, "k": k_chunks}
            )

        prompt = PromptTemplate.from_template(self.prompt_manager.get_system_template())

        return self.llm_interface.create_qa_chain(
            retriever_with_filter,
            OllamaLLM(model=self.llm_model, temperature=0),
            prompt
        )

    def _create_hybrid_retriever(self, vectorstore, file, n_positional=3, n_similarity=5):
        """
        Create a hybrid retriever that combines positional and similarity-based retrieval.

        For title/author/affiliation extraction, the first chunks of a document (by position)
        are critical because they contain the title page. Pure similarity-based retrieval
        may miss these chunks in large documents. This retriever guarantees that the first
        N chunks by chunk_index are always included, plus additional chunks by similarity.

        Args:
            vectorstore: ChromaDB vector store
            file: Filename to filter
            n_positional: Number of first chunks to always include (by chunk_index)
            n_similarity: Number of additional chunks to retrieve by similarity

        Returns:
            HybridPositionalRetriever instance
        """
        from langchain_core.retrievers import BaseRetriever
        from langchain_core.callbacks import CallbackManagerForRetrieverRun
        from langchain_core.documents import Document
        from typing import List

        class HybridPositionalRetriever(BaseRetriever):
            """Retriever combining positional (first N chunks) with similarity retrieval.

            Ensures the document beginning (title, author, affiliation) is always in context,
            while also including semantically relevant chunks from the rest of the document.
            """

            def __init__(self, vectorstore, filename: str, n_pos: int, n_sim: int):
                super().__init__()
                object.__setattr__(self, '_vectorstore', vectorstore)
                object.__setattr__(self, '_filename', filename)
                object.__setattr__(self, '_n_pos', n_pos)
                object.__setattr__(self, '_n_sim', n_sim)

            def _get_relevant_documents(
                self,
                query: str,
                *,
                run_manager: CallbackManagerForRetrieverRun = None
            ) -> List[Document]:
                """Retrieve first N chunks by position + additional by similarity."""
                # 1. Get first N chunks by chunk_index (positional)
                try:
                    positional_docs = self._vectorstore.similarity_search(
                        query,
                        k=self._n_pos,
                        filter={
                            "$and": [
                                {"filename": self._filename},
                                {"chunk_index": {"$lte": self._n_pos - 1}}
                            ]
                        }
                    )
                except Exception:
                    # Fallback if chunk_index metadata not available (old embeddings)
                    positional_docs = []

                # 2. Get additional chunks by similarity (from entire document)
                similarity_docs = self._vectorstore.similarity_search(
                    query,
                    k=self._n_pos + self._n_sim,
                    filter={"filename": self._filename}
                )

                # 3. Combine: positional first, then similarity (deduplicated)
                seen_content = set()
                combined = []

                # Add positional chunks first (sorted by chunk_index)
                positional_docs.sort(
                    key=lambda d: d.metadata.get('chunk_index', 999)
                )
                for doc in positional_docs:
                    content_hash = hash(doc.page_content[:200])
                    if content_hash not in seen_content:
                        seen_content.add(content_hash)
                        combined.append(doc)

                # Add similarity chunks (skip duplicates)
                for doc in similarity_docs:
                    content_hash = hash(doc.page_content[:200])
                    if content_hash not in seen_content:
                        seen_content.add(content_hash)
                        combined.append(doc)

                return combined[:self._n_pos + self._n_sim]

        return HybridPositionalRetriever(vectorstore, file, n_positional, n_similarity)
    
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
            Dictionary with processed field data (empty dict on error/skip).
            After calling this method, check self.llm_interface.last_error
            to determine if the empty result was due to an LLM error.
        """
        should_process, process_type = self.config_manager.should_process_field(field_name, existing_metadata)

        if not should_process:
            if process_type == "max_retries_exceeded":
                logging.info("Skipping field %s for %s: max error retries exceeded", field_name, file)
            return {}

        # Reset error state before processing
        self.llm_interface.last_error = None

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
            elif field_name == 'ai:education_level':
                return self.education_processor.process_education_level(file, chain)
            elif field_name == 'ai:target_audience':
                return self.education_processor.process_target_audience(file, chain)
            else:
                logging.warning("Unknown field type: %s", field_name)
                return {}

        except Exception as e:
            logging.error("Error processing field %s for file %s: %s", field_name, file, e)
            self.llm_interface.last_error = "llm_error"
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
                value = str(metadata[field_name])  # Convert to string to handle float/NaN values
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

        # Filter only validated LiaScript files (if validation column exists)
        if 'pipe:is_valid_liascript' in df_files.columns:
            initial_count = len(df_files)
            df_files = df_files[df_files['pipe:is_valid_liascript'] == True]
            filtered_count = len(df_files)
            logging.info(f"Filtered files: {initial_count} -> {filtered_count} (removed {initial_count - filtered_count} invalid files)")

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
        
        # Get file chunk counts (paginated to avoid SQLite variable limit)
        filenames_list = []
        batch_size = 5000
        offset = 0
        while True:
            name_result = collection.get(limit=batch_size, offset=offset, include=["metadatas"])
            if not name_result['metadatas']:
                break
            filenames_list.extend(x["filename"] for x in name_result['metadatas'])
            if len(name_result['metadatas']) < batch_size:
                break
            offset += batch_size
        chunk_counter = Counter(filenames_list)
        
        # Process each file
        processing_fields = self._get_processing_fields()

        # Statistics for debugging
        total_files = 0
        skipped_files = 0
        processed_files = 0
        batch_counter = 0
        batch_buffer = []

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
            
            # Copy existing metadata (including error tracking)
            if existing_metadata is not None:
                for key in existing_metadata.index:
                    if key.startswith('ai:'):
                        metadata[key] = existing_metadata[key]
                # Ensure ai:_errors is a proper dict (may be NaN from pickle)
                if 'ai:_errors' in metadata and not isinstance(metadata.get('ai:_errors'), dict):
                    metadata['ai:_errors'] = {}

            # Process each field (chain is now created per-field with field-specific strategies)
            needs_processing = False
            for field_name in processing_fields:
                should_process, process_type = self.config_manager.should_process_field(field_name, existing_metadata)

                if should_process:
                    field_result = self._process_single_field(
                        field_name, file, vectorstore, pages, collection, existing_metadata
                    )

                    # Check if there was an LLM error for this field
                    last_error = self.llm_interface.last_error

                    if field_result:
                        metadata.update(field_result)
                        # Successful extraction — clear any previous error for this field
                        self.config_manager.clear_field_error(metadata, field_name)
                    elif last_error:
                        # LLM error (timeout or other) — record it, do NOT mark field as processed
                        self.config_manager.record_field_error(metadata, field_name, last_error)
                        logging.warning(
                            "Error '%s' for field %s on file %s (attempt %d)",
                            last_error, field_name, file,
                            metadata.get('ai:_errors', {}).get(field_name, {}).get('count', 0)
                        )
                    else:
                        # No error but empty result — legitimate empty value, mark as processed
                        if field_name in ['ai:dewey']:
                            metadata[field_name] = []
                        elif field_name in ['ai:keywords_ext', 'ai:keywords_gen', 'ai:keywords_dnb']:
                            metadata[field_name] = []
                        else:
                            metadata[field_name] = ""
                        self.config_manager.clear_field_error(metadata, field_name)
                    needs_processing = True

            # Skip if no processing was needed
            if not needs_processing:
                continue

            processed_files += 1

            # Display results
            output = self._format_output(file, metadata, existing_metadata)
            print(f"\n{output}\n")

            # Ensure ai:dewey is always a list
            if 'ai:dewey' not in metadata:
                metadata['ai:dewey'] = []

            # Add to batch buffer
            batch_buffer.append(metadata)
            batch_counter += 1

            # Save batch when batch_size is reached
            if batch_counter >= self.batch_size:
                # Update DataFrame with batch
                for batch_metadata in batch_buffer:
                    # Remove existing entry if present
                    if 'pipe:ID' in df_metadata.columns:
                        df_metadata = df_metadata[df_metadata['pipe:ID'] != batch_metadata['pipe:ID']]
                    # Add new entry
                    df_aux = pd.DataFrame([batch_metadata])
                    df_metadata = pd.concat([df_metadata, df_aux], ignore_index=True)

                # Save to disk
                df_metadata.to_pickle(self.file_name_output)
                print(f"✓ Batch saved: {batch_counter} documents processed (Total: {len(df_metadata)} in metadata)", flush=True)

                # Reset batch
                batch_buffer = []
                batch_counter = 0
        
        # Save remaining items in batch buffer
        if batch_buffer:
            for batch_metadata in batch_buffer:
                # Remove existing entry if present
                if 'pipe:ID' in df_metadata.columns:
                    df_metadata = df_metadata[df_metadata['pipe:ID'] != batch_metadata['pipe:ID']]
                # Add new entry
                df_aux = pd.DataFrame([batch_metadata])
                df_metadata = pd.concat([df_metadata, df_aux], ignore_index=True)

            # Save final batch
            df_metadata.reset_index(drop=True, inplace=True)
            df_metadata.to_pickle(self.file_name_output)
            print(f"✓ Final batch saved: {len(batch_buffer)} documents", flush=True)

        # Final save (ensure index is clean)
        df_metadata.reset_index(drop=True, inplace=True)
        df_metadata.to_pickle(self.file_name_output)

        # Print statistics
        print(f"\n{'='*70}")
        print(f"=== PROCESSING STATISTICS ===")
        print(f"Total files checked: {total_files}")
        print(f"Files skipped (complete): {skipped_files}")
        print(f"Files processed: {processed_files}")
        print(f"Batch size configured: {self.batch_size}")
        print(f"Final metadata count: {len(df_metadata)}")

        # Error retry statistics
        if self.config_manager.max_error_retries > 0 and 'ai:_errors' in df_metadata.columns:
            docs_with_errors = df_metadata['ai:_errors'].apply(
                lambda x: isinstance(x, dict) and len(x) > 0
            ).sum()
            if docs_with_errors > 0:
                print(f"\n--- Error Tracking (max_error_retries: {self.config_manager.max_error_retries}) ---")
                print(f"Documents with field errors: {docs_with_errors}")
                # Count errors by field
                field_error_counts = {}
                for errors in df_metadata['ai:_errors']:
                    if isinstance(errors, dict):
                        for field, info in errors.items():
                            if field not in field_error_counts:
                                field_error_counts[field] = {'total': 0, 'max_retries_reached': 0}
                            field_error_counts[field]['total'] += 1
                            if info.get('count', 0) >= self.config_manager.max_error_retries:
                                field_error_counts[field]['max_retries_reached'] += 1
                for field, counts in sorted(field_error_counts.items()):
                    print(f"  {field}: {counts['total']} docs with errors, "
                          f"{counts['max_retries_reached']} at max retries")

        print(f"{'='*70}\n")
