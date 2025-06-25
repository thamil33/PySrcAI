"""Document ingestion pipeline."""

from typing import List, Optional, Dict, Any, Callable
from pathlib import Path
from langchain.schema import Document

from ..config.config import AgentConfig, ChunkingConfig
from ..adapters.embeddings.factory import create_embedder
from ..adapters.vectorstore.factory import create_vectorstore
from .loaders import create_loader, DirectoryLoader
from .chunkers import create_chunker


class IngestionPipeline:
    """Pipeline for ingesting documents into a vector store."""
    
    def __init__(self, config: AgentConfig):
        """Initialize the ingestion pipeline.
        
        Args:
            config: Agent configuration containing all necessary settings
        """
        import logging, time, pprint
        self.config = config
        self.logger = logging.getLogger("pysrcai.agentica.ingestion.pipeline")
        self.logger.info("--- FULL AGENT CONFIG ---")
        self.logger.info(pprint.pformat(vars(config)))
        self.logger.info("--- EMBEDDING CONFIG ---")
        self.logger.info(pprint.pformat(vars(config.embedding)))
        self.logger.info("--- CHUNKING CONFIG ---")
        self.logger.info(pprint.pformat(vars(config.chunking)))
        self.logger.info("--- RAG CONFIG ---")
        self.logger.info(pprint.pformat(vars(config.rag)))
        start = time.time()
        self.logger.info("Initializing embedder...")
        self.embeddings = create_embedder(config.embedding)
        self.logger.info(f"Embedder initialized in {time.time() - start:.2f}s")

        start = time.time()
        self.logger.info("Initializing vectorstore...")
        self.vectorstore = create_vectorstore(config.vectordb, self.embeddings)
        self.logger.info(f"Vectorstore initialized in {time.time() - start:.2f}s")

        start = time.time()
        self.logger.info("Initializing chunker...")
        self.chunker = create_chunker(config.chunking)
        self.logger.info(f"Chunker initialized in {time.time() - start:.2f}s")

        # Preprocessing hooks
        self.preprocessing_hooks: List[Callable[[Document], Document]] = []
    
    def add_preprocessing_hook(self, hook: Callable[[Document], Document]):
        """Add a preprocessing hook.
        
        Args:
            hook: Function that takes a Document and returns a modified Document
        """
        self.preprocessing_hooks.append(hook)
    
    def ingest_documents(
        self, 
        documents: List[Document],
        chunk: bool = True,
        **kwargs
    ) -> List[str]:
        """Ingest a list of documents.
        
        Args:
            documents: List of documents to ingest
            chunk: Whether to chunk the documents
            **kwargs: Additional arguments for vector store
            
        Returns:
            List of document IDs
        """
        processed_docs = documents.copy()
        
        # Apply preprocessing hooks
        for hook in self.preprocessing_hooks:
            processed_docs = [hook(doc) for doc in processed_docs]
        
        # Chunk documents if requested
        if chunk:
            processed_docs = self.chunker.chunk_documents(processed_docs)
        
        # Add to vector store
        return self.vectorstore.add_documents(processed_docs, **kwargs)
    
    def ingest_files(
        self, 
        file_paths: List[str],
        loader_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[str]:
        """Ingest files.
        
        Args:
            file_paths: List of file paths to ingest
            loader_kwargs: Additional arguments for loaders
            **kwargs: Additional arguments for vector store
            
        Returns:
            List of document IDs
        """
        loader_kwargs = loader_kwargs or {}
        documents = []
        
        for file_path in file_paths:
            try:
                loader = create_loader(file_path, **loader_kwargs)
                docs = loader.load()
                documents.extend(docs)
            except Exception as e:
                print(f"Warning: Failed to load {file_path}: {e}")
        
        return self.ingest_documents(documents, **kwargs)
    
    def ingest_directory(
        self,
        directory_path: str,
        file_extensions: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None,
        recursive: bool = True,
        **kwargs
    ) -> List[str]:
        """Ingest all files in a directory.
        
        Args:
            directory_path: Path to the directory
            file_extensions: List of file extensions to include
            exclude_patterns: List of patterns to exclude
            recursive: Whether to search recursively
            **kwargs: Additional arguments for vector store
            
        Returns:
            List of document IDs
        """
        glob_pattern = "**/*" if recursive else "*"
        
        loader = DirectoryLoader(
            directory_path=directory_path,
            glob_pattern=glob_pattern,
            file_extensions=file_extensions,
            exclude_patterns=exclude_patterns
        )
        
        documents = loader.load()
        return self.ingest_documents(documents, **kwargs)
    
    def ingest_from_config(self, **kwargs) -> List[str]:
        """Ingest documents specified in the configuration.
        
        Args:
            **kwargs: Additional arguments for vector store
            
        Returns:
            List of document IDs
        """
        if not self.config.data_paths:
            print("No data paths specified in configuration.")
            return []
        
        all_doc_ids = []
        
        for data_path in self.config.data_paths:
            path_obj = Path(data_path)
            
            if path_obj.is_file():
                doc_ids = self.ingest_files([data_path], **kwargs)
            elif path_obj.is_dir():
                doc_ids = self.ingest_directory(data_path, **kwargs)
            else:
                print(f"Warning: Path does not exist: {data_path}")
                continue
            
            all_doc_ids.extend(doc_ids)
            print(f"Ingested {len(doc_ids)} documents from {data_path}")
        
        return all_doc_ids
    
    def clear_vectorstore(self) -> bool:
        """Clear all documents from the vector store.
        
        Returns:
            True if successful
        """
        return self.vectorstore.clear()
    
    def get_vectorstore_info(self) -> Dict[str, Any]:
        """Get information about the vector store.
        
        Returns:
            Dictionary with vector store information
        """
        return self.vectorstore.get_collection_info()
    
    def search(
        self, 
        query: str, 
        k: int = None,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[Document]:
        """Search for similar documents.
        
        Args:
            query: Query string
            k: Number of documents to return (defaults to config.rag.top_k)
            filter: Optional metadata filter
            **kwargs: Additional arguments
            
        Returns:
            List of similar documents
        """
        if k is None:
            k = self.config.rag.top_k
        
        return self.vectorstore.similarity_search(
            query=query,
            k=k,
            filter=filter,
            **kwargs
        )
    
    def search_with_scores(
        self, 
        query: str, 
        k: int = None,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[tuple]:
        """Search for similar documents with scores.
        
        Args:
            query: Query string
            k: Number of documents to return (defaults to config.rag.top_k)
            filter: Optional metadata filter
            **kwargs: Additional arguments
            
        Returns:
            List of (document, score) tuples
        """
        if k is None:
            k = self.config.rag.top_k
        
        results = self.vectorstore.similarity_search_with_score(
            query=query,
            k=k,
            filter=filter,
            **kwargs
        )
        
        # Filter by similarity threshold if configured
        if self.config.rag.similarity_threshold > 0:
            results = [
                (doc, score) for doc, score in results 
                if score >= self.config.rag.similarity_threshold
            ]
        
        return results
