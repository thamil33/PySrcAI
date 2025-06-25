"""Chroma vector store adapter."""

from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path

try:
    # Requires: pip install langchain-chroma
    from langchain_chroma import Chroma
except ImportError:
    # Fallback to deprecated import if langchain-chroma not installed
    from langchain_community.vectorstores import Chroma
    
from langchain.schema import Document
from langchain.embeddings.base import Embeddings

from .base import BaseVectorStore
from pyscrai.config.config import VectorDBConfig


class ChromaVectorStore(BaseVectorStore):
    """Vector store adapter using ChromaDB."""

    def __init__(self, config: VectorDBConfig, embeddings: Embeddings):
        """Initialize the Chroma vector store.
        
        Args:
            config: Vector database configuration
            embeddings: Embeddings model to use
        """
        self.config = config
        self.embeddings = embeddings
        
        # Ensure the persist directory exists
        Path(config.persist_directory).mkdir(parents=True, exist_ok=True)
        
        # Initialize Chroma vector store
        self.vectorstore = Chroma(
            collection_name=config.collection_name,
            embedding_function=embeddings,
            persist_directory=config.persist_directory,
            collection_metadata=config.settings
        )

    def add_documents(self, documents: List[Document], **kwargs) -> List[str]:
        """Add documents to the Chroma vector store.
        
        Args:
            documents: List of documents to add
            **kwargs: Additional arguments for Chroma
            
        Returns:
            List of document IDs
        """
        return self.vectorstore.add_documents(documents, **kwargs)

    def similarity_search(
        self, 
        query: str, 
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[Document]:
        """Search for similar documents.
        
        Args:
            query: Query string
            k: Number of documents to return
            filter: Optional metadata filter
            **kwargs: Additional arguments
            
        Returns:
            List of similar documents
        """
        return self.vectorstore.similarity_search(
            query=query,
            k=k,
            filter=filter,
            **kwargs
        )

    def similarity_search_with_score(
        self, 
        query: str, 
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[Tuple[Document, float]]:
        """Search for similar documents with similarity scores.
        
        Args:
            query: Query string
            k: Number of documents to return
            filter: Optional metadata filter
            **kwargs: Additional arguments
            
        Returns:
            List of (document, score) tuples
        """
        return self.vectorstore.similarity_search_with_score(
            query=query,
            k=k,
            filter=filter,
            **kwargs
        )

    def delete(self, ids: List[str], **kwargs) -> bool:
        """Delete documents by IDs.
        
        Args:
            ids: List of document IDs to delete
            **kwargs: Additional arguments
            
        Returns:
            True if successful
        """
        try:
            self.vectorstore.delete(ids=ids, **kwargs)
            return True
        except Exception:
            return False

    def clear(self) -> bool:
        """Clear all documents from the vector store.
        
        Returns:
            True if successful
        """
        try:
            # Get all document IDs and delete them
            collection = self.vectorstore._collection
            result = collection.get()
            if result['ids']:
                collection.delete(ids=result['ids'])
            return True
        except Exception:
            return False

    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the Chroma collection.
        
        Returns:
            Dictionary with collection information
        """
        try:
            collection = self.vectorstore._collection
            count = collection.count()
            return {
                "name": self.config.collection_name,
                "count": count,
                "persist_directory": self.config.persist_directory,
                "embedding_function": str(type(self.embeddings).__name__),
                "metadata": self.config.settings
            }
        except Exception as e:
            return {
                "name": self.config.collection_name,
                "error": str(e)
            }

    def persist(self):
        """Persist the vector store to disk."""
        if hasattr(self.vectorstore, 'persist'):
            self.vectorstore.persist()

    def add_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> List[str]:
        """Add raw texts to the vector store.
        
        Args:
            texts: List of texts to add
            metadatas: Optional list of metadata dicts
            **kwargs: Additional arguments
            
        Returns:
            List of document IDs
        """
        return self.vectorstore.add_texts(
            texts=texts,
            metadatas=metadatas,
            **kwargs
        )

    def from_documents(
        cls,
        documents: List[Document],
        config: VectorDBConfig,
        embeddings: Embeddings,
        **kwargs
    ) -> "ChromaVectorStore":
        """Create a ChromaVectorStore from documents.
        
        Args:
            documents: List of documents to add
            config: Vector database configuration
            embeddings: Embeddings model to use
            **kwargs: Additional arguments
            
        Returns:
            ChromaVectorStore instance
        """
        vectorstore = cls(config=config, embeddings=embeddings)
        if documents:
            vectorstore.add_documents(documents, **kwargs)
        return vectorstore
