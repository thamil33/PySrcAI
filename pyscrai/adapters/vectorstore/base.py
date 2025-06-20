"""Base interface for vector stores."""

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Dict, Any
from langchain.schema import Document


class BaseVectorStore(ABC):
    """Abstract base class for vector store adapters."""

    @abstractmethod
    def add_documents(self, documents: List[Document], **kwargs) -> List[str]:
        """Add documents to the vector store.
        
        Args:
            documents: List of documents to add
            **kwargs: Additional arguments for the vector store
            
        Returns:
            List of document IDs
        """
        pass

    @abstractmethod
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
        pass

    @abstractmethod
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
        pass

    @abstractmethod
    def delete(self, ids: List[str], **kwargs) -> bool:
        """Delete documents by IDs.
        
        Args:
            ids: List of document IDs to delete
            **kwargs: Additional arguments
            
        Returns:
            True if successful
        """
        pass

    @abstractmethod
    def clear(self) -> bool:
        """Clear all documents from the vector store.
        
        Returns:
            True if successful
        """
        pass

    @abstractmethod
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the vector store collection.
        
        Returns:
            Dictionary with collection information
        """
        pass
