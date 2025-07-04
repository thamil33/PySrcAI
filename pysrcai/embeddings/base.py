"""Base interface and adapters for text embeddings."""

from abc import ABC, abstractmethod
from typing import List, Optional
import numpy as np

from langchain.embeddings.base import Embeddings


class BaseEmbedder(Embeddings):
    """Base class for embedding adapters, conforming to LangChain's Embeddings interface."""

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        pass

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """Generate embedding for a single query text."""
        pass

    def _normalize_embedding(self, embedding: List[float]) -> List[float]:
        """Normalize an embedding vector to unit length."""
        array = np.array(embedding)
        normalized = array / np.linalg.norm(array)
        return normalized.tolist()

    def _normalize_embeddings(self, embeddings: List[List[float]]) -> List[List[float]]:
        """Normalize multiple embedding vectors."""
        return [self._normalize_embedding(emb) for emb in embeddings]
