"""Factory for creating vector store instances."""

from typing import Union
from langchain.embeddings.base import Embeddings

from pysrcai.src.config.embedding_config import VectorDBConfig
from .base import BaseVectorStore
from .chroma_adapter import ChromaVectorStore


def create_vectorstore(
    config: VectorDBConfig,
    embeddings: Embeddings
) -> BaseVectorStore:
    """Create a vector store instance based on configuration.

    Args:
        config: Vector database configuration
        embeddings: Embeddings model to use

    Returns:
        Vector store instance

    Raises:
        ValueError: If provider is not supported
    """
    # For now, we only support Chroma, but this can be extended later
    # to support other vector stores like Pinecone, Weaviate, etc.
    return ChromaVectorStore(config=config, embeddings=embeddings)
