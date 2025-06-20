"""Embedding adapters package."""

from .base import BaseEmbedder
from .factory import create_embedder
from .huggingface import HuggingFaceEmbeddings
from .sentence_transformers import SentenceTransformerEmbeddings

__all__ = [
    "BaseEmbedder",
    "create_embedder",
    "HuggingFaceEmbeddings",
    "SentenceTransformerEmbeddings"
]
