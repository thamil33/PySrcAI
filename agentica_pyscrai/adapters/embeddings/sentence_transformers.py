"""Local sentence-transformers embeddings."""

from typing import List, Optional
import torch
from sentence_transformers import SentenceTransformer

from ...config.config import EmbeddingConfig
from .base import BaseEmbedder


class SentenceTransformerEmbeddings(BaseEmbedder):
    """Local embeddings using sentence-transformers models."""
    
    def __init__(self, config: EmbeddingConfig):
        """Initialize the sentence-transformers embeddings adapter.
        
        Args:
            config: Embedding configuration containing model name and device settings
        """
        self.config = config
        try:
            self.model = SentenceTransformer(
                model_name_or_path=config.model,
                device=config.device,
                cache_folder=config.cache_folder
            )
        except Exception as e:
            # Try fallback models if available
            if config.fallback_models:
                for model in config.fallback_models:
                    try:
                        self.model = SentenceTransformer(
                            model_name_or_path=model,
                            device=config.device,
                            cache_folder=config.cache_folder
                        )
                        self.config.model = model  # Update to working model
                        return
                    except Exception:                        continue
            raise ValueError(f"Failed to load sentence-transformers model: {e}")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        embeddings = self.model.encode(
            texts,
            convert_to_tensor=True,
            device=self.config.device
        )
        # Convert to list format and normalize
        # Handle both tensor and numpy array cases
        if hasattr(embeddings, 'cpu'):
            # PyTorch tensor
            embeddings_list = embeddings.cpu().numpy().tolist()
        else:
            # Numpy array (e.g., from mocks)
            embeddings_list = embeddings.tolist()
        return self._normalize_embeddings(embeddings_list)
    
    def embed_query(self, text: str) -> List[float]:
        """Generate embedding for a single query text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        embeddings = self.embed_documents([text])
        return embeddings[0]
