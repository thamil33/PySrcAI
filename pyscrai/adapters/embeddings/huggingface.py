"""HuggingFace API-based embeddings."""

import os
from typing import List, Optional
import requests

from ...config.config import EmbeddingConfig
from .base import BaseEmbedder


class HuggingFaceEmbeddings(BaseEmbedder):
    """Embeddings via HuggingFace's Inference API."""
    
    def __init__(self, config: EmbeddingConfig):
        """Initialize the HuggingFace embeddings adapter.
        
        Args:
            config: Embedding configuration containing model name and API settings
        """
        self.config = config
        self.api_key = os.getenv(config.hf_api_token_env)
        if not self.api_key:
            raise ValueError(f"Environment variable {config.hf_api_token_env} not set")
        
        self.api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{config.model}"
        self.headers = {"Authorization": f"Bearer {self.api_key}"}
        
        # Test connection
        self._test_connection()
    
    def _test_connection(self):
        """Test the API connection with a simple request."""
        try:
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json={"inputs": ["test"], "options": {"wait_for_model": True}}
            )
            response.raise_for_status()
        except Exception as e:
            # Try fallback models if available
            if self.config.fallback_models:
                for model in self.config.fallback_models:
                    try:
                        self.api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model}"
                        response = requests.post(
                            self.api_url,
                            headers=self.headers,
                            json={"inputs": ["test"], "options": {"wait_for_model": True}}
                        )
                        response.raise_for_status()
                        self.config.model = model  # Update to working model
                        return
                    except Exception:
                        continue
            raise ValueError(f"Failed to connect to HuggingFace API: {e}")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        response = requests.post(
            self.api_url,
            headers=self.headers,
            json={"inputs": texts, "options": {"wait_for_model": True}}
        )
        response.raise_for_status()
        embeddings = response.json()
        return self._normalize_embeddings(embeddings)
    
    def embed_query(self, text: str) -> List[float]:
        """Generate embedding for a single query text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        embeddings = self.embed_documents([text])
        return embeddings[0]
