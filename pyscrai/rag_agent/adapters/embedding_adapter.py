"""Embedding adapter for RAG Agents."""

import numpy as np
from typing import List, Union
from ..config_loader import AgentConfig, get_hf_home, get_api_key


class EmbeddingAdapter:
    """Adapter class to handle different embedding providers."""
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.provider = config.models.embedding_provider
        self.model_name = config.models.embedding_model
        self.embedder = None
        self._initialize_embedder()
    
    def _initialize_embedder(self):
        """Initialize the appropriate embedder based on configuration."""
        if self.provider == "local_sentencetransformers":
            self._initialize_sentence_transformers()
        elif self.provider == "huggingface_api":
            self._initialize_huggingface_api()
        else:
            raise ValueError(f"Unsupported embedding provider: {self.provider}")
    
    def _initialize_sentence_transformers(self):
        """Initialize SentenceTransformers embedder (local)."""
        try:
            from sentence_transformers import SentenceTransformer
            import os
            
            self.embedder = SentenceTransformer(
                model_name_or_path=self.model_name,
                device="cpu",  # Default to CPU, could be made configurable
                cache_folder=get_hf_home(),
                local_files_only=False
            )
            print(f"Initialized SentenceTransformers with model: {self.model_name}")
            
        except ImportError:
            raise ImportError("sentence-transformers not installed. Run: pip install -r requirements-cpu.txt")
    
    def _initialize_huggingface_api(self):
        """Initialize HuggingFace API embedder (cloud)."""
        try:
            # Import the HF embedder from pyscrai
            from ...embedding.hf_embedding import HFEmbedder
              # Get API token
            hf_token = get_api_key("HF_API_TOKEN")
            
            # Initialize with the specific model
            models = [self.model_name] if self.model_name else None
            self.embedder = HFEmbedder(token=hf_token, models=models)
            print(f"Initialized HuggingFace API with model: {self.model_name}")
            
        except ImportError as e:
            raise ImportError(f"Failed to import HuggingFace embedder: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize HuggingFace API embedder: {e}")
    
    def embed_text(self, text: str) -> np.ndarray:
        """Embed a single text string."""
        if self.provider == "local_sentencetransformers":
            return self.embedder.encode(text, convert_to_numpy=True)
        elif self.provider == "huggingface_api":
            return self.embedder(text)  # HFEmbedder uses __call__ method
    
    def embed_texts(self, texts: List[str]) -> List[np.ndarray]:
        """Embed a list of text strings."""
        if self.provider == "local_sentencetransformers":
            # SentenceTransformers can handle batch processing efficiently
            embeddings = self.embedder.encode(texts, convert_to_numpy=True)
            return [embedding for embedding in embeddings]
        elif self.provider == "huggingface_api":
            # Process individually for HuggingFace API
            return [self.embedder(text) for text in texts]
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of the embeddings."""
        if self.provider == "local_sentencetransformers":
            return self.embedder.get_sentence_embedding_dimension()
        elif self.provider == "huggingface_api":
            return self.embedder.embedding_dim
        else:
            return 768  # Default dimension
            return [self.embedder.embed(text) for text in texts]
    
    def get_dimension(self) -> int:
        """Get the embedding dimension."""
        if self.provider == "sentence_transformers":
            return self.embedder.get_sentence_embedding_dimension()
        elif self.provider == "huggingface":
            # For HuggingFace, we need to make a test embedding to get dimension
            test_embedding = self.embed_text("test")
            return len(test_embedding)
