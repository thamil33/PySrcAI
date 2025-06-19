"""Embedding adapter for the Concordia Assistant."""

import numpy as np
from typing import List, Union
from .config_loader import AssistantConfig, get_hf_home, get_api_key


class EmbeddingAdapter:
    """Adapter class to handle different embedding providers."""
    
    def __init__(self, config: AssistantConfig):
        self.config = config
        self.provider = config.embedding.provider
        self.embedder = None
        self._initialize_embedder()
    
    def _initialize_embedder(self):
        """Initialize the appropriate embedder based on configuration."""
        if self.provider == "sentence_transformers":
            self._initialize_sentence_transformers()
        elif self.provider == "huggingface":
            self._initialize_huggingface()
        else:
            raise ValueError(f"Unsupported embedding provider: {self.provider}")
    
    def _initialize_sentence_transformers(self):
        """Initialize SentenceTransformers embedder."""
        try:
            from sentence_transformers import SentenceTransformer
            import os
            
            st_config = self.config.embedding.sentence_transformers
            model_name = st_config.get("model", "sentence-transformers/all-mpnet-base-v2")
            device = st_config.get("device", "cpu")
            local_files_only = st_config.get("local_files_only", True)
            
            self.embedder = SentenceTransformer(
                model_name_or_path=model_name,
                device=device,
                cache_folder=get_hf_home(),
                local_files_only=local_files_only
            )
            print(f"Initialized SentenceTransformers with model: {model_name}")
            
        except ImportError:
            raise ImportError("sentence-transformers not installed. Run: pip install sentence-transformers")
    
    def _initialize_huggingface(self):
        """Initialize HuggingFace embedder."""
        try:
            # Import the HuggingFace embedding function from pyscrai
            import sys
            import os
            
            # Add the parent directory to sys.path to import pyscrai modules
            parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            if parent_dir not in sys.path:
                sys.path.append(parent_dir)
            
            from pyscrai.embedding.hf_embedding import HuggingFaceEmbedding
            
            hf_config = self.config.embedding.huggingface
            model_name = hf_config.get("model", "sentence-transformers/all-MiniLM-L6-v2")
            api_token_env = hf_config.get("api_token_env", "HF_API_TOKEN")
            
            api_token = get_api_key(api_token_env)
            if not api_token:
                raise ValueError(f"API token not found in environment variable: {api_token_env}")
            
            self.embedder = HuggingFaceEmbedding(
                model_name=model_name,
                api_token=api_token
            )
            print(f"Initialized HuggingFace embedder with model: {model_name}")
            
        except ImportError as e:
            raise ImportError(f"Failed to import HuggingFace embedding: {e}")
    
    def embed_text(self, text: str) -> np.ndarray:
        """Embed a single text string."""
        if self.provider == "sentence_transformers":
            return self.embedder.encode(text, convert_to_numpy=True)
        elif self.provider == "huggingface":
            return self.embedder.embed(text)
    
    def embed_texts(self, texts: List[str]) -> List[np.ndarray]:
        """Embed a list of text strings."""
        if self.provider == "sentence_transformers":
            # SentenceTransformers can handle batch processing efficiently
            embeddings = self.embedder.encode(texts, convert_to_numpy=True)
            return [embedding for embedding in embeddings]
        elif self.provider == "huggingface":
            # Process individually for HuggingFace API
            return [self.embedder.embed(text) for text in texts]
    
    def get_dimension(self) -> int:
        """Get the embedding dimension."""
        if self.provider == "sentence_transformers":
            return self.embedder.get_sentence_embedding_dimension()
        elif self.provider == "huggingface":
            # For HuggingFace, we need to make a test embedding to get dimension
            test_embedding = self.embed_text("test")
            return len(test_embedding)
