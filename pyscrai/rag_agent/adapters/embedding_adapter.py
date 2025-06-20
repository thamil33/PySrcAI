"""Embedding adapter for RAG Agents."""

import numpy as np
from typing import List, Union

from ..config_loader import AgentConfig, get_hf_home, get_api_key


class EmbeddingAdapter:
    """Adapter class to handle different embedding providers."""

    def __init__(self, config: AgentConfig):
        self.config = config
        self.provider = config.embedding.provider
        self.model = config.embedding.model
        self.embedder = None
        # Log config accesses
        self._log_config_access("embedding.provider", self.provider)
        self._log_config_access("embedding.model", self.model)
        self._log_config_access("embedding.device", getattr(config.embedding, 'device', None))
        self._initialize_embedder()

    def _log_config_access(self, key, value):
        try:
            from ..config_access_logger import is_logging_enabled, logger
            if is_logging_enabled():
                logger.info(f"embedding_adapter accessed {key} -> {repr(value)}")
        except Exception:
            pass

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
            
            self.embedder = SentenceTransformer(
                model_name_or_path=self.model,
                device=self.config.embedding.device,
                cache_folder=self.config.embedding.cache_folder or get_hf_home(),
                local_files_only=self.config.embedding.local_files_only,
            )
            print(f"Initialized SentenceTransformers with model: {self.model} on device: {self.config.embedding.device}")

        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. Run: pip install -r requirements-cpu.txt"
            )

    def _initialize_huggingface_api(self):
        """Initialize HuggingFace API embedder (cloud)."""
        try:
            from ...embedding.hf_embedding import HFEmbedder

            # Get API token
            hf_token = get_api_key("HF_API_TOKEN")

            # Initialize with model and fallback options
            self.embedder = HFEmbedder(
                token=hf_token,
                models=self.config.embedding.fallback_models
            )
            print(f"Initialized HuggingFace API with model: {self.model}")

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
