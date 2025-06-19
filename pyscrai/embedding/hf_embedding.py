"""Hugging Face embedding utilities."""

from __future__ import annotations

import logging
import os
import time
from typing import Iterable, Optional

import numpy as np
import requests
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.ERROR)

DEFAULT_MODELS: list[str] = [
    "BAAI/bge-base-en-v1.5",
    "BAAI/bge-small-en-v1.5", 
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-mpnet-base-v2",
    "intfloat/e5-base-v2",
]


class HFEmbedder:
    """Embed text using the Hugging Face Inference API."""

    def __init__(self, token: Optional[str] = None, models: Optional[Iterable[str]] = None) -> None:
        self.token = token or os.getenv("HF_API_TOKEN")
        if not self.token:
            raise ValueError("HF_API_TOKEN not provided")
        self.models = list(models) if models else DEFAULT_MODELS
        self._embedding_dim = None  # Will be determined from first successful embedding

    @property
    def embedding_dim(self) -> int:
        """Get the embedding dimension. Defaults to 768 if not yet determined."""
        return self._embedding_dim or 768

    def __call__(self, text: str) -> np.ndarray:
        headers = {"Authorization": f"Bearer {self.token}"}
        for model in self.models:
            api_url = f"https://api-inference.huggingface.co/models/{model}"
            payload = {
                "inputs": text,
                "options": {"wait_for_model": True},
            }
            try:
                response = requests.post(api_url, headers=headers, json=payload)
                response.raise_for_status()
                embedding = np.array(response.json())
                
                # Store embedding dimension on first successful call
                if self._embedding_dim is None:
                    self._embedding_dim = len(embedding)
                    
                return embedding
            except requests.exceptions.RequestException as exc:
                logging.error("Error with model %s: %s", model, exc)
                time.sleep(2)
        logging.error("All models failed. Returning zero vector.")
        return np.zeros(self.embedding_dim)


# Backwards compatible helper
embedder = HFEmbedder()
