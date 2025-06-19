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
    "Xenova/gte-small",
    "FlagEmbedding",
    "Ember",
    "E5",
]


class HFEmbedder:
    """Embed text using the Hugging Face Inference API."""

    def __init__(self, token: Optional[str] = None, models: Optional[Iterable[str]] = None) -> None:
        self.token = token or os.getenv("HF_API_TOKEN")
        if not self.token:
            raise ValueError("HF_API_TOKEN not provided")
        self.models = list(models) if models else DEFAULT_MODELS

    def __call__(self, text: str) -> np.ndarray:
        headers = {"Authorization": f"Bearer {self.token}"}
        for model in self.models:
            api_url = f"https://api-inference.huggingface.co/models/{model}"
            try:
                payload = {
                    "inputs": text,
                    "options": {"wait_for_model": True},
                }
                response = requests.post(api_url, headers=headers, json=payload)
                response.raise_for_status()
                return np.array(response.json()[0])
            except requests.exceptions.RequestException as exc:
                logging.error("Error with model %s: %s", model, exc)
                time.sleep(2)
        logging.error("All models failed. Returning zero vector.")
        return np.zeros(768)


# Backwards compatible helper
embedder = HFEmbedder()
