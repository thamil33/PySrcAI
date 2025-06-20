
"""Test embedding adapter functionality."""

import os
import pytest
import numpy as np
from dataclasses import field

from pyscrai.rag_agent.adapters.embedding_adapter import EmbeddingAdapter
from pyscrai.rag_agent.config_loader import AgentConfig, EmbeddingConfig, ModelsConfig

@pytest.fixture
def dummy_config_hf(monkeypatch):
    monkeypatch.setenv("HF_API_TOKEN", "dummy_token")
    
    class DummyEmbeddingConfig:
        provider = "huggingface_api"
        model = "BAAI/bge-base-en-v1.5"
        fallback_models = ["BAAI/bge-base-en-v1.5"]
        device = "cpu"
        local_files_only = False
        cache_folder = None
        
    class DummyModelsConfig:
        language_model = "mistral/model"
        
    class DummyConfig:
        models = DummyModelsConfig()
        embedding = DummyEmbeddingConfig()
    
    return DummyConfig()

@pytest.fixture
def dummy_config_st_cpu():
    class DummyEmbeddingConfig:
        provider = "local_sentencetransformers"
        model = "all-MiniLM-L6-v2"
        device = "cpu"
        local_files_only = False
        cache_folder = None
        fallback_models = []
    
    class DummyModelsConfig:
        language_model = "mistral/model"
        
    class DummyConfig:
        models = DummyModelsConfig()
        embedding = DummyEmbeddingConfig()
    
    return DummyConfig()

@pytest.fixture
def dummy_config_st_cuda(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    
    class DummyEmbeddingConfig:
        provider = "local_sentencetransformers"
        model = "all-MiniLM-L6-v2"
        device = "cuda"
        local_files_only = False
        cache_folder = None
        fallback_models = []
    
    class DummyModelsConfig:
        language_model = "mistral/model"
        
    class DummyConfig:
        models = DummyModelsConfig()
        embedding = DummyEmbeddingConfig()
    
    return DummyConfig()

def test_huggingface_api_embedding(monkeypatch, dummy_config_hf):
    # Mock HFEmbedder to avoid real API call
    class MockHFEmbedder:
        embedding_dim = 384
        def __init__(self, token, models=None):
            self.token = token
            self.models = models or []
        def __call__(self, text):
            return np.ones(384)
    
    # Patch the import and environment checks
    import sys
    from unittest.mock import MagicMock
    
    # Mock the embedding module and environment check
    mock_module = MagicMock()
    mock_module.HFEmbedder = MockHFEmbedder
    sys.modules["pyscrai.embedding.hf_embedding"] = mock_module
    monkeypatch.setattr("os.getenv", lambda x, default=None: "dummy_token" if x == "HF_API_TOKEN" else default)
    
    # Now run the test
    adapter = EmbeddingAdapter(dummy_config_hf)
    emb = adapter.embed_text("test")
    assert isinstance(emb, np.ndarray)
    assert emb.shape[0] == 384

def test_sentence_transformers_cpu(monkeypatch, dummy_config_st_cpu):
    # Mock SentenceTransformer to avoid real model load
    class MockSentenceTransformer:
        def __init__(self, model_name_or_path, device=None, cache_folder=None, local_files_only=False):
            self.model_name = model_name_or_path
            self.device = device
            self.cache_folder = cache_folder
            self.local_files_only = local_files_only
            
        def encode(self, text, convert_to_numpy=True):
            return np.ones(384)
            
        def get_sentence_embedding_dimension(self):
            return 384
    
    # Patch the import
    import sys
    from unittest.mock import MagicMock
    
    # Create mock module
    mock_module = MagicMock()
    mock_module.SentenceTransformer = MockSentenceTransformer
    sys.modules["sentence_transformers"] = mock_module
    
    # Now run the test
    adapter = EmbeddingAdapter(dummy_config_st_cpu)
    emb = adapter.embed_text("test")
    assert isinstance(emb, np.ndarray)
    assert emb.shape[0] == 384

def test_sentence_transformers_cuda(monkeypatch, dummy_config_st_cuda):
    # Mock SentenceTransformer to avoid real model load
    class MockSentenceTransformer:
        def __init__(self, model_name_or_path, device=None, cache_folder=None, local_files_only=False):
            self.model_name = model_name_or_path
            self.device = device
            self.cache_folder = cache_folder
            self.local_files_only = local_files_only
            assert device == "cuda", "Expected CUDA device"
            
        def encode(self, text, convert_to_numpy=True):
            return np.ones(384)
            
        def get_sentence_embedding_dimension(self):
            return 384
    
    # Patch the import
    import sys
    from unittest.mock import MagicMock
    
    # Create mock module
    mock_module = MagicMock()
    mock_module.SentenceTransformer = MockSentenceTransformer
    sys.modules["sentence_transformers"] = mock_module
    
    # Now run the test
    adapter = EmbeddingAdapter(dummy_config_st_cuda)
    emb = adapter.embed_text("test")
    assert isinstance(emb, np.ndarray)
    assert emb.shape[0] == 384
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "0"

# This helper function is no longer needed since we mock the modules
# def _has_sentence_transformers():
#     try:
#         import sentence_transformers
#         return True
#     except ImportError:
#         return False