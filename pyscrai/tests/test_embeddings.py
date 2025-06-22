"""Tests for embedding adapters."""

import pytest
from unittest.mock import Mock, patch
import numpy as np

from pyscrai.config.config import EmbeddingConfig
from pyscrai.adapters.embeddings import (
    BaseEmbedder,
    create_embedder,
    SentenceTransformerEmbeddings
)


def test_base_embedder_normalize():
    """Test embedding normalization."""
    class TestEmbedder(BaseEmbedder):
        def embed_documents(self, texts):
            pass
        def embed_query(self, text):
            pass
    
    embedder = TestEmbedder()
    vec = [1.0, 2.0, 2.0]  # Length 3
    normalized = embedder._normalize_embedding(vec)
    assert len(normalized) == 3
    assert abs(sum(x*x for x in normalized) - 1.0) < 1e-6  # Unit length





@pytest.fixture
def mock_st_config():
    """Mock sentence-transformers config."""
    return EmbeddingConfig(
        provider="local_sentencetransformers",
        model="all-MiniLM-L6-v2",
        device="cpu"
    )





@patch('pyscrai.adapters.embeddings.sentence_transformers.SentenceTransformer')
def test_sentence_transformer_embeddings(mock_st, mock_st_config):
    """Test sentence-transformers embeddings."""
    # Mock successful model loading and encoding
    mock_model = Mock()
    mock_model.encode.return_value = np.array([[1.0, 0.0, 0.0]])
    mock_st.return_value = mock_model
    
    embedder = SentenceTransformerEmbeddings(mock_st_config)
    result = embedder.embed_query("test")
    
    assert len(result) == 3
    assert abs(sum(x*x for x in result) - 1.0) < 1e-6


def test_create_embedder(mock_st_config):
    """Test embedding factory (sentence-transformers only)."""
    with patch('sentence_transformers.SentenceTransformer') as mock_st:
        # Mock successful model loading
        mock_model = Mock()
        mock_model.encode.return_value = np.array([[1.0, 0.0, 0.0]])
        mock_st.return_value = mock_model
        # Test sentence-transformers creation
        embedder = create_embedder(mock_st_config)
        assert isinstance(embedder, SentenceTransformerEmbeddings)
    # Test invalid provider
    with pytest.raises(ValueError):
        create_embedder(EmbeddingConfig(provider="invalid"))
