"""Tests for vector store adapters."""

import pytest
import tempfile
import shutil
from unittest.mock import Mock, patch
from pathlib import Path

from pyscrai.config.config import VectorDBConfig
from pyscrai.adapters.vectorstore import ChromaVectorStore, create_vectorstore
from pyscrai.adapters.embeddings.base import BaseEmbedder
from langchain.schema import Document


class MockEmbeddings(BaseEmbedder):
    """Mock embeddings for testing."""
    
    def embed_documents(self, texts):
        return [[1.0, 0.0, 0.0] for _ in texts]
    
    def embed_query(self, text):
        return [1.0, 0.0, 0.0]


@pytest.fixture
def mock_embeddings():
    """Fixture for mock embeddings."""
    return MockEmbeddings()


@pytest.fixture
def vector_config():
    """Fixture for vector database configuration."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield VectorDBConfig(
            persist_directory=temp_dir,
            collection_name="test_collection",
            settings={"hnsw_space": "cosine"}
        )


@pytest.fixture
def test_documents():
    """Fixture for test documents."""
    return [
        Document(
            page_content="This is the first test document about machine learning.",
            metadata={"source": "doc1.txt", "type": "test"}
        ),
        Document(
            page_content="This is the second test document about artificial intelligence.",
            metadata={"source": "doc2.txt", "type": "test"}
        ),
        Document(
            page_content="This is the third test document about deep learning.",
            metadata={"source": "doc3.txt", "type": "test"}
        )
    ]


@patch('pyscrai.adapters.vectorstore.chroma_adapter.Chroma')
def test_chroma_vectorstore_initialization(mock_chroma, vector_config, mock_embeddings):
    """Test ChromaVectorStore initialization."""
    vectorstore = ChromaVectorStore(vector_config, mock_embeddings)
    
    assert vectorstore.config == vector_config
    assert vectorstore.embeddings == mock_embeddings
    mock_chroma.assert_called_once()


@patch('pyscrai.adapters.vectorstore.chroma_adapter.Chroma')
def test_chroma_add_documents(mock_chroma, vector_config, mock_embeddings, test_documents):
    """Test adding documents to ChromaVectorStore."""
    mock_vectorstore = Mock()
    mock_vectorstore.add_documents.return_value = ["doc1", "doc2", "doc3"]
    mock_chroma.return_value = mock_vectorstore
    
    vectorstore = ChromaVectorStore(vector_config, mock_embeddings)
    doc_ids = vectorstore.add_documents(test_documents)
    
    assert doc_ids == ["doc1", "doc2", "doc3"]
    mock_vectorstore.add_documents.assert_called_once_with(test_documents)


@patch('pyscrai.adapters.vectorstore.chroma_adapter.Chroma')
def test_chroma_similarity_search(mock_chroma, vector_config, mock_embeddings):
    """Test similarity search in ChromaVectorStore."""
    mock_vectorstore = Mock()
    mock_vectorstore.similarity_search.return_value = [
        Document(page_content="Relevant document", metadata={"source": "test.txt"})
    ]
    mock_chroma.return_value = mock_vectorstore
    
    vectorstore = ChromaVectorStore(vector_config, mock_embeddings)
    results = vectorstore.similarity_search("test query", k=5)
    
    assert len(results) == 1
    assert results[0].page_content == "Relevant document"
    mock_vectorstore.similarity_search.assert_called_once_with(
        query="test query",
        k=5,
        filter=None
    )


@patch('pyscrai.adapters.vectorstore.chroma_adapter.Chroma')
def test_chroma_similarity_search_with_score(mock_chroma, vector_config, mock_embeddings):
    """Test similarity search with scores in ChromaVectorStore."""
    mock_vectorstore = Mock()
    mock_vectorstore.similarity_search_with_score.return_value = [
        (Document(page_content="Relevant document", metadata={"source": "test.txt"}), 0.95)
    ]
    mock_chroma.return_value = mock_vectorstore
    
    vectorstore = ChromaVectorStore(vector_config, mock_embeddings)
    results = vectorstore.similarity_search_with_score("test query", k=3)
    
    assert len(results) == 1
    doc, score = results[0]
    assert doc.page_content == "Relevant document"
    assert score == 0.95
    mock_vectorstore.similarity_search_with_score.assert_called_once_with(
        query="test query",
        k=3,
        filter=None
    )


@patch('pyscrai.adapters.vectorstore.chroma_adapter.Chroma')
def test_chroma_delete(mock_chroma, vector_config, mock_embeddings):
    """Test deleting documents from ChromaVectorStore."""
    mock_vectorstore = Mock()
    mock_vectorstore.delete.return_value = None
    mock_chroma.return_value = mock_vectorstore
    
    vectorstore = ChromaVectorStore(vector_config, mock_embeddings)
    success = vectorstore.delete(["doc1", "doc2"])
    
    assert success is True
    mock_vectorstore.delete.assert_called_once_with(ids=["doc1", "doc2"])


@patch('pyscrai.adapters.vectorstore.chroma_adapter.Chroma')
def test_chroma_clear(mock_chroma, vector_config, mock_embeddings):
    """Test clearing ChromaVectorStore."""
    mock_collection = Mock()
    mock_collection.get.return_value = {"ids": ["doc1", "doc2", "doc3"]}
    mock_collection.delete.return_value = None
    
    mock_vectorstore = Mock()
    mock_vectorstore._collection = mock_collection
    mock_chroma.return_value = mock_vectorstore
    
    vectorstore = ChromaVectorStore(vector_config, mock_embeddings)
    success = vectorstore.clear()
    
    assert success is True
    mock_collection.get.assert_called_once()
    mock_collection.delete.assert_called_once_with(ids=["doc1", "doc2", "doc3"])


@patch('pyscrai.adapters.vectorstore.chroma_adapter.Chroma')
def test_chroma_collection_info(mock_chroma, vector_config, mock_embeddings):
    """Test getting collection info from ChromaVectorStore."""
    mock_collection = Mock()
    mock_collection.count.return_value = 42
    
    mock_vectorstore = Mock()
    mock_vectorstore._collection = mock_collection
    mock_chroma.return_value = mock_vectorstore
    
    vectorstore = ChromaVectorStore(vector_config, mock_embeddings)
    info = vectorstore.get_collection_info()
    
    assert info["name"] == "test_collection"
    assert info["count"] == 42
    assert info["persist_directory"] == vector_config.persist_directory
    assert "MockEmbeddings" in info["embedding_function"]


def test_create_vectorstore_factory(vector_config, mock_embeddings):
    """Test the vectorstore factory function."""
    with patch('langchain.vectorstores.Chroma'):
        vectorstore = create_vectorstore(vector_config, mock_embeddings)
        assert isinstance(vectorstore, ChromaVectorStore)
