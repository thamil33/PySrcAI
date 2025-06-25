"""Tests for document ingestion pipeline."""

import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch

from pyscrai.agentica.config.config import AgentConfig, ChunkingConfig, VectorDBConfig, EmbeddingConfig
from pyscrai.agentica.ingestion.pipeline import IngestionPipeline
from pyscrai.agentica.ingestion.loaders import TextLoader, JSONLoader, DirectoryLoader
from pyscrai.agentica.ingestion.chunkers import TextChunker, SemanticChunker
from langchain.schema import Document


@pytest.fixture
def mock_config():
    """Mock agent configuration for testing."""
    return AgentConfig(
        chunking=ChunkingConfig(chunk_size=100, overlap=20),
        vectordb=VectorDBConfig(
            persist_directory="./test_vector_storage",
            collection_name="test_collection"
        ),        embedding=EmbeddingConfig(
            provider="local_sentencetransformers", 
            model="all-MiniLM-L6-v2",
            device="cpu"
        )
    )


@pytest.fixture
def test_documents():
    """Test documents for ingestion."""
    return [
        Document(
            page_content="This is a test document about machine learning and artificial intelligence.",
            metadata={"source": "test1.txt", "type": "test"}
        ),
        Document(
            page_content="This is another test document about deep learning and neural networks.",
            metadata={"source": "test2.txt", "type": "test"}
        )
    ]


class TestTextLoader:
    """Tests for TextLoader."""

    def test_text_loader(self):
        """Test loading a text file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("This is a test document.")
            temp_path = f.name

        try:
            loader = TextLoader(temp_path)
            documents = loader.load()
            
            assert len(documents) == 1
            assert documents[0].page_content == "This is a test document."
            assert documents[0].metadata["source"] == temp_path
            assert documents[0].metadata["file_type"] == "text"
        finally:
            Path(temp_path).unlink()


class TestJSONLoader:
    """Tests for JSONLoader."""

    def test_json_loader_single_object(self):
        """Test loading a single JSON object."""
        test_data = {"title": "Test", "content": "This is test content"}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_data, f)
            temp_path = f.name

        try:
            loader = JSONLoader(temp_path, content_key="content", metadata_keys=["title"])
            documents = loader.load()
            
            assert len(documents) == 1
            assert documents[0].page_content == "This is test content"
            assert documents[0].metadata["title"] == "Test"
            assert documents[0].metadata["file_type"] == "json"
        finally:
            Path(temp_path).unlink()

    def test_json_loader_array(self):
        """Test loading a JSON array."""
        test_data = [
            {"title": "Doc 1", "content": "First document"},
            {"title": "Doc 2", "content": "Second document"}
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_data, f)
            temp_path = f.name

        try:
            loader = JSONLoader(temp_path, content_key="content", metadata_keys=["title"])
            documents = loader.load()
            
            assert len(documents) == 2
            assert documents[0].page_content == "First document"
            assert documents[0].metadata["title"] == "Doc 1"
            assert documents[1].page_content == "Second document"
            assert documents[1].metadata["title"] == "Doc 2"
        finally:
            Path(temp_path).unlink()


class TestDirectoryLoader:
    """Tests for DirectoryLoader."""

    def test_directory_loader(self):
        """Test loading files from a directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test files
            (Path(temp_dir) / "test1.txt").write_text("First document")
            (Path(temp_dir) / "test2.md").write_text("# Second document")
            (Path(temp_dir) / "ignore.log").write_text("Log file")
            
            loader = DirectoryLoader(
                temp_dir,
                file_extensions=['.txt', '.md']
            )
            documents = loader.load()
            
            assert len(documents) == 2
            sources = [doc.metadata["source"] for doc in documents]
            assert any("test1.txt" in source for source in sources)
            assert any("test2.md" in source for source in sources)
            assert not any("ignore.log" in source for source in sources)


class TestTextChunker:
    """Tests for TextChunker."""

    def test_text_chunker(self):
        """Test basic text chunking."""
        config = ChunkingConfig(chunk_size=50, overlap=10)
        chunker = TextChunker(config)
        
        long_text = "This is a very long document that should be split into multiple chunks. " * 10
        document = Document(page_content=long_text, metadata={"source": "test.txt"})
        
        chunks = chunker.chunk_documents([document])
        
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk.page_content) <= 50 + 10  # Allow for overlap
            assert "chunk_index" in chunk.metadata
            assert "total_chunks" in chunk.metadata
            assert chunk.metadata["chunking_strategy"] == "text"


class TestSemanticChunker:
    """Tests for SemanticChunker."""

    def test_semantic_chunker_markdown(self):
        """Test semantic chunking for markdown."""
        config = ChunkingConfig(chunk_size=200, overlap=20)
        chunker = SemanticChunker(config)
        
        markdown_text = """# Header 1
This is content under header 1.

## Subheader 1.1
This is content under subheader 1.1.

# Header 2
This is content under header 2."""
        
        document = Document(
            page_content=markdown_text,
            metadata={"source": "test.md", "file_type": "markdown"}
        )
        
        chunks = chunker.chunk_documents([document])
        
        assert len(chunks) > 1
        for chunk in chunks:
            assert chunk.metadata["chunking_strategy"] == "semantic"


@patch('pyscrai.agentica.ingestion.pipeline.create_chunker')
@patch('pyscrai.agentica.ingestion.pipeline.create_vectorstore')
@patch('pyscrai.agentica.ingestion.pipeline.create_embedder')
class TestIngestionPipeline:
    """Tests for IngestionPipeline."""
    
    def test_pipeline_initialization(self, mock_create_embedder, mock_create_vectorstore, mock_create_chunker, mock_config):
        """Test pipeline initialization."""
        mock_embedder = Mock()
        mock_vectorstore = Mock()
        mock_chunker = Mock()
        
        mock_create_embedder.return_value = mock_embedder
        mock_create_vectorstore.return_value = mock_vectorstore
        mock_create_chunker.return_value = mock_chunker
        
        pipeline = IngestionPipeline(mock_config)
        
        assert pipeline.config == mock_config
        assert pipeline.embeddings == mock_embedder
        assert pipeline.vectorstore == mock_vectorstore
        assert pipeline.chunker == mock_chunker
        
        mock_create_embedder.assert_called_once_with(mock_config.embedding)
        mock_create_vectorstore.assert_called_once_with(mock_config.vectordb, mock_embedder)
        mock_create_chunker.assert_called_once_with(mock_config.chunking)

    def test_ingest_documents(self, mock_create_embedder, mock_create_vectorstore, mock_create_chunker, mock_config, test_documents):
        """Test document ingestion."""
        mock_embedder = Mock()
        mock_vectorstore = Mock()
        mock_chunker = Mock()
        mock_vectorstore.add_documents.return_value = ["doc1", "doc2"]
        
        mock_create_embedder.return_value = mock_embedder
        mock_create_vectorstore.return_value = mock_vectorstore
        mock_create_chunker.return_value = mock_chunker
        
        pipeline = IngestionPipeline(mock_config)
        doc_ids = pipeline.ingest_documents(test_documents)
        
        assert doc_ids == ["doc1", "doc2"]
        mock_vectorstore.add_documents.assert_called_once()

    def test_search(self, mock_create_embedder, mock_create_vectorstore, mock_create_chunker, mock_config):
        """Test document search."""
        mock_embedder = Mock()
        mock_vectorstore = Mock()
        mock_chunker = Mock()
        mock_vectorstore.similarity_search.return_value = [
            Document(page_content="Result", metadata={"source": "test.txt"})
        ]
        
        mock_create_embedder.return_value = mock_embedder
        mock_create_vectorstore.return_value = mock_vectorstore
        mock_create_chunker.return_value = mock_chunker
        
        pipeline = IngestionPipeline(mock_config)
        results = pipeline.search("test query")
        
        assert len(results) == 1
        assert results[0].page_content == "Result"
        mock_vectorstore.similarity_search.assert_called_once_with(query="test query", k=5, filter=None)

    def test_clear_vectorstore(self, mock_create_embedder, mock_create_vectorstore, mock_create_chunker, mock_config):
        """Test clearing vector store."""
        mock_embedder = Mock()
        mock_vectorstore = Mock()
        mock_chunker = Mock()
        mock_vectorstore.clear.return_value = True
        
        mock_create_embedder.return_value = mock_embedder
        mock_create_vectorstore.return_value = mock_vectorstore
        mock_create_chunker.return_value = mock_chunker
        
        pipeline = IngestionPipeline(mock_config)
        success = pipeline.clear_vectorstore()
        
        assert success is True
        mock_vectorstore.clear.assert_called_once()
