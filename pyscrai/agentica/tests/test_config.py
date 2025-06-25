from pyscrai.agentica.config import (
    load_config, load_template, list_templates, AgentConfig, ModelConfig, EmbeddingConfig,
    VectorDBConfig, ChunkingConfig, RAGConfig
)
from pathlib import Path
import os
import tempfile
import pytest


def test_load_config_env_interpolation(tmp_path):
    cfg_file = tmp_path / "cfg.yaml"
    cfg_file.write_text(
        """
models:
  model: test-model
system_prompt: ${SYS_PROMPT}
"""
    )
    os.environ["SYS_PROMPT"] = "hello"
    cfg = load_config(str(cfg_file))
    assert isinstance(cfg, AgentConfig)
    assert cfg.system_prompt == "hello"


def test_default_config():
    """Test default configuration values."""
    cfg = AgentConfig()
    assert isinstance(cfg.models, ModelConfig)
    assert isinstance(cfg.embedding, EmbeddingConfig)
    assert isinstance(cfg.vectordb, VectorDBConfig)
    assert isinstance(cfg.chunking, ChunkingConfig)
    assert isinstance(cfg.rag, RAGConfig)
    assert cfg.models.provider == "openrouter"
    assert cfg.embedding.provider == "local_sentencetransformers"
    assert len(cfg.embedding.fallback_models) > 0
    assert cfg.chunking.chunk_size == 512


def test_load_full_config(tmp_path):
    """Test loading a complete configuration file."""
    cfg_file = tmp_path / "full_cfg.yaml"
    cfg_file.write_text(
        """
models:
  provider: lmstudio
  model: local
  model_kwargs:
    temperature: 0.5

embedding:
  provider: local_sentencetransformers
  model: all-MiniLM-L6-v2
  device: cpu
  fallback_models:
    - model1
    - model2

vectordb:
  collection_name: test_collection
  settings:
    hnsw_space: cosine

chunking:
  chunk_size: 256
  overlap: 25

rag:
  top_k: 3
  enable_reranking: true

system_prompt: test prompt
data_paths:
  - path1
  - path2
"""
    )
    cfg = load_config(str(cfg_file))
    assert cfg.models.provider == "lmstudio"
    assert cfg.models.model_kwargs["temperature"] == 0.5
    assert cfg.embedding.provider == "local_sentencetransformers"
    assert len(cfg.embedding.fallback_models) == 2
    assert cfg.vectordb.collection_name == "test_collection"
    assert cfg.chunking.chunk_size == 256
    assert cfg.rag.top_k == 3
    assert cfg.rag.enable_reranking is True
    assert len(cfg.data_paths) == 2


def test_invalid_config_path():
    """Test behavior with nonexistent config file."""
    with pytest.raises(FileNotFoundError):
        load_config("nonexistent.yaml")


def test_empty_config_file(tmp_path):
    """Test loading an empty config file."""
    cfg_file = tmp_path / "empty.yaml"
    cfg_file.write_text("")
    cfg = load_config(str(cfg_file))
    assert isinstance(cfg, AgentConfig)
    # Should use all default values
    assert cfg.models.provider == "openrouter"


def test_load_template():
    """Test loading template configurations."""
    # Test default template
    cfg = load_template()
    assert isinstance(cfg, AgentConfig)
    assert cfg.models.provider == "openrouter"

    # Test local models template
    cfg = load_template("local_models")
    assert isinstance(cfg, AgentConfig)
    assert cfg.models.provider == "lmstudio"
    assert cfg.embedding.provider == "local_sentencetransformers"


def test_list_templates():
    """Test listing available templates."""
    templates = list_templates()
    assert "default" in templates
    assert "local_models" in templates
