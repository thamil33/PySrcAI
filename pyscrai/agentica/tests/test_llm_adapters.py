"""Tests for LLM adapters."""

import pytest
from unittest.mock import Mock, patch, MagicMock
import json
import requests

from pyscrai.agentica.config.config import ModelConfig
from pyscrai.agentica.adapters.llm import (
    OpenRouterLLM,
    LMStudioLLM,
    create_llm
)


class TestOpenRouterLLM:
    """Test OpenRouter LLM adapter."""

    @pytest.fixture
    def mock_config(self):
        """Mock model config for OpenRouter."""
        return ModelConfig(
            provider="openrouter",
            model="mistralai/mistral-small-3.1-24b-instruct:free",
            model_kwargs={
                "temperature": 0.7,
                "max_tokens": 500
            }
        )

    @patch.dict('os.environ', {'OPENROUTER_API_KEY': 'test-api-key'})
    def test_initialization(self, mock_config):
        """Test OpenRouter LLM initialization."""
        llm = OpenRouterLLM(
            model=mock_config.model,
            **mock_config.model_kwargs
        )
        
        assert llm.model == mock_config.model
        assert llm.temperature == 0.7
        assert llm.max_tokens == 500
        assert llm._llm_type == "openrouter"

    def test_initialization_without_api_key(self):
        """Test that initialization fails without API key."""
        with patch.dict('os.environ', {}, clear=True):
            with pytest.raises(ValueError, match="OpenRouter API key must be provided"):
                OpenRouterLLM(model="test-model")

    @patch.dict('os.environ', {'OPENROUTER_API_KEY': 'test-api-key'})
    @patch('requests.post')
    def test_call_success(self, mock_post):
        """Test successful API call."""
        # Mock successful response
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": "Test response"
                }
            }]
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response
        
        llm = OpenRouterLLM(model="test-model")
        result = llm._call("Test prompt")
        
        assert result == "Test response"
        assert mock_post.called
        
        # Verify request was made with correct parameters
        call_args = mock_post.call_args
        assert call_args[1]['json']['model'] == "test-model"
        assert call_args[1]['json']['messages'][0]['content'] == "Test prompt"

    @patch.dict('os.environ', {'OPENROUTER_API_KEY': 'test-api-key'})
    @patch('requests.post')
    def test_call_with_stop_sequences(self, mock_post):
        """Test API call with stop sequences."""
        # Mock successful response
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": "Test response"
                }
            }]
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response
        
        llm = OpenRouterLLM(model="test-model")
        result = llm._call("Test prompt", stop=["END", "STOP"])
        
        # Verify stop sequences were included
        call_args = mock_post.call_args
        assert call_args[1]['json']['stop'] == ["END", "STOP"]

    @patch.dict('os.environ', {'OPENROUTER_API_KEY': 'test-api-key'})
    @patch('requests.post')
    def test_call_retry_on_failure(self, mock_post):
        """Test retry logic on request failure."""
        # Mock first request failure, second success
        mock_response_fail = Mock()
        mock_response_fail.raise_for_status.side_effect = requests.exceptions.RequestException("API Error")
        
        mock_response_success = Mock()
        mock_response_success.json.return_value = {
            "choices": [{
                "message": {
                    "content": "Success after retry"
                }
            }]
        }
        mock_response_success.raise_for_status.return_value = None
        
        mock_post.side_effect = [mock_response_fail, mock_response_success]
        
        llm = OpenRouterLLM(model="test-model", max_retries=1, retry_delay=0.1)
        result = llm._call("Test prompt")
        
        assert result == "Success after retry"
        assert mock_post.call_count == 2

    @patch.dict('os.environ', {'OPENROUTER_API_KEY': 'test-api-key'})
    @patch('requests.post')
    def test_streaming(self, mock_post):
        """Test streaming API calls."""
        # Mock streaming response
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.iter_lines.return_value = [
            b'data: {"choices": [{"delta": {"content": "Hello"}}]}',
            b'data: {"choices": [{"delta": {"content": " world"}}]}',
            b'data: [DONE]'
        ]
        mock_post.return_value = mock_response
        
        llm = OpenRouterLLM(model="test-model")
        chunks = list(llm._stream("Test prompt"))
        
        assert len(chunks) == 2
        assert chunks[0].text == "Hello"
        assert chunks[1].text == " world"

    @patch.dict('os.environ', {'OPENROUTER_API_KEY': 'test-api-key'})
    @patch('requests.get')
    def test_get_available_models(self, mock_get):
        """Test getting available models."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "data": [
                {"id": "model1", "name": "Model 1"},
                {"id": "model2", "name": "Model 2"}
            ]
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        llm = OpenRouterLLM(model="test-model")
        models = llm.get_available_models()
        
        assert len(models) == 2
        assert models[0]["id"] == "model1"


class TestLMStudioLLM:
    """Test LMStudio LLM adapter."""

    def test_initialization(self):
        """Test LMStudio LLM initialization."""
        llm = LMStudioLLM(model="test-model", temperature=0.5)
        
        assert llm.model == "test-model"
        assert llm.model_kwargs["temperature"] == 0.5
        assert llm._llm_type == "lmstudio"

    def test_call_placeholder(self):
        """Test placeholder implementation."""
        llm = LMStudioLLM(model="test-model")
        result = llm._call("Test prompt")
        
        assert "Simulated response for: Test prompt" in result


class TestLLMFactory:
    """Test LLM factory function."""

    def test_create_openrouter_llm(self):
        """Test creating OpenRouter LLM via factory."""
        config = ModelConfig(
            provider="openrouter",
            model="test-model",
            model_kwargs={"temperature": 0.8}
        )
        
        with patch.dict('os.environ', {'OPENROUTER_API_KEY': 'test-key'}):
            llm = create_llm(config)
            
        assert isinstance(llm, OpenRouterLLM)
        assert llm.model == "test-model"
        assert llm.temperature == 0.8

    def test_create_lmstudio_llm(self):
        """Test creating LMStudio LLM via factory."""
        config = ModelConfig(
            provider="lmstudio",
            model="test-model",
            model_kwargs={"temperature": 0.9}
        )
        
        llm = create_llm(config)
        
        assert isinstance(llm, LMStudioLLM)
        assert llm.model == "test-model"

    def test_unsupported_provider(self):
        """Test error for unsupported provider."""
        config = ModelConfig(
            provider="unsupported",
            model="test-model"
        )
        
        with pytest.raises(ValueError, match="Unsupported LLM provider: unsupported"):
            create_llm(config)
