"""OpenRouter API LLM Adapter."""

import json
import os
import time
from typing import Any, Dict, Iterator, List, Optional, Union, Mapping
import requests
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.outputs import GenerationChunk
from pydantic import Field, SecretStr


class OpenRouterLLM(LLM):
    """Comprehensive OpenRouter API LLM adapter with robust error handling and LangGraph integration."""

    api_key: SecretStr = Field(default_factory=lambda: SecretStr(""))
    model: str = Field(default="mistralai/mistral-small-3.1-24b-instruct:free")
    base_url: str = Field(default="https://openrouter.ai/api/v1")
    max_tokens: Optional[int] = Field(default=500)
    temperature: float = Field(default=0.7)
    top_p: Optional[float] = Field(default=None)
    top_k: Optional[int] = Field(default=None)
    frequency_penalty: Optional[float] = Field(default=None)
    presence_penalty: Optional[float] = Field(default=None)
    repetition_penalty: Optional[float] = Field(default=None)
    min_p: Optional[float] = Field(default=None)
    top_a: Optional[float] = Field(default=None)
    seed: Optional[int] = Field(default=None)
    logit_bias: Optional[Dict[str, float]] = Field(default=None)
    user: Optional[str] = Field(default=None)
    
    # OpenRouter-specific parameters
    provider: Optional[Dict[str, Any]] = Field(default=None)
    models: Optional[List[str]] = Field(default=None)
    route: Optional[str] = Field(default=None)
    
    # Application identification (for OpenRouter leaderboards)
    app_name: Optional[str] = Field(default="pyscrai")
    site_url: Optional[str] = Field(default="https://github.com/tyler-richardson/pyscrai_workstation")
    
    # Request configuration
    timeout: int = Field(default=60)
    max_retries: int = Field(default=3)
    retry_delay: float = Field(default=1.0)
    
    class Config:
        """Configuration for this pydantic object."""
        extra = "forbid"

    def __init__(self, **data: Any):
        """Initialize the OpenRouter LLM adapter."""
        super().__init__(**data)
        
        # Set API key from environment if not provided
        if not self.api_key.get_secret_value():
            api_key = os.getenv("OPENROUTER_API_KEY")
            if not api_key:
                raise ValueError(
                    "OpenRouter API key must be provided either as 'api_key' parameter "
                    "or set in OPENROUTER_API_KEY environment variable"
                )
            self.api_key = SecretStr(api_key)

    @property
    def _llm_type(self) -> str:
        """Return the type of LLM."""
        return "openrouter"

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
            "repetition_penalty": self.repetition_penalty,
        }

    def _get_headers(self) -> Dict[str, str]:
        """Get request headers for OpenRouter API."""
        headers = {
            "Authorization": f"Bearer {self.api_key.get_secret_value()}",
            "Content-Type": "application/json",
        }
        
        # Add OpenRouter-specific headers for leaderboards
        if self.app_name:
            headers["HTTP-Referer"] = self.site_url or ""
            headers["X-Title"] = self.app_name
            
        return headers

    def _prepare_request_payload(self, prompt: str, stop: Optional[List[str]] = None) -> Dict[str, Any]:
        """Prepare the request payload for OpenRouter API."""
        # Convert prompt to messages format (required by OpenRouter)
        messages = [{"role": "user", "content": prompt}]
        
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "stream": False,
        }
        
        # Add optional parameters
        if self.top_p is not None:
            payload["top_p"] = self.top_p
        if self.top_k is not None:
            payload["top_k"] = self.top_k
        if self.frequency_penalty is not None:
            payload["frequency_penalty"] = self.frequency_penalty
        if self.presence_penalty is not None:
            payload["presence_penalty"] = self.presence_penalty
        if self.repetition_penalty is not None:
            payload["repetition_penalty"] = self.repetition_penalty
        if self.min_p is not None:
            payload["min_p"] = self.min_p
        if self.top_a is not None:
            payload["top_a"] = self.top_a
        if self.seed is not None:
            payload["seed"] = self.seed
        if self.logit_bias is not None:
            payload["logit_bias"] = self.logit_bias
        if self.user is not None:
            payload["user"] = self.user
        if stop is not None:
            payload["stop"] = stop
            
        # Add OpenRouter-specific parameters
        if self.provider is not None:
            payload["provider"] = self.provider
        if self.models is not None:
            payload["models"] = self.models
        if self.route is not None:
            payload["route"] = self.route
            
        return payload

    def _make_request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Make a request to OpenRouter API with retry logic."""
        url = f"{self.base_url}/chat/completions"
        headers = self._get_headers()
        
        last_exception = None
        for attempt in range(self.max_retries + 1):
            try:
                response = requests.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout
                )
                response.raise_for_status()
                return response.json()
                
            except requests.exceptions.RequestException as e:
                last_exception = e
                if attempt < self.max_retries:
                    wait_time = self.retry_delay * (2 ** attempt)  # Exponential backoff
                    time.sleep(wait_time)
                    continue
                break
                
        # If we get here, all retries failed
        if last_exception:
            raise last_exception
        else:
            raise RuntimeError("Failed to make request after all retries")

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Call the OpenRouter API."""
        payload = self._prepare_request_payload(prompt, stop)
        
        try:
            response_data = self._make_request(payload)
            
            # Extract the response text
            if "choices" in response_data and len(response_data["choices"]) > 0:
                choice = response_data["choices"][0]
                if "message" in choice and "content" in choice["message"]:
                    return choice["message"]["content"]
                elif "text" in choice:
                    return choice["text"]
            
            # Fallback if response format is unexpected
            return str(response_data)
            
        except Exception as e:
            raise RuntimeError(f"OpenRouter API call failed: {str(e)}") from e

    def _stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        """Stream the OpenRouter API response."""
        payload = self._prepare_request_payload(prompt, stop)
        payload["stream"] = True
        
        url = f"{self.base_url}/chat/completions"
        headers = self._get_headers()
        
        try:
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=self.timeout,
                stream=True
            )
            response.raise_for_status()
            
            for line in response.iter_lines():
                if line:
                    line_str = line.decode("utf-8")
                    if line_str.startswith("data: "):
                        data_str = line_str[6:]  # Remove "data: " prefix
                        if data_str.strip() == "[DONE]":
                            break
                        try:
                            data = json.loads(data_str)
                            if "choices" in data and len(data["choices"]) > 0:
                                choice = data["choices"][0]
                                if "delta" in choice and "content" in choice["delta"]:
                                    content = choice["delta"]["content"]
                                    if content:
                                        chunk = GenerationChunk(text=content)
                                        if run_manager:
                                            run_manager.on_llm_new_token(content, chunk=chunk)
                                        yield chunk
                        except json.JSONDecodeError:
                            continue
                            
        except Exception as e:
            raise RuntimeError(f"OpenRouter streaming API call failed: {str(e)}") from e

    @property
    def _supports_stream(self) -> bool:
        """Return whether this LLM supports streaming."""
        return True

    def get_available_models(self) -> List[Dict[str, Any]]:
        """Get list of available models from OpenRouter."""
        url = f"{self.base_url}/models"
        headers = self._get_headers()
        
        try:
            response = requests.get(url, headers=headers, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            return data.get("data", [])
        except Exception as e:
            raise RuntimeError(f"Failed to fetch available models: {str(e)}") from e

    def get_generation_info(self, generation_id: str) -> Dict[str, Any]:
        """Get detailed information about a specific generation."""
        url = f"{self.base_url}/generation"
        headers = self._get_headers()
        params = {"id": generation_id}
        
        try:
            response = requests.get(url, headers=headers, params=params, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            raise RuntimeError(f"Failed to fetch generation info: {str(e)}") from e
