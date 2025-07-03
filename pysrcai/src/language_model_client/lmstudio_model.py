"""Language Model that uses LMStudio local server API."""

import logging
import inspect
from collections.abc import Collection, Mapping, Sequence
from typing import Any

from pysrcai.src.language_model_client import language_model

from pysrcai.src.utils import sampling
from pysrcai.src.utils import measurements as measurements_lib
from openai import OpenAI
from typing_extensions import override


_MAX_MULTIPLE_CHOICE_ATTEMPTS = 20


class LMStudioLanguageModel(language_model.LanguageModel):
  """Language Model that uses LMStudio local server API."""

  def __init__(
      self,
      model_name: str = "local-model",
      *,
      base_url: str = "http://localhost:1234/v1",
      measurements: measurements_lib.Measurements | None = None,
      channel: str = language_model.DEFAULT_STATS_CHANNEL,
      verbose_logging: bool = False,
  ):
    """Initializes the instance.

    Args:
      model_name: The local model name (can be anything for LMStudio).
      base_url: The LMStudio server URL (default: http://localhost:1234/v1).
      measurements: The measurements object to log usage statistics to.
      channel: The channel to write the statistics to.
      verbose_logging: If True, log detailed information about each LLM call.
    """
    self._model_name = model_name
    self._measurements = measurements
    self._channel = channel
    self._verbose_logging = verbose_logging
    self._logger = logging.getLogger(f"concordia.lmstudio.{model_name}")

    # Configure logging level based on verbose setting
    if verbose_logging:
      self._logger.setLevel(logging.INFO)
      if not self._logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '[%(asctime)s] %(name)s: %(message)s',
            datefmt='%H:%M:%S'
        )
        handler.setFormatter(formatter)
        self._logger.addHandler(handler)
        self._logger.propagate = False

    # Use OpenAI client with LMStudio endpoint (no API key needed)
    self._client = OpenAI(
        api_key="not-needed",
        base_url=base_url
    )

  def _extract_entity_context(self, frame):
    """Extract entity/agent context from call stack."""
    current_frame = frame
    for _ in range(10):  # Look up to 10 frames back
      if current_frame is None:
        break

      local_vars = current_frame.f_locals

      # Look for common entity/agent variables
      for var_name in ['self', 'entity', 'agent', 'character']:
        if var_name in local_vars:
          obj = local_vars[var_name]
          if hasattr(obj, '_name'):
            return f"{obj.__class__.__name__}({obj._name})"
          elif hasattr(obj, 'name'):
            return f"{obj.__class__.__name__}({obj.name})"
          elif hasattr(obj, '__class__') and 'Entity' in obj.__class__.__name__:
            return f"{obj.__class__.__name__}"

      current_frame = current_frame.f_back

    return None

  @override
  def sample_text(
      self,
      prompt: str,
      *,
      max_tokens: int = language_model.DEFAULT_MAX_TOKENS,
      terminators: Collection[str] = language_model.DEFAULT_TERMINATORS,
      temperature: float = language_model.DEFAULT_TEMPERATURE,
      timeout: float = language_model.DEFAULT_TIMEOUT_SECONDS,
      seed: int | None = None,
  ) -> str:
    """Samples text from the LMStudio local model."""

    # Get caller context for verbose logging
    if self._verbose_logging:
      caller_frame = inspect.currentframe().f_back
      caller_info = f"{caller_frame.f_code.co_filename}:{caller_frame.f_lineno}"
      entity_context = self._extract_entity_context(caller_frame)

      self._logger.info(
          f"LLM Call from {entity_context or 'Unknown'} | "
          f"Caller: {caller_info} | "
          f"Model: {self._model_name} (LMStudio) | "
          f"Prompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}"
      )

    messages = [{"role": "user", "content": prompt}]

    try:
      response = self._client.chat.completions.create(
          model=self._model_name,
          messages=messages,
          max_tokens=max_tokens,
          temperature=temperature,
          timeout=timeout,
          stop=list(terminators) if terminators else None,
      )
    except Exception as e:
      self._logger.error(f"LMStudio API call failed: {e}")
      raise

    result = response.choices[0].message.content

    # Handle terminators
    if terminators:
      for terminator in terminators:
        if terminator in result:
          result = result.split(terminator)[0]

    # Log measurements if available
    if self._measurements is not None:
      # LMStudio might not return usage stats, so we estimate
      prompt_tokens = len(prompt.split()) * 1.3  # Rough estimate
      completion_tokens = len(result.split()) * 1.3
      self._measurements.publish_datum(
          channel=self._channel,
          datum={'prompt_tokens': prompt_tokens,
                 'completion_tokens': completion_tokens,
                 'model': self._model_name})

    return result

  @override
  def sample_choice(
      self,
      prompt: str,
      responses: Sequence[str],
      *,
      seed: int | None = None,
  ) -> tuple[int, str, Mapping[str, Any]]:
    """Samples a response from those available using multiple choice."""
    question = (
        prompt +
        '\nRespond EXACTLY with one of the following options:\n' +
        '\n'.join(f'{i}: {response}' for i, response in enumerate(responses))
    )

    for _ in range(_MAX_MULTIPLE_CHOICE_ATTEMPTS):
      # Use sample_text which already has logging
      answer = self.sample_text(question, seed=seed)

      # Try to parse the choice
      for i, response in enumerate(responses):
        if str(i) in answer[:10]:  # Look for the number in first part of answer
          return i, response, {'answer': answer}

      # If parsing fails, try fuzzy matching
      answer_lower = answer.lower()
      for i, response in enumerate(responses):
        if response.lower() in answer_lower:
          return i, response, {'answer': answer}

    # If all attempts fail, return random choice
    choice_idx = sampling.sample_from_scores([1.0] * len(responses), seed=seed)
    return choice_idx, responses[choice_idx], {'answer': answer, 'fallback': True}
