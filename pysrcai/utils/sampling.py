"""Helper functions for language model sampling.
"""

import re
import random
from typing import Sequence

def _extract_parenthesized_choice(sample: str):
  """Given text formatted as 'lorum(a)ipsum', return 'a'."""
  match = re.search(r'\(?(\w)\)', sample)
  if match:
    return match.group(1)
  else:
    return None


def extract_choice_response(sample: str) -> str | None:
  """Given a sample such as "a", "a)", or "foo(a)bar, return the choice."""
  if len(sample) == 1:
    # i.e. this would be a sample such as "a"
    return sample
  elif len(sample) == 2:
    # i.e. this would be a sample such as "a)"
    return sample[0]
  else:
    # extract a substring like "(a)" wherever it may be in a longer string
    return _extract_parenthesized_choice(sample)


def dynamically_adjust_temperature(
    attempts: int,
    max_attempts: int,
) -> float:
  """Adjusts choice sampling temperature based on number of attempts so far."""
  # Increase temperature after the first failed attempt.
  temperature = 0.0
  if attempts > 1 and attempts < (max_attempts / 2.0):
    temperature = 0.5
  elif attempts > (max_attempts / 2.0):
    temperature = 0.75
  return temperature


def sample_from_scores(scores: Sequence[float], seed: int | None = None) -> int:
  """Sample an index from a sequence of scores using weighted random sampling.
  
  Args:
    scores: A sequence of numeric scores/weights.
    seed: Optional random seed for reproducible sampling.
    
  Returns:
    The index of the sampled item.
  """
  if seed is not None:
    random.seed(seed)
  
  # Handle edge cases
  if not scores:
    return 0
  if len(scores) == 1:
    return 0
  
  # Convert to weights (handle negative scores by shifting)
  min_score = min(scores)
  if min_score < 0:
    adjusted_scores = [score - min_score for score in scores]
  else:
    adjusted_scores = list(scores)
  
  # If all scores are zero after adjustment, use uniform distribution
  total = sum(adjusted_scores)
  if total == 0:
    return random.randint(0, len(scores) - 1)
  
  # Weighted random sampling
  cumulative_weights = []
  cumulative_sum = 0
  for weight in adjusted_scores:
    cumulative_sum += weight
    cumulative_weights.append(cumulative_sum)
  
  rand_val = random.uniform(0, total)
  for i, cum_weight in enumerate(cumulative_weights):
    if rand_val <= cum_weight:
      return i
  
  # Fallback (shouldn't reach here)
  return len(scores) - 1
