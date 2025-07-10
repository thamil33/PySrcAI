"""Mock language model that returns placeholder responses for demos."""

from collections.abc import Collection, Mapping, Sequence
from typing import Any
import random

from pysrcai.llm import language_model
from typing_extensions import override

class NoLanguageModel(language_model.LanguageModel):
    """Mock language model for testing and demonstrations.
    
    Returns formatted mock responses based on the prompt content to simulate
    actual LLM behavior for demos.
    """

    def __init__(self) -> None:
        """Initialize the mock language model."""
        self._mock_responses = {
            "debate_opening": [
                "ACTION: propose\nCONTENT: I strongly support the four-day work week as it improves work-life balance, reduces burnout, and maintains productivity through increased focus and efficiency.",
                "ACTION: propose\nCONTENT: I oppose the four-day work week as it could reduce overall productivity, create scheduling challenges, and potentially decrease customer satisfaction due to reduced availability.",
            ],
            "debate_rebuttal": [
                "ACTION: respond\nCONTENT: Studies show that four-day weeks actually increase productivity through better focus and reduced burnout. Many companies have maintained or improved output with this schedule.",
                "ACTION: respond\nCONTENT: While some studies show benefits, others reveal logistical challenges, especially for customer-facing roles. Not all industries can adapt without service disruptions.",
            ],
            "debate_closing": [
                "ACTION: respond\nCONTENT: The evidence clearly shows four-day work weeks benefit both workers and companies through improved wellbeing, retention, and sustained productivity. It's time for change.",
                "ACTION: respond\nCONTENT: We should approach this cautiously with optional pilot programs rather than mandates. Some industries and workers may prefer traditional schedules.",
            ],
            "moderator": [
                "ACTION: respond\nCONTENT: Both speakers have made compelling points. The proponent emphasizes well-being while the opponent raises practical concerns. Let's continue the discussion.",
                "ACTION: respond\nCONTENT: This phase saw thoughtful rebuttals from both sides. The debate is highlighting both the potential benefits and implementation challenges of this policy.",
            ],
            "moderator_final": [
                "ACTION: respond\nCONTENT: After careful consideration, I declare the Proponent the winner for presenting more evidence-based arguments and addressing the opponent's concerns effectively.",
                "ACTION: respond\nCONTENT: After careful deliberation, the Opponent wins for raising important practical considerations that weren't fully addressed by the Proponent's idealistic view.",
            ],
        }
        
    def _determine_response_type(self, prompt: str) -> str:
        """Determine which type of response to return based on the prompt."""
        prompt_lower = prompt.lower()
        
        if "opening argument" in prompt_lower:
            return "debate_opening"
        elif "respond to your opponent" in prompt_lower:
            return "debate_rebuttal"
        elif "closing summary" in prompt_lower:
            return "debate_closing"
        elif "provide a brief assessment" in prompt_lower:
            return "moderator"
        elif "provide your final judgment" in prompt_lower:
            return "moderator_final"
        else:
            # Default fallback for unknown prompts
            return "debate_rebuttal"

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
        """Generate a mock response based on the prompt content."""
        response_type = self._determine_response_type(prompt)
        responses = self._mock_responses.get(response_type, ["ACTION: respond\nCONTENT: This is a mock response."])
        
        # Use seed if provided, otherwise use random choice
        if seed is not None:
            random.seed(seed)
            
        response = random.choice(responses)
        return response

    @override
    def sample_choice(
        self,
        prompt: str,
        responses: Sequence[str],
        *,
        seed: int | None = None,
    ) -> tuple[int, str, Mapping[str, Any]]:
        """Select from provided choices."""
        # Use seed if provided
        if seed is not None:
            random.seed(seed)
            
        choice_idx = random.randint(0, len(responses) - 1)
        return choice_idx, responses[choice_idx], {}
