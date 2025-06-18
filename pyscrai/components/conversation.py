"""Component for handling conversations between entities."""

from collections.abc import Sequence
import dataclasses
from typing import List, Optional, Callable

from concordia.components.agent import action_spec_ignored
from concordia.document import interactive_document
from concordia.language_model import language_model
from concordia.typing import entity as entity_lib
from concordia.typing import entity_component

@dataclasses.dataclass
class DialogueTurn:
    """A single turn in a conversation."""
    speaker: str
    message: str
    timestamp: Optional[str] = None

class Conversation(action_spec_ignored.ActionSpecIgnored, entity_component.ComponentWithLogging):
    """Manages a conversation between multiple participants."""

    def __init__(
        self,
        model: language_model.LanguageModel,
        participants: Sequence,
        topic: str,
        context: str,
        max_turns: int = 10,
        pre_act_label: str = '\nConversation',
        logging_channel: Optional[Callable[[dict], None]] = None
    ):
        super().__init__(
            pre_act_label=pre_act_label
        )
        self.model = model
        self.participants = list(participants)
        self.topic = topic
        self.context = context
        self.max_turns = max_turns
        self._dialogue = []  # List[DialogueTurn]
        self.current_turn = 0
        self._current_speaker_idx = 0
        self._logging_channel = logging_channel

    @property
    def dialogue(self):
        return self._dialogue

    def _get_conversation_history(self) -> str:
        history = []
        for turn in self._dialogue:
            history.append(f"{turn.speaker}: {turn.message}")
        return "\n".join(history)

    def _get_next_speaker(self):
        speaker = self.participants[self._current_speaker_idx]
        self._current_speaker_idx = (self._current_speaker_idx + 1) % len(self.participants)
        return speaker

    def _generate_prompt(self, current_speaker) -> str:
        history = self._get_conversation_history()
        prompt = interactive_document.InteractiveDocument()
        prompt.statement(f"Context: {self.context}")
        prompt.statement(f"Topic: {self.topic}")
        prompt.statement("\nPrevious conversation:")
        if history:
            prompt.statement(history)
        prompt.statement(f"\n{current_speaker.name} is a {current_speaker.description}.")
        prompt.open_question(
            f"What would {current_speaker.name} say next in this philosophical conversation? "
            "Consider their personality, perspective, and previous points made in the conversation. "
            "Response should express their unique viewpoint on the topic while engaging with previous statements."
        )
        return prompt.view().text()

    def _make_pre_act_value(self) -> str:
        """Generate a value for pre_act based on the current state of the conversation."""
        if self.current_turn >= self.max_turns:
            return "The conversation has reached its maximum number of turns."

        current_speaker = self._get_next_speaker()
        prompt = self._generate_prompt(current_speaker)
        response = self.model.complete(prompt)

        if response:
            self._dialogue.append(DialogueTurn(
                speaker=current_speaker.name,
                message=response
            ))
            self.current_turn += 1

            if self._logging_channel:
                self._logging_channel({
                    'speaker': current_speaker.name,
                    'turn': self.current_turn,
                    'message': response,
                    'prompt': prompt,
                })

        return response

    def pre_act(self, action_spec: Optional[entity_lib.ActionSpec] = None) -> str:
        # Delegate to _make_pre_act_value to avoid duplicate logic
        return self._make_pre_act_value()

    def pre_observe(self, observation: str) -> None:
        pass

    def update(self) -> None:
        pass
