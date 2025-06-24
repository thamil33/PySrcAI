"""Prefab for a moderator game master."""

from collections.abc import Sequence
import dataclasses

from concordia.language_model import language_model
from concordia.clocks.game_clock import clock as clock_lib        
from concordia.typing import entity as entity_lib
from concordia.typing import prefab as prefab_lib
from concordia.associative_memory import basic_associative_memory
from concordia.agents.entity_agent_with_logging import EntityAgentWithLogging
from concordia.environment.deprecated import game_master

class DebateModerator(EntityAgentWithLogging):
    """A simple game master that moderates a turn-based debate."""

    def __init__(
        self,
        *args,
        max_turns: int = 4,
        **kwargs,
    ):
        """Initializes the moderator game master.

        Args:
            *args: Arguments to pass to the base class.
            max_turns: The maximum number of turns for the debate.
            **kwargs: Keyword arguments to pass to the base class.
        """
        super().__init__(*args, **kwargs)
        self._max_turns = max_turns
        self._turn = 0
        self._last_statement = "The debate is now open. The first speaker may begin."

    def pre_act_observation_for_entity(
        self,
        entity: entity_lib.Entity,
    ) -> str | None:
        """Provide an observation to the current speaker."""
        if entity.name != self.current_player.name:
            return None

        if self._turn >= self._max_turns:
            self.terminate_simulation()
            return f"{self.name} declares: The debate has concluded."

        observation = (
            f"Moderator's instruction: It is your turn to speak (Turn {self._turn + 1}/{self._max_turns}).\n"
            f"The previous statement was: '{self._last_statement}'"
        )
        return observation

    def update_state_after_entity_act(
        self,
        entity: entity_lib.Entity,
        action: str,
    ) -> None:
        """Update state after an entity has acted."""
        self._last_statement = f"{entity.name} stated: {action}"
        self._turn += 1


@dataclasses.dataclass
class ModeratorGmPrefab(prefab_lib.Prefab):
    """Prefab for a moderator game master."""

    description: str = "A game master that moderates a turn-based debate."
    params: dict[str, any] = dataclasses.field(
        default_factory=lambda: {
            "name": "Moderator",
            "max_turns": 4,
            "acting_order": "fixed",
            "verbose": True,
        }
    )

    def build(
        self,
        model: language_model.LanguageModel,
        memory: basic_associative_memory,
        clock: clock_lib,
        players: Sequence[entity_lib.Entity],
    ) -> game_master.GameMaster:
        """Builds the game master agent.

        Args:
            model: The language model to use.
            memory: The memory of the game master.
            clock: The clock of the simulation.
            players: The sequence of players in the debate.

        Returns:
            A game master agent.
        """
        name = self.params.get("name")
        max_turns = self.params.get("max_turns")
        acting_order_str = self.params.get("acting_order")
        verbose = self.params.get("verbose")

        acting_order = [p.name for p in players]
        if acting_order_str == "random":
            # This part is not strictly needed for our current use case,
            # but good to have for completeness.
            self.random_state.shuffle(acting_order)

        return DebateModerator(
            model=model,
            memory=memory,
            clock=clock,
            name=name,
            players=players,
            acting_order=acting_order,
            update_time_every_turn=True,
            max_turns=max_turns,
            verbose=verbose,
        )
