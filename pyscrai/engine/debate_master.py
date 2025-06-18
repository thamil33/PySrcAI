"""Game master implementation for philosophical debates."""

from typing import Mapping, Sequence
import dataclasses

from concordia.agents import entity_agent_with_logging
from concordia.associative_memory.basic_associative_memory import AssociativeMemoryBank
from concordia.language_model import language_model
from concordia.typing import prefab as prefab_lib
from concordia.components import agent as actor_components
from concordia.components.game_master import switch_act

@dataclasses.dataclass
class DebateGameMaster(prefab_lib.Prefab):
    """A game master specialized for handling philosophical debates."""

    description: str = 'A game master that oversees philosophical debates and ensures productive discourse.'
    params: Mapping[str, str] = dataclasses.field(
        default_factory=lambda: {
            'name': 'Debate Moderator',
        }
    )
    entities: Sequence[entity_agent_with_logging.EntityAgentWithLogging] = ()

    def build(
        self,
        model: language_model.LanguageModel,
        memory_bank: AssociativeMemoryBank,
    ) -> entity_agent_with_logging.EntityAgentWithLogging:
        """Build the debate game master.

        Args:
            model: The language model to use
            memory_bank: The memory bank for storing debate history

        Returns:
            A configured game master entity
        """
        name = self.params.get('name', 'Debate Moderator')
        player_names = [entity.name for entity in self.entities]

        # Set up the observation component to track debate history
        observation = actor_components.observation.LastNObservations(
            history_length=100,
        )

        # Set up memory component to store debate context
        memory = actor_components.memory.AssociativeMemory(
            memory_bank=memory_bank
        )

        # Components that the game master will use
        components = {
            actor_components.observation.DEFAULT_OBSERVATION_COMPONENT_KEY: observation,
            actor_components.memory.DEFAULT_MEMORY_COMPONENT_KEY: memory,
        }

        # Create the game master entity with SwitchAct component
        game_master = entity_agent_with_logging.EntityAgentWithLogging(
            agent_name=name,
            act_component=switch_act.SwitchAct(
                model=model,
                entity_names=player_names,
                component_order=list(components.keys())
            ),
            context_components=components
        )

        return game_master
