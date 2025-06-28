"""Orchestrator prefab that wraps the dialogic & dramaturgic GameMaster."""

from collections.abc import Mapping, Sequence
import dataclasses

from concordia.agents import entity_agent_with_logging
from concordia.associative_memory import basic_associative_memory
from concordia.language_model import language_model
from concordia.prefabs.game_master import dialogic_and_dramaturgic as dd_gm
from concordia.typing import prefab as prefab_lib


@dataclasses.dataclass
class Orchestrator(prefab_lib.Prefab):
    """A thin wrapper around the dialogic & dramaturgic GameMaster."""

    description: str = "Game master orchestrating the debate using scenes."
    params: Mapping[str, object] = dataclasses.field(
        default_factory=lambda: {
            "name": "Debate_Orchestrator",
            "scenes": (),
        }
    )
    entities: Sequence[entity_agent_with_logging.EntityAgentWithLogging] = ()

    def build(
        self,
        model: language_model.LanguageModel,
        memory_bank: basic_associative_memory.AssociativeMemoryBank,
    ) -> entity_agent_with_logging.EntityAgentWithLogging:
        gm = dd_gm.GameMaster(
            params={
                "name": self.params.get("name", "Debate_Orchestrator"),
                "scenes": self.params.get("scenes", ()),
            },
            entities=self.entities,
        )
        return gm.build(model, memory_bank)
