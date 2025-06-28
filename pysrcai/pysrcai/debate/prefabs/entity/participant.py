"""Participant prefab for debate entities."""

from collections.abc import Mapping
import dataclasses

from concordia.agents import entity_agent_with_logging
from concordia.associative_memory import basic_associative_memory
from concordia.associative_memory import formative_memories
from concordia.components import agent as agent_components
from concordia.language_model import language_model
from concordia.typing import prefab as prefab_lib
import numpy as np

DEFAULT_GOAL_COMPONENT_KEY = "goal"
DEFAULT_CONTEXT_COMPONENT_KEY = "context"


@dataclasses.dataclass
class ParticipantEntity(prefab_lib.Prefab):
    """A prefab representing a debate participant."""

    description: str = (
        "An entity representing a debate participant with a specific stance."
    )
    params: Mapping[str, str] = dataclasses.field(
        default_factory=lambda: {
            "name": "Participant",
            "goal": "Present convincing arguments.",
            "context": "General debater.",
            "word_limits": None,
        }
    )

    def build(
        self,
        model: language_model.LanguageModel,
        memory_bank: basic_associative_memory.AssociativeMemoryBank,
    ) -> entity_agent_with_logging.EntityAgentWithLogging:
        """Build the participant agent."""
        agent_name = self.params.get("name")
        goal = self.params.get("goal")
        context = self.params.get("context")
        word_limits = self.params.get("word_limits")

        observation_to_memory = agent_components.observation.ObservationToMemory()
        observation = agent_components.observation.LastNObservations(
            history_length=10,
            pre_act_label="\nRecent statements:"
        )

        try:
            from sentence_transformers import SentenceTransformer
            embedder = SentenceTransformer("all-MiniLM-L6-v2")
            def embed_fn(text: str) -> np.ndarray:
                return embedder.encode(text)
        except Exception:
            def embed_fn(text: str) -> np.ndarray:  # pragma: no cover - fallback
                return np.random.randn(384)

        memory_factory = formative_memories.FormativeMemoryFactory(
            model=model,
            embedder=embed_fn,
            shared_memories=[
                f"Name: {agent_name}",
                f"Goal: {goal}",
                f"Context: {context}",
            ],
        )
        raw_memory_bank = memory_factory._blank_memory_factory_call()

        if word_limits:
            opening = word_limits["opening"]
            rebuttal = word_limits["rebuttal"]
            closing = word_limits["closing"]
            limit_text = (
                f"RESPONSE LIMIT: opening {opening['min']}-{opening['max']} words, "
                f"rebuttal {rebuttal['min']}-{rebuttal['max']} words, "
                f"closing {closing['min']}-{closing['max']} words."
            )
        else:
            limit_text = (
                "RESPONSE LIMIT: keep responses under 200 words for clarity."
            )

        for mem in [
            f"I am {agent_name}.",
            f"My goal: {goal}",
            f"Background: {context}",
            limit_text,
            "Focus on strong arguments and concision.",
        ]:
            raw_memory_bank.add(mem)

        memory = agent_components.memory.AssociativeMemory(raw_memory_bank)

        goal_component = agent_components.constant.Constant(
            state=goal,
            pre_act_label=f"\n{agent_name}'s goal:"
        )
        context_component = agent_components.constant.Constant(
            state=context,
            pre_act_label=f"\nContext for {agent_name}:"
        )

        components = {
            "observation_to_memory": observation_to_memory,
            agent_components.observation.DEFAULT_OBSERVATION_COMPONENT_KEY: observation,
            agent_components.memory.DEFAULT_MEMORY_COMPONENT_KEY: memory,
            DEFAULT_GOAL_COMPONENT_KEY: goal_component,
            DEFAULT_CONTEXT_COMPONENT_KEY: context_component,
        }

        component_order = [
            DEFAULT_CONTEXT_COMPONENT_KEY,
            DEFAULT_GOAL_COMPONENT_KEY,
            agent_components.memory.DEFAULT_MEMORY_COMPONENT_KEY,
            agent_components.observation.DEFAULT_OBSERVATION_COMPONENT_KEY,
        ]

        act_component = agent_components.concat_act_component.ConcatActComponent(
            model=model,
            component_order=component_order,
        )

        return entity_agent_with_logging.EntityAgentWithLogging(
            agent_name=agent_name,
            act_component=act_component,
            context_components=components,
        )
