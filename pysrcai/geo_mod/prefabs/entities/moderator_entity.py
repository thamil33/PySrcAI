"""Prefab for a debate moderator entity."""

from collections.abc import Mapping
import dataclasses

from concordia.agents import entity_agent_with_logging
from concordia.associative_memory import basic_associative_memory
from concordia.associative_memory import formative_memories
from concordia.components import agent as agent_components
import numpy as np
from concordia.language_model import language_model
from concordia.typing import prefab as prefab_lib
from sentence_transformers import SentenceTransformer

DEFAULT_GOAL_COMPONENT_KEY = 'goal'
DEFAULT_CONTEXT_COMPONENT_KEY = 'context'
DEFAULT_JUDGMENT_COMPONENT_KEY = 'judgment_criteria'


@dataclasses.dataclass
class ModeratorEntity(prefab_lib.Prefab):
    """A prefab for a  debate moderator entity.

    This prefab creates a neutral moderator capable of:
    - Facilitating structured debate phases
    - Maintaining diplomatic decorum
    - Providing objective assessment and judgment
    - Declaring winners based on argument quality
    """

    description: str = (
        'An entity representing a neutral debate moderator. '
        'It is defined by its diplomatic experience, neutrality, '
        'and ability to assess argument quality objectively.'
    )
    params: Mapping[str, str] = dataclasses.field(
        default_factory=lambda: {
            'name': ' Moderator',
            'goal': 'Facilitate fair debate and provide objective assessment.',
            'context': 'Experienced neutral representative with diplomatic expertise.',
        }
    )

    def build(
        self,
        model: language_model.LanguageModel,
        memory_bank: basic_associative_memory.AssociativeMemoryBank,
    ) -> entity_agent_with_logging.EntityAgentWithLogging:
        """Build the moderator agent.

        Args:
            model: The language model to use for generating responses.
            memory_bank: The agent's memory bank.

        Returns:
            An entity agent representing the moderator.
        """
        agent_name = self.params.get('name')
        goal = self.params.get('goal')
        context = self.params.get('context')

        # 1. Core Components for Memory and Observation
        observation_to_memory = agent_components.observation.ObservationToMemory()
        observation = agent_components.observation.LastNObservations(
            history_length=20,  # Longer history for debate context
            pre_act_label=f'\nRecent debate events and statements:'
        )
        
        # Create a memory bank with formative memories for the moderator
        embedder = SentenceTransformer("all-MiniLM-L6-v2")

        def simple_embedder(text: str) -> np.ndarray:
            return embedder.encode(text)
            
        # Create memory bank using FormativeMemoryFactory
        memory_factory = formative_memories.FormativeMemoryFactory(
            model=model,
            embedder=simple_embedder,
            shared_memories=[
                f"Name: {agent_name}",
                f"Role: {goal}",
                f"Background: {context}",
                "Diplomatic Protocol: Maintain neutrality and fairness",
                "Assessment Criteria: Logic, evidence, persuasiveness, international law",
            ],
        )
        
        # Create the raw memory bank
        raw_memory_bank = memory_factory._blank_memory_factory_call()
        
        # Add the moderator's core memories to the raw bank
        for mem in [
            f"I am {agent_name}, a neutral representative.",
            f"My role is to {goal}",
            f"Background: {context}",
            "I must maintain strict neutrality throughout the debate.",
            "I will assess arguments based on logic, evidence, persuasiveness, and adherence to international law.",
            "At the end of the debate, I must declare a winner with detailed reasoning.",
        ]:
            raw_memory_bank.add(mem)
            
        # Wrap the memory bank in an AssociativeMemory component
        memory = agent_components.memory.AssociativeMemory(raw_memory_bank)

        # 2. Moderator-Specific Components
        goal_component = agent_components.constant.Constant(
            state=goal,
            pre_act_label=f'\n{agent_name}\'s role:'
        )
        
        context_component = agent_components.constant.Constant(
            state=context,
            pre_act_label=f'\nBackground for {agent_name}:'
        )
        
        # Add judgment criteria component
        judgment_criteria = (
            "Assessment Criteria for Debate Winner:\n"
            "1. Logical Consistency: Are arguments internally coherent?\n"
            "2. Factual Evidence: Are claims supported by verifiable facts?\n"
            "3. Persuasiveness: How compelling is the overall case?\n"
            "4. International Law: Do arguments align with established legal principles?\n"
            "5. Diplomatic Merit: Are solutions practical and achievable?\n"
            "You must weigh all these factors and declare a clear winner with detailed reasoning."
        )
        
        judgment_component = agent_components.constant.Constant(
            state=judgment_criteria,
            pre_act_label=f'\nJudgment criteria:'
        )

        # 3. Assemble the Agent's "Mind"
        components_of_agent = {
            'observation_to_memory': observation_to_memory,
            agent_components.observation.DEFAULT_OBSERVATION_COMPONENT_KEY: observation,
            agent_components.memory.DEFAULT_MEMORY_COMPONENT_KEY: memory,
            DEFAULT_GOAL_COMPONENT_KEY: goal_component,
            DEFAULT_CONTEXT_COMPONENT_KEY: context_component,
            DEFAULT_JUDGMENT_COMPONENT_KEY: judgment_component,
        }

        # The order determines how the final prompt is constructed.
        component_order = [
            DEFAULT_CONTEXT_COMPONENT_KEY,
            DEFAULT_GOAL_COMPONENT_KEY,
            DEFAULT_JUDGMENT_COMPONENT_KEY,
            agent_components.memory.DEFAULT_MEMORY_COMPONENT_KEY,
            agent_components.observation.DEFAULT_OBSERVATION_COMPONENT_KEY,
        ]

        # 4. Create the Act Component
        act_component = agent_components.concat_act_component.ConcatActComponent(
            model=model,
            component_order=component_order,
        )

        # 5. Build and return the final agent object
        agent = entity_agent_with_logging.EntityAgentWithLogging(
            agent_name=agent_name,
            act_component=act_component,
            context_components=components_of_agent,
        )

        return agent
