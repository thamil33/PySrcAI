from .engine import SimulationEngine, SequentialEngine
from .agents.actor import Actor
from .agents.archon import Archon
from .agents.memory.memory_components import MemoryComponent
from .agents.memory.memory_components import BasicMemoryBank, AssociativeMemoryBank
from .agents.memory.embedders import create_simple_embedder
from .language_model_client import create_language_model

class SimulationFactory:
    """
    Base class for simulation factories. Responsible for creating engines and agents from configuration.
    """
    def create_engine(self, config):
        """
        Create and return a SimulationEngine instance based on the provided config.
        """
        agents = []
        archon = None
        for agent_cfg in config.get('agents', []):
            agent_type = agent_cfg.get('type', 'actor')
            name = agent_cfg.get('name', 'Agent')
            personality = agent_cfg.get('personality', {})
            memory_type = agent_cfg.get('memory', 'basic')
            llm_type = agent_cfg.get('llm', None)
            llm = create_language_model(llm_type)
            context_components = {}

            # Memory setup
            if memory_type == 'associative':
                embedder = create_simple_embedder()
                memory_bank = AssociativeMemoryBank(embedder=embedder.embed, max_memories=100)
            else:
                memory_bank = BasicMemoryBank(max_memories=100)
            memory_component = MemoryComponent(memory_bank, max_context_memories=5)
            context_components['memory'] = memory_component

            if agent_type == 'archon':
                archon = Archon(
                    agent_name=name,
                    context_components=context_components,
                    authority_level=agent_cfg.get('authority_level', 'observer'),
                    language_model=llm
                )
            else:
                agents.append(Actor(
                    agent_name=name,
                    context_components=context_components,
                    personality_traits=personality,
                    language_model=llm
                ))

        # Engine selection
        engine_cfg = config.get('engine', {})
        engine_type = engine_cfg.get('type', 'sequential')
        steps = engine_cfg.get('steps', 10)
        state = config.get('scenario', {}).get('initial_state', {})

        if engine_type == 'sequential':
            return SequentialEngine(agents=agents, archon=archon, state=state), steps
        else:
            return SimulationEngine(agents=agents, state=state), steps 