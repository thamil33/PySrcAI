from .engine import SimulationEngine, SequentialEngine
from ..agents.base.agent_factory import AgentFactory
from ..agents.components.component_factory import ComponentFactory
from ..agents.environment.environment_components import EnvironmentComponentFactory
from ..core.objects import create_environment_from_config

class SimulationFactory:
    """
    Factory for creating simulations from configuration.
    
    This factory creates all components of a simulation (engine, agents, environment)
    from a single configuration dictionary, promoting modularity and configurability.
    """
    def create_engine(self, config):
        """
        Create and return a SimulationEngine instance based on the provided config.
        
        Args:
            config: Dictionary containing simulation configuration
            
        Returns:
            Tuple of (engine, steps)
        """
        # Create agents with agent factory
        agents = []
        archon = None
        
        # Process agent configurations
        for agent_cfg in config.get('agents', []):
            agent_type = agent_cfg.get('type', 'actor')
            name = agent_cfg.get('name', 'Agent')
            
            # Create compatible component configs for the agent factory
            
            # Memory component config
            memory_type = agent_cfg.get('memory', 'basic')
            memory_component_config = {
                'name': 'memory',
                'type': 'memory',
                'memory': {
                    'memory_bank': {'type': memory_type},
                    'max_context_memories': 5
                }
            }
            
            # Language model config for acting component
            llm_type = agent_cfg.get('llm', 'mock')
            acting_component_config = {
                'type': 'llm',
                'language_model': {'type': llm_type},
                'temperature': 0.7,
                'max_tokens': 256
            }
            
            # Environment component config if needed
            env_component_config = {
                'name': 'environment',
                'type': 'custom',
                'class_path': 'pysrcai.agents.environment.environment_components.EnvironmentContextComponent',
                'constructor_args': {}
            }
            
            # Build full agent config
            full_agent_config = {
                'name': name,
                'agent_type': agent_type,
                'acting_component': acting_component_config,
                'context_components': [memory_component_config, env_component_config]
            }
            
            # Add agent-specific configuration
            if agent_type == 'actor':
                full_agent_config['personality_traits'] = agent_cfg.get('personality', {})
            elif agent_type == 'archon':
                full_agent_config['authority_level'] = agent_cfg.get('authority_level', 'observer')
                full_agent_config['moderation_rules'] = config.get('scenario', {}).get('rules', [])
            
            # Create agent
            with_logging = True  # Enable logging for development
            agent = AgentFactory.create_agent(full_agent_config, with_logging)
            
            # Optional: per-agent word limit
            if 'word_limit' in agent_cfg:
                setattr(agent, 'word_limit', agent_cfg['word_limit'])
                
            # Add to appropriate collection
            if agent_type == 'archon':
                archon = agent
            else:
                agents.append(agent)

        # Create and configure environment if needed
        environment = None
        if "scenario" in config and "initial_state" in config["scenario"]:
            environment = create_environment_from_config(config)
            
            # Connect environment to agent components
            for agent in agents + ([archon] if archon else []):
                if 'environment' in agent.get_all_context_components():
                    env_component = agent.get_component('environment')
                    env_component.set_environment(environment)

        # Engine selection
        engine_cfg = config.get('engine', {})
        engine_type = engine_cfg.get('type', 'sequential')
        steps = engine_cfg.get('steps', 10)
        state = config.get('scenario', {}).get('initial_state', {})

        # Create appropriate engine
        if engine_type == 'sequential':
            return SequentialEngine(agents=agents, archon=archon, state=state, config=config), steps
        else:
            return SimulationEngine(agents=agents, state=state), steps