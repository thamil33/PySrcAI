"""Agent factory for PySrcAI.

This module provides factory functions for creating agents from configuration data.
It supports creating both Actor and Archon agents with appropriate components.
"""

from typing import Any, Dict, List, Optional, Union, cast
import importlib

from .actor import Actor, ActorWithLogging
from .archon import Archon, ArchonWithLogging
from ..components.component_factory import ComponentFactory


class AgentFactory:
    """Factory class for creating agents from configuration."""
    
    @staticmethod
    def create_actor(config: Dict[str, Any], with_logging: bool = False) -> Actor:
        """Create an Actor agent from configuration.
        
        Args:
            config: Configuration dictionary with the following keys:
                name: The name of the actor
                acting_component: Configuration for the acting component
                context_components: List of context component configurations
                goals: Optional list of goals for the actor
                personality_traits: Optional personality traits for the actor
                
            with_logging: Whether to create an ActorWithLogging
                
        Returns:
            A configured Actor instance
        """
        # Extract basic configuration
        name = config.get('name', 'UnnamedActor')
        goals = config.get('goals', [])
        personality_traits = config.get('personality_traits', {})
        
        # Create acting component
        acting_component_config = config.get('acting_component', {'type': 'llm'})
        acting_component = ComponentFactory.create_acting_component(
            acting_component_config, agent_type='actor'
        )
        
        # Create context components
        context_components_config = config.get('context_components', [])
        context_components = ComponentFactory.create_context_components(context_components_config)
        
        # Create actor
        actor_cls = ActorWithLogging if with_logging else Actor
        return actor_cls(
            agent_name=name,
            act_component=acting_component,
            context_components=context_components,
            goals=goals,
            personality_traits=personality_traits
        )
    
    @staticmethod
    def create_archon(config: Dict[str, Any], with_logging: bool = False) -> Archon:
        """Create an Archon agent from configuration.
        
        Args:
            config: Configuration dictionary with the following keys:
                name: The name of the archon
                acting_component: Configuration for the acting component
                context_components: List of context component configurations
                moderation_rules: Optional list of rules for the archon
                authority_level: Optional authority level for the archon
                managed_entities: Optional list of managed entity names
                
            with_logging: Whether to create an ArchonWithLogging
                
        Returns:
            A configured Archon instance
        """
        # Extract basic configuration
        name = config.get('name', 'UnnamedArchon')
        moderation_rules = config.get('moderation_rules', [])
        authority_level = config.get('authority_level', 'standard')
        managed_entities = config.get('managed_entities', [])
        
        # Create acting component
        acting_component_config = config.get('acting_component', {'type': 'llm'})
        acting_component = ComponentFactory.create_acting_component(
            acting_component_config, agent_type='archon'
        )
        
        # Create context components
        context_components_config = config.get('context_components', [])
        context_components = ComponentFactory.create_context_components(context_components_config)
        
        # Create archon
        archon_cls = ArchonWithLogging if with_logging else Archon
        return archon_cls(
            agent_name=name,
            act_component=acting_component,
            context_components=context_components,
            moderation_rules=moderation_rules,
            authority_level=authority_level,
            managed_entities=managed_entities
        )
    
    @staticmethod
    def create_agent(config: Dict[str, Any], with_logging: bool = False) -> Union[Actor, Archon]:
        """Create an agent (Actor or Archon) based on its type.
        
        Args:
            config: Configuration dictionary with an 'agent_type' key
            with_logging: Whether to create a logging-enabled agent
                
        Returns:
            Either an Actor or Archon instance
        """
        agent_type = config.get('agent_type', '').lower()
        
        if agent_type == 'actor':
            return AgentFactory.create_actor(config, with_logging)
        elif agent_type == 'archon':
            return AgentFactory.create_archon(config, with_logging)
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")
    
    @staticmethod
    def create_agents_from_config(
        config: Dict[str, Any], 
        with_logging: bool = False
    ) -> Dict[str, Union[Actor, Archon]]:
        """Create multiple agents from a configuration dictionary.
        
        Args:
            config: Dictionary with an 'agents' key containing a list of agent configs
            with_logging: Whether to create logging-enabled agents
                
        Returns:
            Dictionary mapping agent names to instances
        """
        agents = {}
        
        # Get agent configurations
        agent_configs = config.get('agents', [])
        
        for agent_config in agent_configs:
            # Map 'type' to 'agent_type' for compatibility with existing config
            if 'type' in agent_config and 'agent_type' not in agent_config:
                agent_config['agent_type'] = agent_config['type']
                
            name = agent_config.get('name', '')
            if not name:
                raise ValueError("Each agent configuration must include a 'name'")
                
            agent = AgentFactory.create_agent(agent_config, with_logging)
            agents[name] = agent
            
        return agents
