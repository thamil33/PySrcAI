"""Component factory for PySrcAI.

This module provides factory functions for creating agent components from
configuration data. It allows for declarative configuration of agents and their
components, promoting modularity and reuse.
"""

from typing import Any, Dict, List, Optional, Type, Union, cast
import importlib

from ..base.agent import ActingComponent, ContextComponent
from ..memory.memory_components import (
    MemoryBank,
    BasicMemoryBank,
    AssociativeMemoryBank,
    MemoryComponent
)
from ...llm.llm_components import (
    ActorLLMComponent, 
    ArchonLLMComponent,
    ConfigurableLLMComponent,
    create_language_model
)


class ComponentFactory:
    """Factory class for creating agent components from configuration."""
    
    @staticmethod
    def create_memory_bank(config: Dict[str, Any]) -> MemoryBank:
        """Create a memory bank from configuration.
        
        Args:
            config: Configuration dictionary with the following keys:
                type: The type of memory bank ('basic' or 'associative')
                max_memories: Maximum number of memories to store
                
        Returns:
            A configured MemoryBank instance
        """
        memory_type = config.get('type', 'basic')
        max_memories = config.get('max_memories', 1000)
        
        if memory_type == 'associative':
            # For associative memory, we need an embedder
            # We'll implement this properly when we have embedder support
            try:
                return AssociativeMemoryBank(embedder=None, max_memories=max_memories)
            except ImportError:
                print("Warning: AssociativeMemoryBank requires numpy and pandas. Falling back to BasicMemoryBank")
                return BasicMemoryBank(max_memories=max_memories)
        else:
            return BasicMemoryBank(max_memories=max_memories)
    
    @staticmethod
    def create_memory_component(config: Dict[str, Any]) -> MemoryComponent:
        """Create a memory component from configuration.
        
        Args:
            config: Configuration dictionary with the following keys:
                memory_bank: Configuration for the memory bank
                importance_threshold: Minimum importance for memories to include in context
                max_context_memories: Maximum number of memories to include in context
                
        Returns:
            A configured MemoryComponent instance
        """
        memory_bank_config = config.get('memory_bank', {'type': 'basic'})
        memory_bank = ComponentFactory.create_memory_bank(memory_bank_config)
        
        importance_threshold = config.get('importance_threshold', 0.5)
        max_context_memories = config.get('max_context_memories', 5)
        
        return MemoryComponent(
            memory_bank=memory_bank,
            memory_importance_threshold=importance_threshold,
            max_context_memories=max_context_memories
        )
    
    @staticmethod
    def create_acting_component(config: Dict[str, Any], agent_type: str = 'actor') -> ActingComponent:
        """Create an acting component from configuration.
        
        Args:
            config: Configuration dictionary with the following keys:
                type: Type of acting component ('llm' or custom class)
                language_model: Configuration for language model (if type='llm')
                temperature: Temperature for LLM generation
                max_tokens: Maximum tokens in LLM response
                class_path: Path to custom class (if type!='llm')
                
            agent_type: Type of agent ('actor' or 'archon')
                
        Returns:
            A configured ActingComponent instance
        """
        component_type = config.get('type', 'llm')
        
        if component_type == 'llm':
            # Create language model
            language_model_config = config.get('language_model', {'type': 'mock'})
            language_model = create_language_model(language_model_config.get('type', 'mock'))
            
            # Create appropriate LLM component based on agent type
            temperature = config.get('temperature', 0.7)
            max_tokens = config.get('max_tokens', 256)
            
            if agent_type == 'actor':
                return ActorLLMComponent(
                    language_model=language_model,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
            elif agent_type == 'archon':
                return ArchonLLMComponent(
                    language_model=language_model,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
            else:
                role_description = config.get('role_description', 'a simulation entity')
                return ConfigurableLLMComponent(
                    language_model=language_model,
                    role_description=role_description,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
        else:
            # Import and instantiate custom class
            class_path = config.get('class_path', '')
            if not class_path:
                raise ValueError("Custom acting component requires a class_path")
            
            module_path, class_name = class_path.rsplit('.', 1)
            module = importlib.import_module(module_path)
            component_class = getattr(module, class_name)
            
            # Get constructor arguments
            constructor_args = config.get('constructor_args', {})
            
            return component_class(**constructor_args)
    
    @staticmethod
    def create_context_component(config: Dict[str, Any]) -> ContextComponent:
        """Create a context component from configuration.
        
        Args:
            config: Configuration dictionary with the following keys:
                type: Type of context component ('memory', 'custom', etc.)
                memory: Memory configuration (if type='memory')
                class_path: Path to custom class (if type='custom')
                
        Returns:
            A configured ContextComponent instance
        """
        component_type = config.get('type', '')
        
        if component_type == 'memory':
            memory_config = config.get('memory', {})
            return ComponentFactory.create_memory_component(memory_config)
        elif component_type == 'custom':
            # Import and instantiate custom class
            class_path = config.get('class_path', '')
            if not class_path:
                raise ValueError("Custom context component requires a class_path")
            
            module_path, class_name = class_path.rsplit('.', 1)
            module = importlib.import_module(module_path)
            component_class = getattr(module, class_name)
            
            # Get constructor arguments
            constructor_args = config.get('constructor_args', {})
            
            return component_class(**constructor_args)
        else:
            raise ValueError(f"Unknown context component type: {component_type}")
    
    @staticmethod
    def create_context_components(configs: List[Dict[str, Any]]) -> Dict[str, ContextComponent]:
        """Create multiple context components from configuration.
        
        Args:
            configs: List of component configurations, each with a 'name' key
                
        Returns:
            Dictionary mapping component names to instances
        """
        components = {}
        
        for config in configs:
            name = config.get('name', '')
            if not name:
                raise ValueError("Each component configuration must include a 'name'")
            
            component = ComponentFactory.create_context_component(config)
            components[name] = component
        
        return components
