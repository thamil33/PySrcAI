"""Environment module interfaces for PySrcAI.

This module provides base classes and interfaces for implementing
environment modules in the PySrcAI framework.
"""

import abc
from typing import Dict, List, Any, Optional, TypeVar, Generic
from enum import Enum

from ...core.module_system import ModuleInterface, ModuleType


# Generic type for environment state
StateT = TypeVar('StateT')


class ActionResult(Generic[StateT]):
    """Result of an agent action in an environment."""
    
    def __init__(
        self,
        success: bool,
        message: str,
        state_delta: Optional[StateT] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Initialize the action result.
        
        Args:
            success: Whether the action was successful.
            message: Human-readable message describing the result.
            state_delta: Optional change to the environment state.
            metadata: Optional additional metadata about the action result.
        """
        self.success = success
        self.message = message
        self.state_delta = state_delta
        self.metadata = metadata or {}


class EnvironmentModule(ModuleInterface, Generic[StateT]):
    """Base class for environment modules in PySrcAI.
    
    Environment modules are responsible for maintaining the state of the
    simulation world and processing agent actions.
    """
    
    @property
    def module_type(self) -> ModuleType:
        """Get the type of the module."""
        return ModuleType.ENVIRONMENT
    
    @abc.abstractmethod
    def get_state(self) -> StateT:
        """Get the current state of the environment.
        
        Returns:
            Current environment state.
        """
        pass
    
    @abc.abstractmethod
    def set_state(self, state: StateT) -> None:
        """Set the state of the environment.
        
        Args:
            state: The new environment state.
        """
        pass
    
    @abc.abstractmethod
    def process_action(self, agent_id: str, action: Dict[str, Any]) -> ActionResult[StateT]:
        """Process an agent's action in the environment.
        
        Args:
            agent_id: ID of the agent performing the action.
            action: The action to process.
            
        Returns:
            Result of the action.
        """
        pass
    
    @abc.abstractmethod
    def get_observation(self, agent_id: str) -> Dict[str, Any]:
        """Get an agent's observation of the environment.
        
        Args:
            agent_id: ID of the agent observing the environment.
            
        Returns:
            Agent's observation.
        """
        pass


class SerializableEnvironment(EnvironmentModule[StateT]):
    """Environment that supports serialization and deserialization.
    
    This is useful for saving and loading environment states.
    """
    
    @abc.abstractmethod
    def serialize(self) -> Dict[str, Any]:
        """Serialize the environment state.
        
        Returns:
            Serialized state.
        """
        pass
    
    @abc.abstractmethod
    def deserialize(self, data: Dict[str, Any]) -> None:
        """Deserialize and load environment state.
        
        Args:
            data: Serialized state data.
        """
        pass
