"""Environment components for PySrcAI.

This module provides components for agent interaction with the environment.
These components allow agents to perceive and act upon the environment in
a modular, configurable way.
"""

from typing import Any, Dict, List, Optional
from collections.abc import Mapping

from ..base.agent import ContextComponent, ActionSpec, ComponentState


class EnvironmentContextComponent(ContextComponent):
    """Context component that provides environment state to agents.
    
    This component gives agents awareness of their environment and enables
    them to include environmental context in their decision-making.
    """
    
    def __init__(self, environment=None):
        """Initialize the environment context component.
        
        Args:
            environment: The environment object (optional, can be set later).
        """
        super().__init__()
        self._environment = environment
        self._last_observation = ""
        self._last_location = None
        self._visible_objects = {}
        self._inventory = []
    
    def set_environment(self, environment):
        """Set or update the environment reference."""
        self._environment = environment
    
    def update_perception(self, location, visible_objects, inventory=None):
        """Update the agent's current perception of the environment.
        
        Args:
            location: The current location data
            visible_objects: Dictionary of visible objects
            inventory: Optional list of items in the agent's inventory
        """
        self._last_location = location
        self._visible_objects = visible_objects
        if inventory is not None:
            self._inventory = inventory
    
    def pre_act(self, action_spec: ActionSpec) -> str:
        """Provide environmental context for acting."""
        if not self._environment:
            return "No environment context available."
        
        # Provide environmental perception
        context_parts = []
        
        # Location information
        if self._last_location:
            loc_name = self._last_location.get('name', 'Unknown location')
            loc_desc = self._last_location.get('description', '')
            context_parts.append(f"Current location: {loc_name}")
            context_parts.append(f"Description: {loc_desc}")
        
        # Visible objects
        if self._visible_objects:
            context_parts.append("\nVisible objects:")
            for obj_id, obj in self._visible_objects.items():
                obj_name = obj.get('name', obj_id)
                obj_desc = obj.get('description', '')
                context_parts.append(f"- {obj_name}: {obj_desc}")
        else:
            context_parts.append("\nNo visible objects in the area.")
        
        # Inventory items
        if self._inventory:
            context_parts.append("\nInventory:")
            for item in self._inventory:
                item_name = item.get('name', 'Unknown item')
                context_parts.append(f"- {item_name}")
        else:
            context_parts.append("\nInventory is empty.")
        
        return "\n".join(context_parts)
    
    def pre_observe(self, observation: str) -> str:
        """Process environmental observations."""
        self._last_observation = observation
        
        # Attempt to update perception based on observation
        # In a full implementation, this would parse observations for
        # environmental changes, but this is a simplified version
        return "Environmental context updated based on observation."
    
    def get_state(self) -> ComponentState:
        """Get the component's state."""
        return {
            'last_observation': self._last_observation,
            'last_location': self._last_location,
            'visible_objects': self._visible_objects,
            'inventory': self._inventory,
        }
    
    def set_state(self, state: ComponentState) -> None:
        """Set the component's state."""
        self._last_observation = state.get('last_observation', "")
        self._last_location = state.get('last_location')
        self._visible_objects = state.get('visible_objects', {})
        self._inventory = state.get('inventory', [])


class InteractionComponent(ContextComponent):
    """Context component that handles agent-environment interactions.
    
    This component translates agent actions into environmental interactions
    and provides feedback on the results.
    """
    
    def __init__(self, environment=None):
        """Initialize the interaction component.
        
        Args:
            environment: The environment object (optional, can be set later).
        """
        super().__init__()
        self._environment = environment
        self._last_action = ""
        self._last_result = {}
        self._available_actions = []
    
    def set_environment(self, environment):
        """Set or update the environment reference."""
        self._environment = environment
    
    def set_available_actions(self, actions):
        """Set the list of available actions in the current context."""
        self._available_actions = actions
    
    def post_act(self, action_attempt: str) -> str:
        """Process the action and interact with environment."""
        if not self._environment:
            return "No environment available for interaction."
        
        self._last_action = action_attempt
        agent = self.get_agent()
        
        # In a full implementation, this would parse the action and
        # translate it into an environment interaction
        # For now, we'll just store it
        
        return f"Action '{action_attempt}' will be processed by the environment."
    
    def get_available_actions(self) -> List[str]:
        """Get the list of available actions in the current context."""
        return self._available_actions.copy()
    
    def get_last_interaction_result(self) -> Dict[str, Any]:
        """Get the result of the last interaction."""
        return self._last_result.copy()
    
    def get_state(self) -> ComponentState:
        """Get the component's state."""
        return {
            'last_action': self._last_action,
            'last_result': self._last_result,
            'available_actions': self._available_actions,
        }
    
    def set_state(self, state: ComponentState) -> None:
        """Set the component's state."""
        self._last_action = state.get('last_action', "")
        self._last_result = state.get('last_result', {})
        self._available_actions = state.get('available_actions', [])


class EnvironmentComponentFactory:
    """Factory class for creating environment-related components."""
    
    @staticmethod
    def create_environment_context(config: Dict[str, Any], environment=None) -> EnvironmentContextComponent:
        """Create an environment context component from configuration.
        
        Args:
            config: Configuration dictionary
            environment: Environment object to connect to the component
            
        Returns:
            A configured EnvironmentContextComponent
        """
        component = EnvironmentContextComponent(environment)
        
        # Configure perception defaults if specified
        if 'default_perception' in config:
            perception = config['default_perception']
            component.update_perception(
                perception.get('location', {}),
                perception.get('visible_objects', {}),
                perception.get('inventory', [])
            )
        
        return component
    
    @staticmethod
    def create_interaction_component(config: Dict[str, Any], environment=None) -> InteractionComponent:
        """Create an interaction component from configuration.
        
        Args:
            config: Configuration dictionary
            environment: Environment object to connect to the component
            
        Returns:
            A configured InteractionComponent
        """
        component = InteractionComponent(environment)
        
        # Configure available actions if specified
        if 'available_actions' in config:
            component.set_available_actions(config['available_actions'])
        
        return component
