"""Enhanced minimal environment implementation for PySrcAI.

This module provides an enhanced version of the minimal environment
that integrates with the module system.
"""

from typing import Dict, List, Any, Optional
import json

from ..environment.module_interface import EnvironmentModule, ActionResult
from ...core.module_system import ModuleType, register_module


@register_module
class EnhancedMinimalEnvironment(EnvironmentModule[List[str]]):
    """Enhanced minimal environment for PySrcAI debates.
    
    This environment provides a simple whiteboard for agents to share information
    and implements the module interface for better integration with the framework.
    """
    
    def __init__(self):
        """Initialize the minimal environment."""
        self._whiteboard: List[str] = []
        self._env_type = "minimal"
    
    @property
    def name(self) -> str:
        """Get the name of the module."""
        return "minimal_environment"
    
    @property
    def version(self) -> str:
        """Get the version of the module."""
        return "1.0.0"
    
    @property
    def env_type(self) -> str:
        """Get the type of environment."""
        return self._env_type
    
    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the module with the provided configuration.
        
        Args:
            config: Configuration dictionary for the module.
        """
        # Initialize whiteboard with any provided initial notes
        initial_notes = config.get("initial_notes", [])
        self._whiteboard = list(initial_notes)
        
        print(f"[MinimalEnvironment] Initialized with {len(self._whiteboard)} initial notes")
    
    def shutdown(self) -> None:
        """Clean up resources when the module is being shut down."""
        # Nothing to clean up in this simple environment
        pass
    
    def get_state(self) -> List[str]:
        """Get the current state of the environment.
        
        Returns:
            Current whiteboard contents.
        """
        return self._whiteboard.copy()
    
    def set_state(self, state: List[str]) -> None:
        """Set the state of the environment.
        
        Args:
            state: New whiteboard contents.
        """
        self._whiteboard = list(state)
    
    def process_action(self, agent_id: str, action: Dict[str, Any]) -> ActionResult[List[str]]:
        """Process an agent's action in the environment.
        
        Args:
            agent_id: ID of the agent performing the action.
            action: The action to process.
            
        Returns:
            Result of the action.
        """
        action_type = action.get("type", "").lower()
        content = action.get("content", "")
        
        if not content:
            return ActionResult(
                success=False,
                message="No content provided for the action",
                state_delta=None
            )
        
        # Process different action types
        if action_type == "propose" or action_type == "respond":
            # Add content to the whiteboard
            note = f"{agent_id}: {content}"
            self._whiteboard.append(note)
            return ActionResult(
                success=True,
                message=f"Added note from {agent_id}",
                state_delta=[note]
            )
        elif action_type == "query":
            # Return matching whiteboard entries
            matches = [note for note in self._whiteboard if content.lower() in note.lower()]
            return ActionResult(
                success=True,
                message=f"Found {len(matches)} matching notes",
                state_delta=None,
                metadata={"query_results": matches}
            )
        else:
            return ActionResult(
                success=False,
                message=f"Unknown action type: {action_type}",
                state_delta=None
            )
    
    def get_observation(self, agent_id: str) -> Dict[str, Any]:
        """Get an agent's observation of the environment.
        
        Args:
            agent_id: ID of the agent observing the environment.
            
        Returns:
            Agent's observation.
        """
        # In this simple environment, all agents observe the entire whiteboard
        return {
            "whiteboard": self._whiteboard.copy()
        }
    
    # Convenience methods specific to this environment
    
    def add_note(self, note: str) -> None:
        """Add a note to the whiteboard.
        
        Args:
            note: The note to add.
        """
        self._whiteboard.append(note)
    
    def clear(self) -> None:
        """Clear the whiteboard."""
        self._whiteboard.clear()
    
    def snapshot(self) -> List[str]:
        """Get a snapshot of the whiteboard.
        
        Returns:
            Copy of the whiteboard contents.
        """
        return self._whiteboard.copy()
    
    # For compatibility with SerializableEnvironment (optional implementation)
    
    def serialize(self) -> Dict[str, Any]:
        """Serialize the environment state.
        
        Returns:
            Serialized state.
        """
        return {
            "whiteboard": self._whiteboard.copy()
        }
    
    def deserialize(self, data: Dict[str, Any]) -> None:
        """Deserialize and load environment state.
        
        Args:
            data: Serialized state data.
        """
        if "whiteboard" in data:
            self._whiteboard = list(data["whiteboard"])
            
            
class EnhancedMinimalActionParser:
    """Parser for actions in the minimal environment."""
    
    @staticmethod
    def parse(response: str) -> tuple[str, str, bool]:
        """Parse an agent's response into an action.
        
        Args:
            response: The agent's response string.
            
        Returns:
            Tuple of (action_type, content, exceeded_limit).
            
        Raises:
            ValueError: If the response format is invalid.
        """
        lines = response.strip().split("\n")
        action_type = ""
        content = ""
        word_limit = 75  # Default word limit
        
        for line in lines:
            if line.startswith("ACTION:"):
                action_type = line.replace("ACTION:", "").strip()
            elif line.startswith("CONTENT:"):
                content = line.replace("CONTENT:", "").strip()
        
        if not action_type:
            raise ValueError("Missing ACTION: field in response")
        
        if not content:
            raise ValueError("Missing CONTENT: field in response")
        
        # Check if word limit was exceeded
        word_count = len(content.split())
        exceeded_limit = word_count > word_limit
        
        return action_type, content, exceeded_limit
