"""Base engine implementation for PySrcAI.

This module provides the foundation for all simulation engines in the PySrcAI framework.
It defines core interfaces and common functionality for running agent-based simulations.
"""

from typing import Dict, List, Optional, Any, Tuple
from abc import ABC, abstractmethod


class SimulationEngine(ABC):
    """Base class for all simulation engines.
    
    This abstract class defines the interface that all simulation engines must implement.
    It provides common functionality for initializing and running simulations.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the simulation engine.
        
        Args:
            config: Configuration dictionary for the simulation.
        """
        self.config = config
        self.agents = []  # List of agent instances
        self.archon = None  # The archon/moderator agent
        self.environment = None  # Environment state
        self.turn = 0  # Current turn number
        self.history = []  # History of actions and observations
        
    def setup_agents(self, agents: List[Any], archon: Any = None) -> None:
        """Set up agents for the simulation.
        
        Args:
            agents: List of agent instances.
            archon: Optional archon/moderator agent.
        """
        self.agents = agents
        self.archon = archon
        
    def setup_environment(self, environment: Any) -> None:
        """Set up the environment for the simulation.
        
        Args:
            environment: The environment object.
        """
        self.environment = environment
        
    @abstractmethod
    def run(self, steps: Optional[int] = None) -> None:
        """Run the simulation for a specified number of steps.
        
        Args:
            steps: Number of steps to run. If None, uses the value from config.
        """
        pass
        
    @abstractmethod
    def step(self) -> Dict[str, Any]:
        """Run a single simulation step.
        
        Returns:
            Dictionary containing step results.
        """
        pass
        
    def get_state(self) -> Dict[str, Any]:
        """Get the current simulation state.
        
        Returns:
            Dictionary containing the current state.
        """
        return {
            "turn": self.turn,
            "agents": self.agents,
            "environment": self.environment,
            "history": self.history
        }
        
    def log_event(self, event_type: str, event_data: Dict[str, Any]) -> None:
        """Log an event in the simulation history.
        
        Args:
            event_type: Type of event.
            event_data: Event data.
        """
        event = {
            "turn": self.turn,
            "type": event_type,
            "data": event_data
        }
        self.history.append(event)
        print(f"[{event_type}] {event_data.get('message', '')}")