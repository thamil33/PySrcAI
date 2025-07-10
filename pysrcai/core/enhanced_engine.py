"""Enhanced base engine for PySrcAI with module support.

This module provides an improved base class for simulation engines that
leverages the module system for extensibility.
"""

import abc
from typing import Dict, List, Optional, Any, Set
from enum import Enum
import time
import uuid

from ..agents.base.agent import Agent
from ..agents.base.archon import Archon
from .module_system import ModuleInterface, ModuleType, register_module
from .event_system import event_bus, Event


class SimulationState(Enum):
    """Possible states of a simulation."""
    INITIALIZED = 0
    RUNNING = 1
    PAUSED = 2
    TERMINATED = 3
    ERROR = 4


class EnhancedSimulationEngine(abc.ABC):
    """Enhanced base class for all simulation engines in PySrcAI.
    
    This class provides common functionality for simulation engines and
    serves as an interface that all engines must implement. This version
    integrates with the module system for better extensibility.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the simulation engine.
        
        Args:
            config: Configuration dictionary for the simulation.
        """
        self.config = config
        self.agents: List[Agent] = []
        self.archon: Optional[Archon] = None
        self.turn = 0
        self.state = SimulationState.INITIALIZED
        self.simulation_id = str(uuid.uuid4())
        self.start_time = time.time()
        self.loaded_modules: Dict[str, ModuleInterface] = {}
        self.events = []
        
    def register_agent(self, agent: Agent) -> None:
        """Register an agent with the simulation.
        
        Args:
            agent: The agent to register.
        """
        self.agents.append(agent)
        # Store a reference to the engine in the agent's context components if supported
        if hasattr(agent, '_engine'):
            agent._engine = self
        # We used to call agent.set_engine(self), but this method doesn't exist
        # in the new Actor implementation, so we skip it
        print(f"[Engine] Registered agent: {agent.name}")
        
    def register_archon(self, archon: Archon) -> None:
        """Register an archon with the simulation.
        
        Args:
            archon: The archon to register.
        """
        self.archon = archon
        # Store a reference to the engine in the archon's context if supported
        if hasattr(archon, '_engine'):
            archon._engine = self
        # We used to call archon.set_engine(self), but this method doesn't exist
        # in the new Archon implementation, so we skip it
        print(f"[Engine] Registered archon: {archon.name}")
    
    def load_module(self, module_name: str, module_config: Dict[str, Any]) -> ModuleInterface:
        """Load and initialize a module.
        
        Args:
            module_name: Name of the module to load.
            module_config: Configuration for the module.
            
        Returns:
            The initialized module instance.
            
        Raises:
            ValueError: If the module is already loaded.
        """
        if module_name in self.loaded_modules:
            raise ValueError(f"Module '{module_name}' already loaded")
        
        from .module_system import registry
        
        # Load module and its dependencies
        dependencies = registry.resolve_dependencies(module_name)
        
        for dep_name in dependencies:
            if dep_name not in self.loaded_modules and dep_name != module_name:
                # Load dependency with empty config if not explicitly provided
                dep_config = module_config.get("dependencies", {}).get(dep_name, {})
                self.load_module(dep_name, dep_config)
        
        # Load the requested module
        instance = registry.create_instance(module_name, module_config)
        self.loaded_modules[module_name] = instance
        print(f"[Engine] Loaded module: {module_name} (version: {instance.version})")
        
        return instance
    
    def emit_event(self, event_type: str, data: Any = None) -> None:
        """Emit an event to all interested modules.
        
        Args:
            event_type: Type of the event.
            data: Data associated with the event.
        """
        # Create event data with additional engine context
        enhanced_data = {
            "original_data": data,
            "engine_context": {
                "turn": self.turn,
                "simulation_id": self.simulation_id,
                "state": self.state.name
            }
        }
        
        # Use the global event bus to publish the event
        event = event_bus.publish_event(event_type, enhanced_data, source="engine")
        
        # Store in local events list as well
        self.events.append({
            "type": event_type,
            "data": data,
            "timestamp": event.timestamp,
            "turn": self.turn,
            "event_id": event.id
        })
        
        # Directly notify modules that have an event_handler method (legacy support)
        for module_name, module in self.loaded_modules.items():
            if hasattr(module, "handle_event") and callable(getattr(module, "handle_event")):
                try:
                    module.handle_event(event_type, data)
                except Exception as e:
                    print(f"[Engine] Error in module {module_name} handling event {event_type}: {e}")
    
    @abc.abstractmethod
    def run(self, steps: Optional[int] = None) -> None:
        """Run the simulation for a specified number of steps.
        
        Args:
            steps: Number of steps to run. If None, runs indefinitely or
                  until termination criteria are met.
        """
        pass
    
    @abc.abstractmethod
    def step(self) -> Dict[str, Any]:
        """Run a single simulation step.
        
        Returns:
            Dictionary containing step results.
        """
        pass
    
    def pause(self) -> None:
        """Pause the simulation."""
        if self.state == SimulationState.RUNNING:
            self.state = SimulationState.PAUSED
            self.emit_event("simulation_paused")
            print("[Engine] Simulation paused.")
    
    def resume(self) -> None:
        """Resume a paused simulation."""
        if self.state == SimulationState.PAUSED:
            self.state = SimulationState.RUNNING
            self.emit_event("simulation_resumed")
            print("[Engine] Simulation resumed.")
    
    def terminate(self) -> None:
        """Terminate the simulation."""
        self.state = SimulationState.TERMINATED
        self.emit_event("simulation_terminated")
        print("[Engine] Simulation terminated.")
        
        # Shutdown all loaded modules
        for module_name in list(self.loaded_modules.keys()):
            try:
                self.loaded_modules[module_name].shutdown()
            except Exception as e:
                print(f"[Engine] Error shutting down module {module_name}: {e}")
            
        self.loaded_modules.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the simulation.
        
        Returns:
            Dictionary containing simulation statistics.
        """
        current_time = time.time()
        return {
            "simulation_id": self.simulation_id,
            "state": self.state.name,
            "turn": self.turn,
            "agent_count": len(self.agents),
            "has_archon": self.archon is not None,
            "runtime_seconds": current_time - self.start_time,
            "loaded_modules": list(self.loaded_modules.keys()),
            "event_count": len(self.events)
        }


class EngineModule(ModuleInterface):
    """Base class for engine modules.
    
    Engine modules extend the functionality of the simulation engine.
    """
    
    @property
    def module_type(self) -> ModuleType:
        """Get the type of the module."""
        return ModuleType.ENGINE
    
    def handle_event(self, event_type: str, data: Any) -> None:
        """Handle an event emitted by the engine.
        
        Args:
            event_type: Type of the event.
            data: Data associated with the event.
        """
        pass
