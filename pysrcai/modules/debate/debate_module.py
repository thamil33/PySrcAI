"""Debate module for PySrcAI.

This module provides a debate scenario implementation using the PySrcAI
module system, serving as an example of how to build a specific simulation
type as a pluggable module.
"""

from typing import Dict, List, Any, Optional
from enum import Enum

from ...core.module_system import ModuleInterface, ModuleType, register_module
from ...core.enhanced_engine import EngineModule
from ...core.event_system import event_bus


class DebatePhase(Enum):
    """Phases in a debate."""
    OPENING = "opening"
    REBUTTAL = "rebuttal"
    CLOSING = "closing"


@register_module
class DebateModule(EngineModule):
    """Module for running structured debates in PySrcAI.
    
    This module provides debate-specific functionality like phase management,
    turn tracking, and moderation.
    """
    
    def __init__(self):
        """Initialize the debate module."""
        self._phase = DebatePhase.OPENING
        self._turn_index = 0
        self._debate_topic = ""
        self._agents_per_side = 0
        self._word_limit = 75
        self._roles = []  # List of agent roles (proponent, opponent)
        self._engine = None
        self._phases = [DebatePhase.OPENING, DebatePhase.REBUTTAL, DebatePhase.CLOSING]
    
    @property
    def name(self) -> str:
        """Get the name of the module."""
        return "debate_module"
    
    @property
    def version(self) -> str:
        """Get the version of the module."""
        return "1.0.0"
    
    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the module with the provided configuration.
        
        Args:
            config: Configuration dictionary for the module.
        """
        self._debate_topic = config.get("debate_topic", "Should we adopt a four-day work week?")
        self._agents_per_side = config.get("agents_per_side", 1)
        self._word_limit = config.get("word_limit", 75)
        
        # Get engine reference
        self._engine = config.get("engine")
        
        # Subscribe to events
        event_bus.subscribe("agent_acted", self._handle_agent_acted)
        event_bus.subscribe("phase_complete", self._handle_phase_complete)
        
        print(f"[DebateModule] Initialized with topic: {self._debate_topic}")
        print(f"[DebateModule] Word limit: {self._word_limit}")
        print(f"[DebateModule] Initial phase: {self._phase.value}")
    
    def shutdown(self) -> None:
        """Clean up resources when the module is being shut down."""
        # Unsubscribe from events
        event_bus.unsubscribe("agent_acted", self._handle_agent_acted)
        event_bus.unsubscribe("phase_complete", self._handle_phase_complete)
    
    def get_current_phase(self) -> DebatePhase:
        """Get the current debate phase.
        
        Returns:
            Current debate phase.
        """
        return self._phase
    
    def advance_phase(self) -> bool:
        """Advance to the next debate phase.
        
        Returns:
            True if there was a next phase to advance to, False otherwise.
        """
        current_index = self._phases.index(self._phase)
        if current_index < len(self._phases) - 1:
            self._phase = self._phases[current_index + 1]
            self._turn_index = 0
            
            print(f"[DebateModule] Advanced to phase: {self._phase.value}")
            
            # Publish phase change event
            event_bus.publish_event("phase_changed", {
                "old_phase": self._phases[current_index].value,
                "new_phase": self._phase.value
            }, source=self.name)
            
            return True
        else:
            print("[DebateModule] Already at final phase")
            return False
    
    def is_phase_complete(self) -> bool:
        """Check if the current phase is complete.
        
        Returns:
            True if the current phase is complete, False otherwise.
        """
        return self._turn_index >= self._agents_per_side * 2  # Both sides get a turn
    
    def get_debate_topic(self) -> str:
        """Get the debate topic.
        
        Returns:
            Debate topic.
        """
        return self._debate_topic
    
    def get_word_limit(self) -> int:
        """Get the word limit for responses.
        
        Returns:
            Word limit.
        """
        return self._word_limit
    
    def _handle_agent_acted(self, event: Any) -> None:
        """Handle an agent acted event.
        
        Args:
            event: The event data.
        """
        # Increment turn index
        self._turn_index += 1
        
        # Check if phase is complete
        if self.is_phase_complete():
            event_bus.publish_event("phase_complete", {
                "phase": self._phase.value,
                "turns_taken": self._turn_index
            }, source=self.name)
    
    def _handle_phase_complete(self, event: Any) -> None:
        """Handle a phase complete event.
        
        Args:
            event: The event data.
        """
        # If we're the source of the event, advance phase
        if event.source == self.name:
            # Get moderator assessment before advancing
            if self._engine and hasattr(self._engine, "archon") and self._engine.archon:
                self._request_moderator_assessment()
            
            # Advance to next phase
            if not self.advance_phase():
                # If we couldn't advance, we're at the final phase
                if self._engine and hasattr(self._engine, "archon") and self._engine.archon:
                    self._request_moderator_final()
                
                # Terminate simulation
                event_bus.publish_event("debate_complete", {
                    "debate_topic": self._debate_topic,
                }, source=self.name)
    
    def _request_moderator_assessment(self) -> None:
        """Request an assessment from the moderator."""
        if not self._engine or not hasattr(self._engine, "archon") or not self._engine.archon:
            return
        
        print("[DebateModule] Requesting moderator assessment")
        
        # This would be implemented in the engine to call the moderator
        event_bus.publish_event("request_moderation", {
            "type": "assessment",
            "phase": self._phase.value
        }, source=self.name)
    
    def _request_moderator_final(self) -> None:
        """Request a final judgment from the moderator."""
        if not self._engine or not hasattr(self._engine, "archon") or not self._engine.archon:
            return
        
        print("[DebateModule] Requesting moderator final judgment")
        
        # This would be implemented in the engine to call the moderator
        event_bus.publish_event("request_moderation", {
            "type": "final",
            "phase": self._phase.value
        }, source=self.name)
