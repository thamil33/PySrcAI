"""Minimal components for the PySrcAI *hello‑world* scenario.

These are **NOT** intended for production—only to satisfy the strictest
possible MVP requirements so that `MinimalEngine` compiles without dragging
in the broader PySrcAI environment/action framework.

--------------------------------------------------------------------
Contents
--------------------------------------------------------------------
1. **MinimalEnvironment** – a thin wrapper around a shared `whiteboard` list.
2. **MinimalActionParser**   – regex‑based validator for exactly two verbs
   (``propose`` and ``respond``) with ≤75‑word content.

Anything more complex (object trees, location graphs, pick‑ups, etc.) is
explicitly out of scope.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import List, Tuple, Dict, Any, Optional

from ...core.engines.base_engine import SimulationEngine

# ---------------------------------------------------------------------
# 1. MinimalEnvironment
# ---------------------------------------------------------------------
@dataclass
class MinimalEnvironment:
    """Single shared artefact: a *whiteboard* visible to every agent."""
    whiteboard: List[str]

    def __init__(self, initial_whiteboard: List[str] | None = None):
        """Initialize the environment with an optional starting whiteboard."""
        self.whiteboard = initial_whiteboard or []

    def add_note(self, note: str) -> None:
        """Append a line to the whiteboard."""
        self.whiteboard.append(note)

    def snapshot(self, n: int | None = None) -> List[str]:
        """Return the last *n* notes (or all if n is ``None``)."""
        return self.whiteboard[-n:] if n else list(self.whiteboard)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a dictionary representation for state tracking."""
        return {
            "whiteboard": self.whiteboard
        }

    @staticmethod
    def from_config(config: Dict[str, Any]) -> MinimalEnvironment:
        """Create a minimal environment from configuration."""
        initial_state = config.get("scenario", {}).get("initial_state", {})
        whiteboard = initial_state.get("whiteboard", [])
        return MinimalEnvironment(whiteboard)

# ---------------------------------------------------------------------
# 2. MinimalActionParser
# ---------------------------------------------------------------------
class MinimalActionParser:
    """Enforces the *only* two legal templates for the MVP scenario.

    Syntax (case‑insensitive on *ACTION* keyword)::

        ACTION: propose\nCONTENT: <argument …>   # opening phase only
        ACTION: respond\nCONTENT: <argument …>   # rebuttal / closing

    *CONTENT* must be ≤75 words (easy token sanity check).
    """

    _TEMPLATE = re.compile(
        r"^ACTION:\s*(?P<verb>propose|respond)\s*\nCONTENT:\s*(?P<content>.+)",
        re.I | re.S,
    )

    MAX_WORDS = 75
    TOLERANCE_WORDS = 15  # Allow up to 15 words over the limit

    @classmethod
    def parse(cls, raw: str) -> Tuple[str, str, bool]:
        """Return ``(verb, content, exceeded_limit)`` or raise ``ValueError``."""
        m = cls._TEMPLATE.match(raw.strip())
        if not m:
            raise ValueError(
                "Invalid template. Expected 'ACTION: <propose|respond>\\nCONTENT: …'"
            )

        verb = m.group("verb").lower()
        content = m.group("content").strip()
        word_count = len(content.split())
        
        if word_count > cls.MAX_WORDS + cls.TOLERANCE_WORDS:
            raise ValueError(
                f"CONTENT exceeds {cls.MAX_WORDS + cls.TOLERANCE_WORDS}‑word limit (got {word_count})."
            )
        
        exceeded_limit = word_count > cls.MAX_WORDS
        return verb, content, exceeded_limit

    @classmethod
    def as_dict(cls, raw: str) -> Dict[str, Any]:
        """Helper that returns ``{"verb": verb, "content": content, "exceeded_limit": bool}``.

        Useful if you prefer dict style for downstream logging.
        """
        verb, content, exceeded_limit = cls.parse(raw)
        return {"verb": verb, "content": content, "exceeded_limit": exceeded_limit}

# ---------------------------------------------------------------------
# MinimalActionProcessor - connects agents to the environment
# ---------------------------------------------------------------------
class MinimalActionProcessor:
    """Processes agent actions in the minimal environment.
    
    This class validates agent actions using the MinimalActionParser
    and applies them to the MinimalEnvironment.
    """
    
    def __init__(self, environment: MinimalEnvironment):
        """Initialize with a reference to the environment."""
        self.environment = environment
    
    def process_action(self, agent_name: str, action_text: str) -> Dict[str, Any]:
        """Process an agent's action.
        
        Args:
            agent_name: The name of the agent performing the action
            action_text: The raw action text from the agent
            
        Returns:
            Dict with results of processing the action
        """
        try:
            # Parse and validate the action
            action = MinimalActionParser.as_dict(action_text)
            
            # Format the whiteboard entry
            entry = f"{agent_name}: {action['content']}"
            
            # Add to whiteboard
            self.environment.add_note(entry)
            
            return {
                "success": True,
                "message": f"{agent_name} contributed to the whiteboard.",
                "verb": action["verb"],
                "content": action["content"]
            }
            
        except ValueError as e:
            # Handle invalid action format
            return {
                "success": False,
                "message": f"Invalid action format: {str(e)}",
                "error": str(e)
            }

# ---------------------------------------------------------------------
# Factory function to create a minimal environment from config
# ---------------------------------------------------------------------
def create_minimal_environment_from_config(config: Dict[str, Any]) -> MinimalEnvironment:
    """Create a minimal environment from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured MinimalEnvironment instance
    """
    return MinimalEnvironment.from_config(config)
