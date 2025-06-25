"""Geo_mod - Geopolitical simulation framework integrating with Concordia.

This module provides specialized tools for creating geopolitical simulations with:
- Nation entities representing countries with goals and contexts
- Debate moderators for structured international discussions
- Pre-configured scenarios for specific geopolitical situations
- Turn-based simulation engines for diplomatic interactions
- Logging and utility functions for simulation management

The framework is designed to model complex international relations,
negotiations, conflicts, and diplomatic scenarios using AI agents.
"""

__version__ = "1.0.0"

# Import key components for easy access
from pyscrai.geo_mod.prefabs.entities import nation_entity
from pyscrai.geo_mod.prefabs.game_masters import moderator_gm
from pyscrai.geo_mod.scenarios import russia_ukraine_debate
from pyscrai.geo_mod.simulations import phase1_debate
from pyscrai.geo_mod.utils import logging_config