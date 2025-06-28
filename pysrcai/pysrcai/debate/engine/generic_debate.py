"""Utility for running debate simulations."""

from collections.abc import Callable
from pathlib import Path
from typing import Any

from concordia.prefabs.simulation import generic as simulation_prefab
from concordia.language_model import openrouter_model, language_model
from concordia.testing import mock_model

from pysrcai.pysrcai.debate.prefabs.entity.participant import ParticipantEntity
from pysrcai.pysrcai.debate.prefabs.entity.moderator import ModeratorEntity
from pysrcai.pysrcai.debate.prefabs.game_master.orchestrator import Orchestrator


def run_simulation(
    scenario_module: Any,
    model: language_model.LanguageModel | None = None,
    embedder_fn: Callable[[str], Any] | None = None,
    max_steps: int | None = None,
) -> str:
    """Run a debate scenario and return HTML log."""
    if model is None:
        try:
            model = openrouter_model.OpenRouterLanguageModel(
                model_name="mistralai/mistral-small-3.1-24b-instruct:free",
                api_key=None,
            )
        except Exception:
            model = mock_model.MockModel()

    if embedder_fn is None:
        try:
            from sentence_transformers import SentenceTransformer

            embedder = SentenceTransformer("all-MiniLM-L6-v2")

            def embed_fn(text: str):
                return embedder.encode(text)
        except Exception:
            import numpy as np

            def embed_fn(text: str):  # pragma: no cover - fallback
                return np.random.randn(384)
    else:
        embed_fn = embedder_fn

    prefabs = {
        "participant_entity": ParticipantEntity(),
        "moderator_entity": ModeratorEntity(),
        "orchestrator": Orchestrator(),
    }

    config = simulation_prefab.Config(
        prefabs=prefabs,
        instances=scenario_module.INSTANCES,
        default_premise=scenario_module.PREMISE,
        default_max_steps=max_steps or 20,
    )

    sim = simulation_prefab.Simulation(
        config=config,
        model=model,
        embedder=embed_fn,
    )

    return sim.play()
