"""Generic debate simulation engine."""

from pathlib import Path
import logging
import sys

import numpy as np
from concordia.language_model import openrouter_model
from concordia.prefabs.game_master import dialogic_and_dramaturgic as dd_gm
from concordia.prefabs.simulation import generic as simulation_prefab
from concordia.testing import mock_model
from sentence_transformers import SentenceTransformer

from pysrcai.pysrcai.debate.prefabs.entity.participant import Participant
from pysrcai.pysrcai.debate.prefabs.entity.moderator import Moderator


class DebateEngine:
    """Run a debate simulation from a scenario module."""

    def __init__(self, scenario_module):
        self.scenario = scenario_module

    def run(self) -> str:
        logger = logging.getLogger("debate")
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.propagate = False

        try:
            model = openrouter_model.OpenRouterLanguageModel(
                model_name="mistralai/mistral-small-3.1-24b-instruct:free",
                api_key=None,
            )
        except Exception as e:  # pragma: no cover - network not available
            logger.warning("Falling back to mock model: %s", e)
            model = mock_model.MockModel()

        try:
            embedder = SentenceTransformer("all-MiniLM-L6-v2")

            def embed_fn(text: str) -> np.ndarray:
                return embedder.encode(text)

        except Exception as e:  # pragma: no cover - model missing
            logger.warning("Using random embeddings: %s", e)

            def embed_fn(text: str) -> np.ndarray:  # noqa: D401
                return np.random.randn(384)

        prefabs = {
            "participant": Participant(),
            "moderator": Moderator(),
            "dialogic_and_dramaturgic_gm": dd_gm.GameMaster(),
        }

        config = simulation_prefab.Config(
            prefabs=prefabs,
            instances=self.scenario.INSTANCES,
            default_premise=self.scenario.PREMISE,
            default_max_steps=20,
        )

        sim = simulation_prefab.Simulation(
            config=config,
            model=model,
            embedder=embed_fn,
        )

        logger.info("Starting debate simulation.")
        results = sim.play()
        logger.info("Debate finished.")

        output_file = Path("debate_results.html")
        with output_file.open("w", encoding="utf-8") as f:
            f.write(results)
        logger.info("Results saved to %s", output_file)

        return results
