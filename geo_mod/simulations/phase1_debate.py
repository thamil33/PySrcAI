"""Main runnable script for the Phase 1 geopolitical debate simulation."""

import os
import sys
from pathlib import Path

# --- Add project root to sys.path ---
# This allows us to import modules from pyscrai and concordia
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))
# -----------------------------------

import sentence_transformers
from concordia.prefabs.simulation import generic as simulation
from concordia.typing import prefab as prefab_lib

# --- Model Imports ---
# We need to handle both OpenRouter and a local model provider like LMStudio
from concordia.language_model import openrouter_model, lmstudio_model, no_language_model

# --- Geo-Mod Imports ---
from pyscrai.geo_mod.prefabs.entities.nation_entity import NationEntity
from pyscrai.geo_mod.prefabs.game_masters.moderator_gm import ModeratorGmPrefab
from pyscrai.geo_mod.scenarios.russia_ukraine_debate import PREMISE, INSTANCES
from pyscrai.geo_mod.utils.logging_config import setup_logging

# --- Environment Setup ---
from dotenv import load_dotenv
load_dotenv()

# --- Main Simulation Logic ---
def main():
    """Sets up and runs the geopolitical debate simulation."""
    # 1. Configure Logging
    logger = setup_logging()
    logger.info("Starting Phase 1: Geopolitical Debate Simulation.")

    # 2. Configure Language Model
    # Use an environment variable to easily switch between models.
    # Set USE_LMSTUDIO=true in your .env file to use a local model.
    use_lmstudio = os.getenv("USE_LMSTUDIO", "false").lower() == "true"
    disable_llm = os.getenv("DISABLE_LANGUAGE_MODEL", "false").lower() == "true"

    model = None
    if disable_llm:
        logger.info("Language model is DISABLED.")
        model = no_language_model.NoLanguageModel()
    elif use_lmstudio:
        logger.info("Using LMStudio model.")
        base_url = os.getenv("LMSTUDIO_API_BASE", "http://localhost:1234/v1")
        model_name = os.getenv("LMSTUDIO_MODEL_NAME", "local-model")
        model = lmstudio_model.LMStudioLanguageModel(
            model_name=model_name,
            base_url=base_url,
            verbose_logging=True,
        )
    else:
        logger.info("Using OpenRouter model.")
        api_key = os.environ.get('OPENROUTER_API_KEY')
        model_name = os.environ.get('MODEL_NAME', 'mistralai/mistral-small-3.1-24b-instruct:free')
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY not found in environment.")
        model = openrouter_model.OpenRouterLanguageModel(
            api_key=api_key,
            model_name=model_name,
            verbose_logging=True,
        )

    # 3. Configure Sentence Embedder
    if disable_llm:
        embedder = lambda _: [0.0] * 768 # Return a dummy vector
    else:
        st_model = sentence_transformers.SentenceTransformer(
            'sentence-transformers/all-mpnet-base-v2'
        )
        embedder = lambda x: st_model.encode(x, show_progress_bar=False)
    logger.info("Sentence embedder configured.")

    # 4. Map Prefab Strings to Classes
    # This connects the scenario data to our prefab implementations.
    prefabs = {
        'nation_entity': NationEntity(),
        'moderator_gm': ModeratorGmPrefab(),
    }

    # 5. Create the Main Simulation Configuration
    config = prefab_lib.Config(
        default_premise=PREMISE,
        prefabs=prefabs,
        instances=INSTANCES,
    )
    logger.info("Simulation config created.")

    # 6. Initialize the Simulation
    runnable_simulation = simulation.Simulation(
        config=config,
        model=model,
        embedder=embedder,
    )
    logger.info("Simulation object initialized. Starting the simulation...")

    # 7. Run the Simulation
    results_log = runnable_simulation.play()

    # 8. Display Results
    # For now, we just print the log. We can save it to a file later.
    print("\n--- Simulation Log ---\n")
    print(results_log)
    logger.info("Simulation finished.")

if __name__ == "__main__":
    main()
