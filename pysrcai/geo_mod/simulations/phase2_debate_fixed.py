"""Phase 2 geopolitical debate simulation with scene-based structure using D&D GameMaster."""

import sys
import logging
from pathlib import Path

# --- Add project root to sys.path ---
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))
# -----------------------------------

import numpy as np
from concordia.prefabs.simulation import generic as simulation_prefab
from concordia.language_model import openrouter_model
from sentence_transformers import SentenceTransformer

# --- Geo-Mod Imports ---
from pysrcai.geo_mod.prefabs.entities.nation_entity import NationEntity
from pysrcai.geo_mod.prefabs.entities.moderator_entity import ModeratorEntity
from pysrcai.geo_mod.scenarios.russia_ukraine_debate_phase2 import (
    PREMISE, INSTANCES
)
from pysrcai.geo_mod.utils.logging_config import setup_logging

# Import D&D GameMaster
from concordia.prefabs.game_master import dialogic_and_dramaturgic as dd_gm

# --- Constants ---
OPENROUTER_MODEL = "mistralai/mistral-small-3.1-24b-instruct:free"
SIMULATION_NAME = "Russia-Ukraine UN Debate Phase 2"


def main():
    """Run the Phase 2 geopolitical debate simulation using D&D GameMaster."""
    
    # 1. Setup logging
    logger = setup_logging(level=logging.INFO)
    logger.info("Starting Phase 2: Scene-Based Geopolitical Debate Simulation.")
    
    # 2. Initialize the language model
    try:
        model = openrouter_model.OpenRouterLanguageModel(
            model_name=OPENROUTER_MODEL,
            api_key=None,  # Reads from OPENROUTER_API_KEY environment variable
        )
        logger.info("Using OpenRouter model.")
    except Exception as e:
        logger.error(f"Failed to initialize OpenRouter model: {e}")
        logger.info("Falling back to a mock model for testing.")
        from concordia.testing import mock_model
        model = mock_model.MockModel()
    
    # 3. Configure the sentence embedder
    try:
        embedder = SentenceTransformer("all-MiniLM-L6-v2")
        def embed_fn(text: str) -> np.ndarray:
            return embedder.encode(text)
        logger.info("Sentence embedder configured.")
    except Exception as e:
        logger.error(f"Failed to load sentence embedder: {e}")
        # Use a simple mock embedder for testing
        def embed_fn(text: str) -> np.ndarray:
            return np.random.randn(384)  # Mock 384-dimensional embedding
    
    # 4. Create prefab instances
    prefabs = {
        'nation_entity': NationEntity(),
        'moderator_entity': ModeratorEntity(), 
        'dialogic_and_dramaturgic_gm': dd_gm.GameMaster(),
    }
    
    # 5. Create simulation configuration
    config = simulation_prefab.Config(
        prefabs=prefabs,
        instances=INSTANCES,
        default_premise=PREMISE,
        default_max_steps=20,  # Allow enough steps for full debate
    )
    
    logger.info("Simulation config created.")
    
    # 6. Initialize and run the simulation
    runnable_simulation = simulation_prefab.Simulation(
        config=config,
        model=model,
        embedder=embed_fn,
    )
    
    logger.info("Simulation object initialized. Starting the simulation...")
    
    try:
        # Run the simulation
        results_log = runnable_simulation.play()
        
        logger.info("Scene-based simulation completed successfully!")
        
        # 7. Print final summary
        print("\n" + "="*80)
        print(f"SIMULATION COMPLETE: {SIMULATION_NAME}")
        print("="*80)
        print(f"Entities: {[entity.name for entity in runnable_simulation.get_entities()]}")
        print(f"Game Masters: {[gm.name for gm in runnable_simulation.get_game_masters()]}")
        print("\nSimulation used Dialogic & Dramaturgic GameMaster with built-in scene management.")
        print("The D&D GameMaster automatically handled scene transitions and structured debate flow.")
        print("\n" + "="*80)
        
        # Optionally save the HTML log
        html_log_path = Path("phase2_debate_results.html")
        with open(html_log_path, 'w', encoding='utf-8') as f:
            f.write(results_log)
        logger.info(f"Detailed HTML results saved to: {html_log_path}")
        
    except KeyboardInterrupt:
        logger.info("Simulation interrupted by user.")
        print("\nSimulation interrupted.")
    except Exception as e:
        logger.error(f"Simulation failed: {e}")
        print(f"Simulation error: {e}")
        raise


if __name__ == "__main__":
    main()
