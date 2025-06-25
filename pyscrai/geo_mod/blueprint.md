#GeoPolitical Module

## Project Structure:
 pyscrai/geo_mod/
|
|-- __init__.py
|-- blueprint.md
|
|-- prefabs/                # Reusable components for our simulations
|   |-- __init__.py
|   |-- entities/           # For the primary actors (nations, citizens)
|   |   |-- __init__.py
|   |   `-- nation_entity.py
|   `-- game_masters/       # For the controlling agents (moderators, judges)
|       |-- __init__.py
|       `-- moderator_gm.py
|
|-- scenarios/              # Configuration for specific simulation runs
|   |-- __init__.py
|   `-- russia_ukraine_debate.py # Will hold the specific goals, premises, etc.
|
|-- simulations/            # The main, runnable scripts
|   |-- __init__.py
|   `-- phase1_debate.py    # The script we will run for our first simulation
|
`-- utils/                  # Helper functions (e.g., logging setup)
    |-- __init__.py
    `-- logging_config.py


### prefabs: We'll create a generic NationEntity prefab that can be configured to represent any country. This is the core of our reusable component model.

### scenarios: This is where we'll define the data for a specific simulation. For our first run, russia_ukraine_debate.py will define the names, goals, and initial statements for Russia and Ukraine. This separates the logic from the data.

### simulations: This holds the "main" script that pulls everything together—it will import the prefabs and the scenario data to configure and run the simulation.

---

## Implementation Status (Phase 1)

*   **Project Structure:** The full directory structure for the `geo_mod` has been created as outlined, with all necessary `__init__.py` files to make it a proper Python module.
*   **Logging Utility:** A centralized `setup_logging` function has been created in `utils/logging_config.py` to ensure consistent, verbose output across the module.
*   **Nation Entity Prefab:** A flexible `NationEntity` prefab is complete (`prefabs/entities/nation_entity.py`). It uses standard Concordia components to represent a nation based on input `name`, `goal`, and `context` parameters.
*   **Moderator GM Prefab:** The `ModeratorGmPrefab` is complete (`prefabs/game_masters/moderator_gm.py`). It builds a `DebateModerator` agent that manages a simple, turn-based debate for a specified number of turns.
*   **Debate Scenario:** The `russia_ukraine_debate.py` scenario is populated with the initial configurations for the Russia and Ukraine entities and the UN Moderator, defining their goals and the debate structure.
*   **Main Simulation Script:** The `phase1_debate.py` script is implemented. It handles:
    *   Dynamic model selection (OpenRouter or LMStudio via environment variables).
    *   Loading of prefabs and scenario data.
    *   Initialization and execution of the Concordia simulation.
    *   Printing the
