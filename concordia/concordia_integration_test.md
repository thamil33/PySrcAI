# Concordia Integration Test

## Overview

The Concordia integration test (`concordia_integration_test.py`) validates the core functionality of the Concordia framework by running a complete multi-agent simulation. The test creates two agents (Alice and Bob) with associative memory, sets up a game master with conversation scenes, and executes a simulated social experiment over several hours of game time.

## Running the Test

The integration test can be run using two different methods to avoid Python module path conflicts:

### Method 1: Using pytest (Recommended)
```bash
pytest -xvs concordia/concordia_integration_test.py::GameMasterTest::test_full_run
```

### Method 2: Using Python module execution
```bash
python -m concordia.concordia_integration_test
```

**Note**: Do not run the test directly with `python concordia/concordia_integration_test.py` as this causes module shadowing issues where `concordia.typing` conflicts with Python's built-in `typing` module.

## What the Test Validates

The integration test verifies:

- **Agent Creation**: Instantiation of agents with associative memory and entity components
- **Memory System**: Agents forming and retrieving memories based on observations
- **Decision Making**: Agents making decisions based on their perceptions and personality
- **Game Master**: Coordination of multi-agent interactions and scene management
- **Environment Engine**: Proper simulation flow and time progression
- **Component Integration**: All framework components working together seamlessly

The test uses a mock language model that returns placeholder text, but demonstrates the complete pipeline from agent creation through multi-hour simulation execution.