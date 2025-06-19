# pyscrai & concordia

This repository contains the **pyscrai** package and the core **concordia** framework.
The project focuses on building modular conversational simulations that can be easily extended.

## Setup
1. Create a Python 3.12 virtual environment named `.venv` at the project root:
   ```bash
   python -m venv .venv
   ```
2. Activate the virtual environment:
   - On Linux/macOS:
     ```bash
     source .venv/bin/activate
     ```
   - On Windows (bash):
     ```bash
     source .venv/Scripts/activate
     ```
3. Install python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Install the packages in editable mode:
   ```bash
   pip install -e .
   ```
5. Run a scenario from the project root:
   ```bash
   python -m pyscrai.scenario.angeldemon
   ```

## Documentation
See the [docs](docs/README.md) folder for architecture notes and usage guides.
