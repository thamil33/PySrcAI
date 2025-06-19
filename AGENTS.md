# Repository Guide for Codex Agents

This repository hosts the **pyscrai** package alongside the **concordia** framework. It implements modular components for conversational simulations. Key design information can be found in the documentation:

- `docs/architecture/overview.md` – overall package structure
- `docs/guides/pyscrai/Development_Plan.md` – roadmap
- `docs/guides/pyscrai/observability.md` – observability and persistence plans

## Working in this Repository

1. **Environment Setup**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   pip install -e .
   ```

2. **Testing**
   - Run `pytest` from the project root to execute all tests.
   - If dependencies cannot be installed due to network restrictions, mention this in the PR.

3. **Coding Guidelines**
   - Keep modules modular and extensible, following PEP8 style.
   - Update or add unit tests when altering functionality.

4. **Commits and Pull Requests**
   - Commit directly to the default branch; avoid creating new branches.
   - Use concise commit messages and provide a short summary of test results in the PR body.

Consult the docs directory for further background and development notes.
