# Modular Architecture Proposal

This document describes the recommended structure for future development of **pyscrai** alongside the existing `concordia` framework.

## Goals
- Encourage code reuse between projects.
- Provide clear extension points for additional components and scenarios.
- Simplify maintenance and testing.

## Package Layout
```
pyscrai/
    components/   # Reusable context and action components
    embedding/    # Embedding backends
    engine/       # Simulation engines and game masters
    scenario/     # Pre-built scenarios and templates
    utils/        # Shared helper utilities
```
Each subpackage exposes its key classes through `__init__.py` to provide a clean public API.

## Integration Points
- `pyscrai.engine` uses engines and game masters that rely on types from `concordia`.
- Components follow the same interfaces as those in `concordia.components` allowing them to be swapped or extended.
- Scenarios should import from `pyscrai.engine` and `pyscrai.components` rather than individual files to reduce coupling.

## Future Expansion
- Implement a configuration system to assemble scenarios from YAML or JSON.
- Add persistent memory implementations using Concordia's memory interfaces.
- Provide more component templates (e.g., analytics, logging).

