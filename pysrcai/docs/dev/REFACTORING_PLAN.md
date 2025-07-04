# PySrcAI Refactoring Plan

## Current Issues Identified

1. **Broken Imports**: Many files reference `pysrcai.src.*` which doesn't exist
2. **Inconsistent Module Structure**: Some modules are missing or misplaced
3. **Duplicate Files**: Multiple memory component versions exist
4. **Missing language_model_client Module**: Referenced but doesn't exist
5. **Inconsistent Naming**: Some modules use different naming conventions

## Refactoring Strategy

### Phase 1: Fix Import Issues
- [x] Update main package `__init__.py` with clean public API
- [ ] Fix all `pysrcai.src.*` imports to use relative imports
- [ ] Consolidate duplicate memory components
- [ ] Remove references to non-existent `language_model_client`

### Phase 2: Restructure Module Organization
- [ ] Create consistent module hierarchy
- [ ] Move misplaced files to correct locations
- [ ] Standardize naming conventions
- [ ] Create proper `__init__.py` files for all modules

### Phase 3: Clean Up and Optimize
- [ ] Remove duplicate files
- [ ] Consolidate similar functionality
- [ ] Improve documentation
- [ ] Add proper type hints

### Phase 4: Testing and Validation
- [ ] Create comprehensive tests
- [ ] Validate all imports work correctly
- [ ] Ensure backward compatibility
- [ ] Update examples and documentation

## Target Structure

```
pysrcai/
├── __init__.py                 # Main package API
├── agents/                     # Agent system
│   ├── __init__.py
│   ├── base/                   # Core agent classes
│   │   ├── __init__.py
│   │   ├── agent.py
│   │   ├── actor.py
│   │   ├── archon.py
│   │   └── agent_factory.py
│   ├── components/             # Agent components
│   │   ├── __init__.py
│   │   ├── component_factory.py
│   │   └── llm_components.py
│   ├── memory/                 # Memory system
│   │   ├── __init__.py
│   │   ├── memory_components.py
│   │   ├── memory_factory.py
│   │   └── embedders.py
│   └── environment/            # Environment components
│       ├── __init__.py
│       └── environment_components.py
├── core/                       # Core simulation engine
│   ├── __init__.py
│   ├── engine.py
│   └── factory.py
├── llm/                        # Language model system
│   ├── __init__.py
│   ├── language_model.py       # Base class
│   ├── lmstudio_model.py
│   ├── openrouter_model.py
│   ├── no_language_model.py
│   ├── retry_wrapper.py
│   ├── call_limit_wrapper.py
│   └── llm_components.py
├── embeddings/                 # Embedding system
│   ├── __init__.py
│   ├── base.py
│   ├── factory.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── sentence_transformers.py
│   └── vectorstore/
│       ├── __init__.py
│       ├── base.py
│       ├── factory.py
│       └── chroma_adapter.py
├── config/                     # Configuration system
│   ├── __init__.py
│   ├── config_loader.py
│   └── embedding_config.py
├── environment/                # Environment system
│   ├── __init__.py
│   └── objects.py
├── utils/                      # Utility functions
│   ├── __init__.py
│   ├── concurrency.py
│   ├── measurements.py
│   ├── sampling.py
│   ├── text.py
│   ├── json.py
│   ├── html.py
│   └── plotting.py
├── examples/                   # Examples and demos
│   ├── __init__.py
│   ├── demos/
│   ├── configs/
│   ├── agents/
│   ├── environments/
│   └── scenario/
└── tests/                      # Test suite
    ├── __init__.py
    ├── test_agents/
    ├── test_core/
    ├── test_llm/
    └── test_embeddings/
```

## Implementation Steps

### Step 1: Fix Critical Imports
1. Update all `pysrcai.src.*` imports to relative imports
2. Fix LLM module imports
3. Fix agent component imports
4. Fix embedding imports

### Step 2: Consolidate Duplicates
1. Choose the best version of memory components
2. Remove duplicate files
3. Update all references

### Step 3: Create Missing Modules
1. Create proper `__init__.py` files
2. Add missing base classes
3. Standardize exports

### Step 4: Update Documentation
1. Update docstrings
2. Create usage examples
3. Update README

## Success Criteria

- [ ] All imports work without errors
- [ ] No duplicate functionality
- [ ] Clear module hierarchy
- [ ] Consistent naming conventions
- [ ] Comprehensive test coverage
- [ ] Updated documentation
- [ ] Backward compatibility maintained 