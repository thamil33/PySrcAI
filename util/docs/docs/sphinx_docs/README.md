# API Documentation

This directory contains the essential configuration files for building API documentation for the Concordia framework and PyScrai packages.

## Building the Documentation from Scratch

### Prerequisites

Make sure you have the required dependencies installed:

```bash
pip install sphinx sphinx-rtd-theme sphinx-autodoc-typehints myst-parser
```

### Full Documentation Build Process

To build the complete documentation from scratch:

```bash
# 1. Generate module documentation files (from the .api_docs directory)
sphinx-apidoc -o modules/ ../.. --separate --force

# 2. Build HTML documentation
# On Windows:
.\make.bat html
# On Unix/Linux/macOS:
make html

# 3. Optional: Build JSON documentation for LLM consumption
# On Windows:
.\make.bat json
# On Unix/Linux/macOS:
make json
```

The generated files will be in:
- `_build/html/` - HTML documentation (open `_build/html/index.html` in browser)
- `_build/json/` - JSON format for programmatic access

### Quick HTML-only Build

If the `modules/` directory already exists and you just want to rebuild HTML:

```bash
# On Windows
.\make.bat html

# On Unix/Linux/macOS
make html
```

### Cleaning Build Files

To clean all generated files and start fresh:

```bash
# On Windows
.\make.bat clean

# On Unix/Linux/macOS
make clean
```

## Repository Structure

The `.api_docs` directory now contains only essential source files:

- `conf.py` - Sphinx configuration file
- `index.rst` - Main documentation index
- `README.md` - This file with build instructions
- `make.bat` / `Makefile` - Build scripts

Generated directories (excluded from git):
- `modules/` - Auto-generated module documentation (.rst files)
- `_build/` - Generated documentation output (HTML, JSON, etc.)
- `_static/` - Static files for documentation (auto-created)
- `_templates/` - Custom templates (auto-created if needed)

## Configuration

The documentation is configured to:

- Use the Read the Docs theme
- Auto-extract docstrings from Python modules  
- Support Google/NumPy style docstrings
- Include type hints in the documentation
- Support Markdown files via MyST parser

## Viewing the Documentation

After building, open `_build/html/index.html` in your web browser to view the complete API documentation.

## For Developers

When adding new modules to the codebase, you'll need to regenerate the module documentation:

```bash
sphinx-apidoc -o modules/ ../.. --separate --force
```

Then rebuild the documentation as usual. The `modules/` directory and all build outputs are excluded from git to keep the repository size minimal.

## LLM Agent Consumption

This documentation system is well-suited for use by LLM agents with the following features:

### Human-Readable Format (HTML)
- Browse `_build/html/index.html` for human consumption
- Hierarchical navigation with search functionality
- Cross-references between modules and classes

### Machine-Readable Format (JSON)
For LLM agents, generate structured JSON documentation:

```bash
# Generate JSON format documentation
sphinx-build -b json . _build/json
```

The JSON format provides:
- **Structured data**: Each module/class documented in separate `.fjson` files
- **Complete API information**: Method signatures, parameters, return types, docstrings
- **Type hints**: Full type annotation extraction for better LLM understanding
- **Cross-references**: Links between related components
- **Metadata**: Module hierarchy, inheritance relationships

### LLM-Friendly Features
- ✅ **Complete API coverage**: All public classes, methods, and functions
- ✅ **Type information**: Parameter types, return types, and type hints
- ✅ **Structured format**: JSON output for programmatic consumption
- ✅ **Hierarchical organization**: Clear module/package structure
- ✅ **Cross-references**: Automatic linking between related components
- ✅ **Search capability**: Built-in search functionality in HTML
- ✅ **Source code links**: Direct links to implementation code

### JSON Documentation Structure
Each `.fjson` file contains:
- `title`: Module/class name
- `body`: Complete HTML-formatted documentation
- `toc`: Table of contents with method/class hierarchy
- `rellinks`: Navigation links to related modules
- `parents`: Module hierarchy information

This makes the documentation highly suitable for LLM agents to understand and work with the Concordia API.
