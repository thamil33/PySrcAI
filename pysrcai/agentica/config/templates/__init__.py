"""Template management for configuration files."""

import os
from pathlib import Path

TEMPLATES_DIR = Path(__file__).parent

def get_template_path(name: str) -> str:
    """Get the absolute path to a template YAML file by name (without .yaml extension)."""
    filename = f"{name}.yaml"
    filepath = TEMPLATES_DIR / filename
    if filepath.exists():
        return str(filepath)
    raise FileNotFoundError(f"No such template: {name}")

def list_templates() -> list[str]:
    """List all available template configurations."""
    return [f.stem for f in TEMPLATES_DIR.glob("*.yaml")]
