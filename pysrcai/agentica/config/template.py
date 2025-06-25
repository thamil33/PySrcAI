"""Configuration template handling for   pysrcai.agentica."""

import os
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional

from pysrcai.agentica.config.config import AgentConfig


def _env_interpolate(value: str) -> str:
    """Interpolate environment variables in a string.
    
    Args:
        value: The string to interpolate.
        
    Returns:
        The interpolated string.
    """
    if not isinstance(value, str):
        return value
        
    if "${" not in value:
        return value
        
    # Replace ${VAR} with os.environ["VAR"]
    for env_var in os.environ:
        placeholder = f"${{{env_var}}}"
        if placeholder in value:
            value = value.replace(placeholder, os.environ[env_var])
            
    return value


def _process_env_vars(config: Dict[str, Any]) -> Dict[str, Any]:
    """Process environment variables in configuration values.
    
    Args:
        config: The configuration dictionary.
        
    Returns:
        The processed configuration with environment variables interpolated.
    """
    result = {}
    
    for key, value in config.items():
        if isinstance(value, dict):
            result[key] = _process_env_vars(value)
        elif isinstance(value, list):
            result[key] = [
                _process_env_vars(item) if isinstance(item, dict) else _env_interpolate(item)
                for item in value
            ]
        else:
            result[key] = _env_interpolate(value)
            
    return result


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from a YAML file.
    
    Args:
        config_path: Path to the configuration file.
        
    Returns:
        The configuration as a dictionary.
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Process environment variables
    if config is not None:
        config = _process_env_vars(config)
    else:
        config = {}
        
    return config


def list_templates() -> List[str]:
    """List available configuration templates.
    
    Returns:
        List of template names.
    """
    template_dir = Path(__file__).parent / "templates"
    
    if not template_dir.exists():
        return []
        
    templates = [f.stem for f in template_dir.glob("*.yaml")]
    return templates


def load_template(template_name: str) -> Optional[AgentConfig]:
    """Load a configuration template.
    
    Args:
        template_name: Name of the template (without .yaml extension).
        
    Returns:
        AgentConfig instance or None if template not found.
    """
    template_dir = Path(__file__).parent / "templates"
    template_path = template_dir / f"{template_name}.yaml"
    
    if not template_path.exists():
        return None
        
    config_dict = load_config(str(template_path))
    return AgentConfig.from_dict(config_dict)
