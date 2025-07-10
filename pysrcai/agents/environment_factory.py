"""Environment factory for PySrcAI.

This module provides a factory for creating environment instances
based on configuration.
"""

from typing import Dict, Any, Type, Optional

from ..core.module_system import registry, ModuleInterface, ModuleType
from .environment.module_interface import EnvironmentModule


class EnvironmentFactory:
    """Factory for creating environments in PySrcAI."""
    
    @staticmethod
    def create_environment(env_config: Dict[str, Any]) -> EnvironmentModule:
        """Create an environment instance based on configuration.
        
        Args:
            env_config: Environment configuration.
            
        Returns:
            Initialized environment instance.
            
        Raises:
            ValueError: If the environment type is not found or initialization fails.
        """
        env_type = env_config.get("type")
        if not env_type:
            raise ValueError("Environment type not specified in config")
        
        # Get all registered environment modules
        env_modules = registry.list_modules(ModuleType.ENVIRONMENT)
        
        # Find a module that matches the requested type
        for module_name in env_modules:
            module_class = registry.get_module(module_name)
            if not module_class:
                continue
                
            # Create temporary instance to check properties
            temp_instance = module_class()
            if hasattr(temp_instance, "env_type") and getattr(temp_instance, "env_type") == env_type:
                # This is the module we want
                try:
                    return registry.create_instance(module_name, env_config)
                except Exception as e:
                    raise ValueError(f"Failed to initialize environment '{env_type}': {e}")
        
        raise ValueError(f"Environment type '{env_type}' not found in registered modules")
