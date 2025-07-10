"""Module system for PySrcAI.

This module provides the core abstractions for PySrcAI's modular architecture,
enabling different simulation types to be implemented as pluggable modules.
"""

import abc
from typing import Dict, List, Any, Optional, Type, Set
from enum import Enum, auto


class ModuleType(Enum):
    """Types of modules that can be registered with PySrcAI."""
    ENVIRONMENT = auto()
    AGENT = auto()
    ACTION = auto()
    SCENARIO = auto()
    ENGINE = auto()
    ANALYTICS = auto()
    UTILITY = auto()


class ModuleInterface(abc.ABC):
    """Base interface for all PySrcAI modules.
    
    All modules must implement this interface to be compatible with
    the PySrcAI framework.
    """
    
    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Get the name of the module."""
        pass
    
    @property
    @abc.abstractmethod
    def version(self) -> str:
        """Get the version of the module."""
        pass
    
    @property
    @abc.abstractmethod
    def module_type(self) -> ModuleType:
        """Get the type of the module."""
        pass
    
    @property
    def dependencies(self) -> List[str]:
        """Get the list of module dependencies.
        
        Returns:
            List of module names that this module depends on.
        """
        return []
    
    @abc.abstractmethod
    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the module with the provided configuration.
        
        Args:
            config: Configuration dictionary for the module.
        """
        pass
    
    @abc.abstractmethod
    def shutdown(self) -> None:
        """Clean up resources when the module is being shut down."""
        pass


class ModuleRegistry:
    """Registry for managing and discovering PySrcAI modules."""
    
    def __init__(self):
        """Initialize the module registry."""
        self._modules: Dict[str, Type[ModuleInterface]] = {}
        self._instances: Dict[str, ModuleInterface] = {}
    
    def register_module(self, module_class: Type[ModuleInterface]) -> None:
        """Register a module class with the registry.
        
        Args:
            module_class: The module class to register.
        
        Raises:
            ValueError: If a module with the same name is already registered.
        """
        # Create a temporary instance to access properties
        temp_instance = module_class()
        
        if temp_instance.name in self._modules:
            raise ValueError(f"Module with name '{temp_instance.name}' already registered")
        
        self._modules[temp_instance.name] = module_class
        print(f"Registered module: {temp_instance.name} (type: {temp_instance.module_type.name})")
    
    def get_module(self, name: str) -> Optional[Type[ModuleInterface]]:
        """Get a module class by name.
        
        Args:
            name: The name of the module to retrieve.
            
        Returns:
            The module class or None if not found.
        """
        return self._modules.get(name)
    
    def create_instance(self, name: str, config: Dict[str, Any]) -> ModuleInterface:
        """Create and initialize a module instance.
        
        Args:
            name: The name of the module to instantiate.
            config: Configuration dictionary for the module.
            
        Returns:
            Initialized module instance.
            
        Raises:
            ValueError: If the module is not registered or already instantiated.
        """
        if name not in self._modules:
            raise ValueError(f"Module '{name}' not registered")
        
        if name in self._instances:
            raise ValueError(f"Module '{name}' already instantiated")
        
        module_class = self._modules[name]
        instance = module_class()
        instance.initialize(config)
        self._instances[name] = instance
        
        return instance
    
    def get_instance(self, name: str) -> Optional[ModuleInterface]:
        """Get a module instance by name.
        
        Args:
            name: The name of the module instance to retrieve.
            
        Returns:
            The module instance or None if not found.
        """
        return self._instances.get(name)
    
    def shutdown_instance(self, name: str) -> None:
        """Shutdown and remove a module instance.
        
        Args:
            name: The name of the module instance to shutdown.
            
        Raises:
            ValueError: If the module instance is not found.
        """
        if name not in self._instances:
            raise ValueError(f"Module instance '{name}' not found")
        
        self._instances[name].shutdown()
        del self._instances[name]
    
    def shutdown_all(self) -> None:
        """Shutdown and remove all module instances."""
        for name in list(self._instances.keys()):
            self.shutdown_instance(name)
    
    def list_modules(self, module_type: Optional[ModuleType] = None) -> List[str]:
        """List all registered modules, optionally filtered by type.
        
        Args:
            module_type: Optional filter for module type.
            
        Returns:
            List of module names.
        """
        if module_type is None:
            return list(self._modules.keys())
        
        filtered_modules = []
        for name, module_class in self._modules.items():
            temp_instance = module_class()
            if temp_instance.module_type == module_type:
                filtered_modules.append(name)
        
        return filtered_modules
    
    def resolve_dependencies(self, name: str) -> List[str]:
        """Resolve dependencies for a module.
        
        Args:
            name: The name of the module to resolve dependencies for.
            
        Returns:
            List of module names in dependency order.
            
        Raises:
            ValueError: If a circular dependency is detected or a dependency is missing.
        """
        if name not in self._modules:
            raise ValueError(f"Module '{name}' not registered")
        
        visited: Set[str] = set()
        result: List[str] = []
        
        def visit(module_name: str) -> None:
            if module_name in visited:
                return
            
            if module_name not in self._modules:
                raise ValueError(f"Dependency '{module_name}' not found")
            
            # Temporary mark to detect cycles
            visited.add(module_name)
            
            # Create a temporary instance to get dependencies
            temp_instance = self._modules[module_name]()
            for dep in temp_instance.dependencies:
                visit(dep)
            
            result.append(module_name)
        
        visit(name)
        return result


# Global module registry instance
registry = ModuleRegistry()


def register_module(module_class: Type[ModuleInterface]) -> Type[ModuleInterface]:
    """Decorator for registering a module class with the registry.
    
    Args:
        module_class: The module class to register.
        
    Returns:
        The module class (unchanged).
    """
    registry.register_module(module_class)
    return module_class
