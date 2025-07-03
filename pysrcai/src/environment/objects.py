"""
Environment objects module for PySrcAI.

This module provides classes and utilities for creating interactive
environments with objects that agents can perceive and manipulate.
"""
from typing import Dict, List, Optional, Any, Union


class EnvironmentObject:
    """Base class for all objects in the environment."""
    
    def __init__(
        self, 
        name: str, 
        description: str, 
        properties: Optional[Dict[str, Any]] = None
    ):
        self.name = name
        self.description = description
        self.properties = properties or {}
        self.interactions = []
    
    def add_interaction(self, interaction: Dict[str, Any]) -> None:
        """Add a record of an interaction with this object."""
        self.interactions.append(interaction)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the object to a dictionary representation."""
        return {
            "name": self.name,
            "description": self.description,
            "properties": self.properties,
            "interactions": self.interactions
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EnvironmentObject':
        """Create an object from its dictionary representation."""
        obj = cls(
            name=data["name"],
            description=data["description"],
            properties=data.get("properties", {})
        )
        obj.interactions = data.get("interactions", [])
        return obj


class Location(EnvironmentObject):
    """A location within the environment that can contain objects."""
    
    def __init__(
        self, 
        name: str, 
        description: str, 
        properties: Optional[Dict[str, Any]] = None,
        objects: Optional[Dict[str, EnvironmentObject]] = None
    ):
        super().__init__(name, description, properties)
        self.objects = objects or {}
    
    def add_object(self, obj_id: str, obj: EnvironmentObject) -> None:
        """Add an object to this location."""
        self.objects[obj_id] = obj
    
    def remove_object(self, obj_id: str) -> Optional[EnvironmentObject]:
        """Remove an object from this location and return it."""
        if obj_id in self.objects:
            return self.objects.pop(obj_id)
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the location to a dictionary representation."""
        result = super().to_dict()
        result["objects"] = {
            obj_id: obj.to_dict() for obj_id, obj in self.objects.items()
        }
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Location':
        """Create a location from its dictionary representation."""
        objects = {}
        obj_data = data.get("objects", {})
        
        for obj_id, obj_dict in obj_data.items():
            objects[obj_id] = EnvironmentObject.from_dict(obj_dict)
            
        loc = cls(
            name=data["name"],
            description=data["description"],
            properties=data.get("properties", {}),
            objects=objects
        )
        loc.interactions = data.get("interactions", [])
        return loc


class Item(EnvironmentObject):
    """An item that can be picked up, used, or otherwise interacted with."""
    
    def __init__(
        self, 
        name: str, 
        description: str, 
        properties: Optional[Dict[str, Any]] = None
    ):
        super().__init__(name, description, properties)
        
    @property
    def is_portable(self) -> bool:
        """Check if the item can be carried by an agent."""
        return self.properties.get("portable", False)
        
    @property
    def is_usable(self) -> bool:
        """Check if the item can be used."""
        return self.properties.get("usable", False)


class Environment:
    """
    Environment class that manages locations, objects, and their interactions.
    This acts as a manager for the environment state.
    """
    
    def __init__(self):
        self.locations: Dict[str, Location] = {}
        self.items: Dict[str, Item] = {}
        self.global_state: Dict[str, Any] = {}
        self.active_location: Optional[str] = None
        
    def add_location(self, location_id: str, location: Location) -> None:
        """Add a location to the environment."""
        self.locations[location_id] = location
        
    def add_item(self, item_id: str, item: Item) -> None:
        """Add an item to the environment's global item registry."""
        self.items[item_id] = item
        
    def set_active_location(self, location_id: str) -> bool:
        """Set the currently active location for context."""
        if location_id in self.locations:
            self.active_location = location_id
            return True
        return False
    
    def get_object_description(self, object_id: str) -> Optional[str]:
        """Get a description of an object from its ID."""
        # Check if it's an item
        if object_id in self.items:
            return self.items[object_id].description
            
        # Check if it's a location
        if object_id in self.locations:
            return self.locations[object_id].description
            
        # Check if it's an object in a location
        for loc in self.locations.values():
            if object_id in loc.objects:
                return loc.objects[object_id].description
                
        return None
        
    def process_interaction(
        self, 
        agent_id: str, 
        action: str, 
        target_id: str, 
        **kwargs
    ) -> Dict[str, Any]:
        """
        Process an agent's interaction with an object.
        
        Args:
            agent_id: The ID of the agent performing the action
            action: The type of action (e.g., "examine", "pick_up", "use")
            target_id: The ID of the target object
            **kwargs: Additional parameters for the action
            
        Returns:
            A dictionary with the result of the interaction
        """
        result = {
            "success": False,
            "message": f"Cannot {action} {target_id}. Object not found."
        }
        
        # Handle location-based interactions
        if target_id in self.locations:
            loc = self.locations[target_id]
            
            if action == "examine":
                objects_desc = ", ".join([obj.name for obj in loc.objects.values()])
                desc = f"{loc.description}\nObjects present: {objects_desc if objects_desc else 'None'}"
                
                result = {
                    "success": True,
                    "message": desc,
                    "observation": desc
                }
                
                # Record the interaction
                loc.add_interaction({
                    "agent": agent_id,
                    "action": action,
                    "timestamp": "now"  # You might want to use a real timestamp
                })
                
        # Handle item interactions
        elif target_id in self.items:
            item = self.items[target_id]
            
            if action == "examine":
                result = {
                    "success": True,
                    "message": item.description,
                    "observation": item.description
                }
                
            elif action == "pick_up":
                if item.is_portable:
                    # Logic for picking up the item would go here
                    # This might involve updating an agent's inventory
                    result = {
                        "success": True,
                        "message": f"You picked up the {item.name}.",
                        "observation": f"You now have the {item.name}."
                    }
                else:
                    result = {
                        "success": False,
                        "message": f"The {item.name} cannot be picked up.",
                        "observation": f"The {item.name} is too heavy or fixed in place."
                    }
                    
            elif action == "use":
                if item.is_usable:
                    # Logic for using the item would go here
                    use_target = kwargs.get("use_on")
                    
                    if use_target and item.properties.get("usable_on", []):
                        if use_target in item.properties["usable_on"]:
                            result = {
                                "success": True,
                                "message": f"You used the {item.name} on {use_target}.",
                                "observation": f"Something happened when you used the {item.name}!"
                            }
                        else:
                            result = {
                                "success": False,
                                "message": f"The {item.name} cannot be used on {use_target}.",
                                "observation": f"Nothing happens when you try to use the {item.name} on {use_target}."
                            }
                    else:
                        result = {
                            "success": True,
                            "message": f"You used the {item.name}.",
                            "observation": f"You used the {item.name}, but nothing obvious happened."
                        }
                else:
                    result = {
                        "success": False,
                        "message": f"The {item.name} cannot be used.",
                        "observation": f"You're not sure how to use the {item.name}."
                    }
            
            # Record the interaction
            item.add_interaction({
                "agent": agent_id,
                "action": action,
                "timestamp": "now"  # You might want to use a real timestamp
            })
            
        # Handle object in location interactions
        else:
            found = False
            for loc_id, loc in self.locations.items():
                if target_id in loc.objects:
                    obj = loc.objects[target_id]
                    found = True
                    
                    if action == "examine":
                        result = {
                            "success": True,
                            "message": obj.description,
                            "observation": obj.description
                        }
                    
                    elif action == "search" and obj.properties.get("searchable", False):
                        contents = obj.properties.get("contents", [])
                        if contents:
                            contents_desc = ", ".join([
                                self.items[item_id].name 
                                for item_id in contents 
                                if item_id in self.items
                            ])
                            result = {
                                "success": True,
                                "message": f"You searched the {obj.name} and found: {contents_desc}",
                                "observation": f"Inside the {obj.name}, you discover: {contents_desc}",
                                "found_items": contents
                            }
                        else:
                            result = {
                                "success": True,
                                "message": f"You searched the {obj.name} but found nothing.",
                                "observation": f"The {obj.name} appears to be empty."
                            }
                    
                    # Record the interaction
                    obj.add_interaction({
                        "agent": agent_id,
                        "action": action,
                        "timestamp": "now"  # You might want to use a real timestamp
                    })
                    break
            
            if not found:
                result = {
                    "success": False,
                    "message": f"Cannot find {target_id} to {action}.",
                    "observation": f"You don't see a {target_id} here."
                }
        
        return result

    def to_dict(self) -> Dict[str, Any]:
        """Convert the entire environment to a dictionary representation."""
        return {
            "locations": {
                loc_id: loc.to_dict() for loc_id, loc in self.locations.items()
            },
            "items": {
                item_id: item.to_dict() for item_id, item in self.items.items()
            },
            "global_state": self.global_state,
            "active_location": self.active_location
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Environment':
        """Create an environment from its dictionary representation."""
        env = cls()
        
        # Load locations
        for loc_id, loc_data in data.get("locations", {}).items():
            env.locations[loc_id] = Location.from_dict(loc_data)
            
        # Load items
        for item_id, item_data in data.get("items", {}).items():
            env.items[item_id] = Item.from_dict(item_data)
            
        env.global_state = data.get("global_state", {})
        env.active_location = data.get("active_location")
        
        return env

    @classmethod
    def from_scenario_config(cls, config: Dict[str, Any]) -> 'Environment':
        """
        Create an environment from a scenario configuration.
        
        Args:
            config: The scenario configuration with environment details
            
        Returns:
            An initialized Environment object
        """
        env = cls()
        env_config = config.get("environment", {})
        
        # Load locations
        for loc_id, loc_data in env_config.get("locations", {}).items():
            location = Location(
                name=loc_data.get("name", loc_id),
                description=loc_data.get("description", "A location."),
                properties=loc_data.get("properties", {})
            )
            
            # Add objects to the location
            for obj_id, obj_data in loc_data.get("objects", {}).items():
                obj = EnvironmentObject(
                    name=obj_data.get("name", obj_id),
                    description=obj_data.get("description", "An object."),
                    properties=obj_data.get("properties", {})
                )
                location.add_object(obj_id, obj)
                
            env.add_location(loc_id, location)
            
        # Load items
        for item_id, item_data in env_config.get("items", {}).items():
            item = Item(
                name=item_data.get("name", item_id),
                description=item_data.get("description", "An item."),
                properties=item_data.get("properties", {})
            )
            env.add_item(item_id, item)
            
        # Set active location (if available)
        if env.locations:
            env.active_location = next(iter(env.locations))
            
        return env


def create_environment_from_config(config: Dict[str, Any]) -> Environment:
    """
    Create an environment from a scenario configuration.
    
    This is a convenience function to create and initialize an Environment
    from a configuration dictionary.
    
    Args:
        config: The scenario configuration
        
    Returns:
        An initialized Environment object
    """
    env_data = config.get("scenario", {}).get("initial_state", {}).get("environment", {})
    return Environment.from_scenario_config(env_data)
