"""Action processing components for PySrcAI.

This module provides components for processing agent actions and translating
them into environment interactions in a modular, extensible way.
"""

from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod

from ..base.agent import ContextComponent, ActionSpec, ComponentState



@dataclass
class ActionResult:
    """Result of processing an agent action."""
    success: bool
    message: str
    observation: Optional[str] = None
    found_items: Optional[List[str]] = None


class ActionHandler(ABC):
    """Abstract base class for action handlers."""
    
    @abstractmethod
    def can_handle(self, action_lower: str) -> bool:
        """Check if this handler can process the given action."""
        pass
    
    @abstractmethod
    def process(self, agent, action: str, action_lower: str, environment: Any) -> ActionResult:
        """Process the action and return the result."""
        pass


class WaitActionHandler(ActionHandler):
    """Handles wait/observe actions."""
    
    def can_handle(self, action_lower: str) -> bool:
        return any(phrase in action_lower for phrase in ["wait", "do nothing", "observe", "wait and observe"])
    
    def process(self, agent, action: str, action_lower: str, environment: Any) -> ActionResult:
        return ActionResult(True, f"{agent.name} chooses to do nothing and observe their surroundings.")


class ExamineActionHandler(ActionHandler):
    """Handles examine actions."""
    
    def can_handle(self, action_lower: str) -> bool:
        return "examine" in action_lower
    
    def process(self, agent, action: str, action_lower: str, environment: Any) -> ActionResult:
        if "examine room" in action_lower:
            return self._process_examine_room(agent, environment)
        else:
            return self._process_examine_object(agent, action_lower, environment)
    
    def _process_examine_room(self, agent, environment: Any) -> ActionResult:
        """Process examine room action."""
        if not environment.active_location:
            return ActionResult(True, f"{agent.name} looks around but sees nothing distinctive.")
        
        result = environment.process_interaction(
            agent_id=agent.name,
            action="examine",
            target_id=environment.active_location
        )
        return ActionResult(
            success=result.get("success", False),
            message=result.get("message", ""),
            observation=result.get("observation", "")
        )
    
    def _process_examine_object(self, agent, action_lower: str, environment: Any) -> ActionResult:
        """Process examine object action."""
        for obj_id in self._get_all_object_ids(environment):
            obj_name = self._get_object_name(obj_id, environment).lower()
            if obj_name in action_lower or obj_id.lower() in action_lower:
                result = environment.process_interaction(
                    agent_id=agent.name,
                    action="examine",
                    target_id=obj_id
                )
                return ActionResult(
                    success=result.get("success", False),
                    message=result.get("message", ""),
                    observation=result.get("observation", "")
                )
        
        return ActionResult(False, "You don't see that object here.")
    
    def _get_all_object_ids(self, environment: Any) -> List[str]:
        """Get all object IDs in the environment for action matching."""
        object_ids = []
        object_ids.extend(environment.items.keys())
        object_ids.extend(environment.locations.keys())
        
        for location in environment.locations.values():
            object_ids.extend(location.objects.keys())
        
        return object_ids
    
    def _get_object_name(self, obj_id: str, environment: Any) -> str:
        """Get the display name of an object by its ID."""
        if obj_id in environment.items:
            return environment.items[obj_id].name
        
        if obj_id in environment.locations:
            return environment.locations[obj_id].name
        
        for location in environment.locations.values():
            if obj_id in location.objects:
                return location.objects[obj_id].name
        
        return obj_id


class SearchActionHandler(ActionHandler):
    """Handles search actions."""
    
    def can_handle(self, action_lower: str) -> bool:
        return "search" in action_lower
    
    def process(self, agent, action: str, action_lower: str, environment: Any) -> ActionResult:
        for obj_id in self._get_all_object_ids(environment):
            obj_name = self._get_object_name(obj_id, environment).lower()
            if obj_name in action_lower or obj_id.lower() in action_lower:
                return self._handle_search_interaction(agent, obj_id, environment)
        
        return ActionResult(False, "You don't see that object here.")
    
    def _handle_search_interaction(self, agent, obj_id: str, environment: Any) -> ActionResult:
        """Handle the actual search interaction logic."""
        if not environment.active_location:
            return self._process_generic_search(agent, obj_id, environment)
        
        active_loc = environment.locations.get(environment.active_location)
        if not active_loc or obj_id not in active_loc.objects:
            return self._process_generic_search(agent, obj_id, environment)
        
        obj = active_loc.objects[obj_id]
        contents = obj.properties.get("contents", [])
        hidden_items = []
        already_found = []
        
        # Check what's already visible vs what's still hidden
        for item_id in contents:
            if item_id in environment.items:
                item = environment.items[item_id]
                if item.properties.get("location", "hidden") == "hidden":
                    hidden_items.append(item_id)
                else:
                    already_found.append(item_id)
        
        # If all items already found, provide that feedback
        if not hidden_items and already_found:
            names = [environment.items[item_id].name for item_id in already_found]
            return ActionResult(
                success=True,
                message=f"You already searched this {obj.name} and found: {', '.join(names)}.",
                observation=f"You search the {obj.name} again, but find nothing new beyond what you already discovered."
            )
        
        # Process normal search interaction
        result = environment.process_interaction(
            agent_id=agent.name,
            action="search",
            target_id=obj_id
        )
        
        # Handle search results
        if result.get("success", False) and "found_items" in result:
            newly_found = []
            for item_id in result["found_items"]:
                if item_id in environment.items:
                    item = environment.items[item_id]
                    if item.properties.get("location", "hidden") == "hidden":
                        item.properties["location"] = "visible"
                        newly_found.append(item_id)
            
            # Update result message if some items were already found
            if not newly_found and already_found:
                names = [environment.items[item_id].name for item_id in already_found]
                result["observation"] = f"You search the {obj.name} again, but find nothing new beyond what you already discovered: {', '.join(names)}"
        
        return ActionResult(
            success=result.get("success", False),
            message=result.get("message", ""),
            observation=result.get("observation", ""),
            found_items=result.get("found_items", [])
        )
    
    def _process_generic_search(self, agent, obj_id: str, environment: Any) -> ActionResult:
        """Process search for non-location objects."""
        result = environment.process_interaction(
            agent_id=agent.name,
            action="search",
            target_id=obj_id
        )
        
        # Handle search results
        if result.get("success", False) and "found_items" in result:
            for item_id in result["found_items"]:
                if item_id in environment.items:
                    environment.items[item_id].properties["location"] = "visible"
        
        return ActionResult(
            success=result.get("success", False),
            message=result.get("message", ""),
            observation=result.get("observation", ""),
            found_items=result.get("found_items", [])
        )
    
    def _get_all_object_ids(self, environment: Any) -> List[str]:
        """Get all object IDs in the environment for action matching."""
        object_ids = []
        object_ids.extend(environment.items.keys())
        object_ids.extend(environment.locations.keys())
        
        for location in environment.locations.values():
            object_ids.extend(location.objects.keys())
        
        return object_ids
    
    def _get_object_name(self, obj_id: str, environment: Any) -> str:
        """Get the display name of an object by its ID."""
        if obj_id in environment.items:
            return environment.items[obj_id].name
        
        if obj_id in environment.locations:
            return environment.locations[obj_id].name
        
        for location in environment.locations.values():
            if obj_id in location.objects:
                return location.objects[obj_id].name
        
        return obj_id


class SpeakActionHandler(ActionHandler):
    """Handles speak/talk actions."""
    
    def can_handle(self, action_lower: str) -> bool:
        return any(word in action_lower for word in ["speak", "talk", "say", "tell"])
    
    def process(self, agent, action: str, action_lower: str, environment: Any) -> ActionResult:
        # This would need access to other agents - simplified for now
        about_index = action_lower.find("about")
        if about_index != -1:
            topic = action[about_index + 5:].strip()
            message = f"Hello, I'd like to discuss {topic}"
        else:
            message = "Hello, let's talk."
        
        return ActionResult(True, f"{agent.name} says: {message}")


class TakeActionHandler(ActionHandler):
    """Handles take/pick up actions."""
    
    def can_handle(self, action_lower: str) -> bool:
        return any(word in action_lower for word in ["take", "pick up", "grab", "get"])
    
    def process(self, agent, action: str, action_lower: str, environment: Any) -> ActionResult:
        for item_id in environment.items:
            if item_id.lower() in action_lower or environment.items[item_id].name.lower() in action_lower:
                item = environment.items[item_id]
                if item.properties.get("location", "visible") == "visible":
                    result = environment.process_interaction(
                        agent_id=agent.name,
                        action="pick_up",
                        target_id=item_id
                    )
                    
                    if result.get("success", False):
                        item.properties["location"] = f"inventory_{agent.name}"
                    
                    return ActionResult(
                        success=result.get("success", False),
                        message=result.get("message", ""),
                        observation=result.get("observation", "")
                    )
                else:
                    return ActionResult(
                        success=False,
                        message=f"You don't see the {item.name} here.",
                        observation=f"The {item.name} is not visible or accessible."
                    )
        
        return ActionResult(False, "You don't see that item here.")


class ReadActionHandler(ActionHandler):
    """Handles read actions."""
    
    def can_handle(self, action_lower: str) -> bool:
        return "read" in action_lower
    
    def process(self, agent, action: str, action_lower: str, environment: Any) -> ActionResult:
        for item_id in environment.items:
            if item_id.lower() in action_lower or environment.items[item_id].name.lower() in action_lower:
                item = environment.items[item_id]
                
                if item.properties.get("readable", False):
                    content = item.properties.get("content", "There's nothing written here.")
                    return ActionResult(
                        success=True,
                        message=f"You read the {item.name}.",
                        observation=f"The {item.name} says: '{content}'"
                    )
                else:
                    return ActionResult(
                        success=False,
                        message=f"The {item.name} is not readable.",
                        observation=f"There's nothing to read on the {item.name}."
                    )
        
        return ActionResult(False, "You don't see that item here.")


class ActionProcessingComponent(ContextComponent):
    """Context component that processes agent actions and translates them into environment interactions."""
    
    def __init__(self, environment: Optional[Any] = None):
        """Initialize the action processing component.
        
        Args:
            environment: The environment object (optional, can be set later).
        """
        super().__init__()
        self._environment = environment
        self._handlers: List[ActionHandler] = []
        self._last_action = ""
        self._last_result: Optional[ActionResult] = None
        
        # Register default handlers
        self._register_default_handlers()
    
    def set_environment(self, environment: Any) -> None:
        """Set or update the environment reference."""
        self._environment = environment
    
    def add_handler(self, handler: ActionHandler) -> None:
        """Add a custom action handler."""
        self._handlers.append(handler)
    
    def remove_handler(self, handler: ActionHandler) -> None:
        """Remove an action handler."""
        if handler in self._handlers:
            self._handlers.remove(handler)
    
    def post_act(self, action_attempt: str) -> str:
        """Process the action and interact with environment."""
        if not self._environment:
            return "No environment available for interaction."
        
        self._last_action = action_attempt
        agent = self.get_agent()
        
        # Process the action through handlers
        result = self._process_action(agent, action_attempt)
        self._last_result = result
        
        # Return the result message
        return result.observation or result.message
    
    def _process_action(self, agent, action: str) -> ActionResult:
        """Process an agent's action through the registered handlers."""
        action_lower = action.lower().strip()
        
        # Try each handler in order
        for handler in self._handlers:
            if handler.can_handle(action_lower):
                return handler.process(agent, action, action_lower, self._environment)
        
        # No handler found
        return ActionResult(
            False, 
            f"{agent.name} attempted an unclear action. Please choose from: examine room, examine [object], search [object], take [item], read [item], speak to [agent], or wait and observe."
        )
    
    def _register_default_handlers(self) -> None:
        """Register the default action handlers."""
        self._handlers = [
            WaitActionHandler(),
            ExamineActionHandler(),
            SearchActionHandler(),
            SpeakActionHandler(),
            TakeActionHandler(),
            ReadActionHandler(),
        ]
    
    def get_last_action(self) -> str:
        """Get the last action processed."""
        return self._last_action
    
    def get_last_result(self) -> Optional[ActionResult]:
        """Get the result of the last action."""
        return self._last_result
    
    def get_state(self) -> ComponentState:
        """Get the component's state."""
        return {
            'last_action': self._last_action,
            'last_result': {
                'success': self._last_result.success if self._last_result else False,
                'message': self._last_result.message if self._last_result else "",
                'observation': self._last_result.observation if self._last_result else None,
                'found_items': self._last_result.found_items if self._last_result else None,
            } if self._last_result else None,
        }
    
    def set_state(self, state: ComponentState) -> None:
        """Set the component's state."""
        self._last_action = str(state.get('last_action', ""))
        
        last_result_data = state.get('last_result')
        if isinstance(last_result_data, dict):
            observation = last_result_data.get('observation')
            found_items = last_result_data.get('found_items')
            
            self._last_result = ActionResult(
                success=bool(last_result_data.get('success', False)),
                message=str(last_result_data.get('message', "")),
                observation=str(observation) if observation is not None else None,
                found_items=[str(item) for item in found_items] if isinstance(found_items, list) else None,
            )
        else:
            self._last_result = None


class ActionComponentFactory:
    """Factory class for creating action-related components."""
    
    @staticmethod
    def create_action_processing_component(config: Dict[str, Any], environment: Optional[Any] = None) -> ActionProcessingComponent:
        """Create an action processing component from configuration.
        
        Args:
            config: Configuration dictionary
            environment: Environment object to connect to the component
            
        Returns:
            A configured ActionProcessingComponent
        """
        component = ActionProcessingComponent(environment)
        
        # Configure custom handlers if specified
        custom_handlers = config.get('custom_handlers', [])
        for handler_config in custom_handlers:
            handler = ActionComponentFactory._create_handler(handler_config)
            if handler:
                component.add_handler(handler)
        
        return component
    
    @staticmethod
    def _create_handler(handler_config: Dict[str, Any]) -> Optional[ActionHandler]:
        """Create a custom action handler from configuration."""
        handler_type = handler_config.get('type', '')
        
        if handler_type == 'custom':
            # Import and instantiate custom handler class
            class_path = handler_config.get('class_path', '')
            if not class_path:
                return None
            
            try:
                module_path, class_name = class_path.rsplit('.', 1)
                module = __import__(module_path, fromlist=[class_name])
                handler_class = getattr(module, class_name)
                
                # Get constructor arguments
                constructor_args = handler_config.get('constructor_args', {})
                
                return handler_class(**constructor_args)
            except (ImportError, AttributeError, TypeError):
                return None
        
        return None
