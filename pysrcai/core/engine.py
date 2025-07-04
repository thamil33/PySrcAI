"""Simulation engine for PySrcAI agents."""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from ..agents.base.agent import ActionSpec, OutputType
from ..core.objects import Environment, create_environment_from_config


@dataclass
class ActionResult:
    """Result of processing an agent action."""
    success: bool
    message: str
    observation: Optional[str] = None
    found_items: Optional[List[str]] = None


class EnvironmentContextGenerator:
    """Handles generation of environment context for agents and archons."""
    
    def __init__(self, environment: Environment):
        self.environment = environment
    
    def get_agent_observation(self, agent_name: str, turn: int, inventory: Optional[List[str]] = None) -> str:
        """Generate observation for a specific agent."""
        parts = [f"Turn {turn}: It's your turn to act."]
        
        # Add inventory information
        if inventory:
            item_names = [self.environment.items[item_id].name for item_id in inventory 
                         if item_id in self.environment.items]
            parts.append(f"Your inventory: {', '.join(item_names)}")
        else:
            parts.append("Your inventory is empty.")
        
        # Add location description
        location_desc = self._get_location_description()
        parts.append(location_desc)
        
        return " ".join(parts)
    
    def get_archon_context(self, scenario_state: Dict[str, Any]) -> str:
        """Generate detailed environment context for archon analysis."""
        if not self.environment.active_location:
            return "CURRENT ENVIRONMENT STATE:\nNo active location set.\n"
        
        active_loc = self.environment.locations[self.environment.active_location]
        context = [
            "CURRENT ENVIRONMENT STATE:",
            f"Location: {active_loc.name}",
            f"Description: {active_loc.description}",
            "",
            "Objects in location:"
        ]
        
        # Add object descriptions
        for obj_id, obj in active_loc.objects.items():
            context.append(f"- {obj_id}: {obj.name} - {obj.description}")
            context.extend(self._get_object_details(obj_id, obj))
        
        # Add visible items
        context.extend(self._get_visible_items_section())
        
        # Add hidden items info
        context.extend(self._get_hidden_items_section())
        
        # Add inventories
        context.extend(self._get_inventory_section(scenario_state))
        
        # Add recent developments
        context.extend(self._get_recent_developments(scenario_state))
        
        return "\n".join(context)
    
    def get_action_prompt(self, agent, other_agents: List) -> str:
        """Generate action prompt with available options."""
        if not self.environment.active_location:
            return self._get_basic_action_prompt(other_agents)
        
        active_loc = self.environment.locations[self.environment.active_location]
        prompt = [
            f"Step {getattr(agent, 'turn', 1)}: Choose ONE specific action from the available options based on your current environment:\n",
            f"CURRENT LOCATION: {active_loc.name}",
            f"{active_loc.description}\n",
            "AVAILABLE ACTIONS:\n"
        ]
        
        # Observation actions
        prompt.append("**OBSERVATION:**")
        prompt.append("- examine room (look around the current location)")
        for obj_id, obj in active_loc.objects.items():
            prompt.append(f"- examine {obj_id} (look closely at the {obj.name})")
        
        # Search actions
        prompt.extend(self._get_search_actions(active_loc))
        
        # Item interactions
        prompt.extend(self._get_item_interactions())
        
        # Social interactions
        prompt.extend(self._get_social_actions(other_agents))
        
        prompt.extend([
            "- wait and observe (do nothing this turn)\n",
            "**IMPORTANT**: Choose exactly ONE action using the format shown above. "
            "Only interact with objects and items that are actually present in your environment."
        ])
        
        return "\n".join(prompt)
    
    def _get_location_description(self) -> str:
        """Get description of current location."""
        if not self.environment.active_location:
            return "You are in an undefined location."
        
        active_location = self.environment.locations.get(self.environment.active_location)
        if not active_location:
            return "You are in an undefined location."
        
        desc = f"You are in the {active_location.name}. {active_location.description}"
        
        objects = [obj.name for obj in active_location.objects.values()]
        if objects:
            desc += f" You see the following objects: {', '.join(objects)}."
        else:
            desc += " There are no notable objects here."
        
        return desc
    
    def _get_object_details(self, obj_id: str, obj) -> List[str]:
        """Get detailed information about an object."""
        details = []
        
        if obj.properties.get("searchable", False):
            contents = obj.properties.get('contents', [])
            discovered = []
            hidden = []
            
            for item_id in contents:
                if item_id in self.environment.items:
                    item = self.environment.items[item_id]
                    if item.properties.get("location", "hidden") == "visible":
                        discovered.append(item_id)
                    else:
                        hidden.append(item_id)
            
            if discovered:
                details.append(f"  * Previously discovered items: {discovered}")
            if hidden:
                details.append(f"  * May contain hidden items: {len(hidden)} remaining")
        
        if obj.properties.get("examination_detail"):
            details.append(f"  * Detailed examination reveals: {obj.properties['examination_detail']}")
        
        return details
    
    def _get_visible_items_section(self) -> List[str]:
        """Get section describing visible items."""
        visible_items = [
            item_id for item_id, item in self.environment.items.items() 
            if item.properties.get("location", "hidden") == "visible"
        ]
        
        section = ["", "Available items (CURRENTLY VISIBLE):"]
        if visible_items:
            for item_id in visible_items:
                item = self.environment.items[item_id]
                section.append(f"- {item_id}: {item.name} - {item.description}")
                if item.properties.get("readable", False):
                    section.append(f"  * Readable content: {item.properties.get('content', 'No content')}")
        else:
            section.append("- No visible items in the current location")
        
        return section
    
    def _get_hidden_items_section(self) -> List[str]:
        """Get section describing hidden items."""
        hidden_items = [
            item_id for item_id, item in self.environment.items.items() 
            if item.properties.get("location", "visible") == "hidden"
        ]
        
        if hidden_items:
            return [
                "",
                "Hidden items (NOT YET DISCOVERED):",
                f"- There are {len(hidden_items)} hidden items that can be found by searching"
            ]
        return []
    
    def _get_inventory_section(self, scenario_state: Dict[str, Any]) -> List[str]:
        """Get section describing agent inventories."""
        if "inventory" not in scenario_state:
            return []
        
        section = ["", "Agent inventories:"]
        for agent_name, inventory in scenario_state["inventory"].items():
            if inventory:
                item_names = [
                    self.environment.items[item_id].name 
                    for item_id in inventory 
                    if item_id in self.environment.items
                ]
                section.append(f"- {agent_name}: {', '.join(item_names)}")
            else:
                section.append(f"- {agent_name}: empty")
        
        return section
    
    def _get_recent_developments(self, scenario_state: Dict[str, Any]) -> List[str]:
        """Get section describing recent developments."""
        if 'conversation_log' not in scenario_state:
            return []
        
        recent_logs = scenario_state['conversation_log'][-3:] if scenario_state['conversation_log'] else []
        if not recent_logs:
            return []
        
        section = ["", "RECENT DEVELOPMENTS:"]
        for entry in recent_logs:
            if 'agent' in entry and 'action' in entry and entry.get('type') != 'analysis':
                section.append(f"- {entry['agent']} {entry['action'][:50]}...")
        
        return section
    
    def _get_basic_action_prompt(self, other_agents: List) -> str:
        """Get basic action prompt when no location is set."""
        prompt = ["**MOVEMENT & OBSERVATION:**", "- examine room (look around the current location)"]
        prompt.extend(self._get_social_actions(other_agents))
        return "\n".join(prompt)
    
    def _get_search_actions(self, active_loc) -> List[str]:
        """Get search action options."""
        searchable_objects = [
            (obj_id, obj) for obj_id, obj in active_loc.objects.items() 
            if obj.properties.get("searchable", False)
        ]
        
        section = ["", "**SEARCH ACTIONS:**"]
        if searchable_objects:
            for obj_id, obj in searchable_objects:
                section.append(f"- search {obj_id} (search the {obj.name} for hidden items)")
        else:
            section.append("- No searchable objects available")
        
        return section
    
    def _get_item_interactions(self) -> List[str]:
        """Get item interaction options."""
        visible_items = [
            (item_id, item) for item_id, item in self.environment.items.items() 
            if item.properties.get("location", "visible") == "visible"
        ]
        
        section = ["", "**ITEM INTERACTIONS:**"]
        if visible_items:
            for item_id, item in visible_items:
                section.append(f"- take {item_id} (pick up the {item.name})")
                if item.properties.get("readable", False):
                    section.append(f"- read {item_id} (read the {item.name})")
        else:
            section.append("- No visible items to interact with")
        
        return section
    
    def _get_social_actions(self, other_agents: List) -> List[str]:
        """Get social interaction options."""
        section = ["", "**SOCIAL INTERACTIONS:**"]
        for other_agent in other_agents:
            section.append(f"- speak to {other_agent.name} about [topic]")
        section.append("- wait and observe (do nothing this turn)")
        return section


class ActionProcessor:
    """Handles parsing and processing of agent actions."""
    
    def __init__(self, environment: Environment):
        self.environment = environment
    
    def process_action(self, agent, action: str) -> ActionResult:
        """Process an agent's action and return the result."""
        action_lower = action.lower().strip()
        
        # Handle different action types
        if self._is_wait_action(action_lower):
            return ActionResult(True, f"{agent.name} chooses to do nothing and observe their surroundings.")
        
        if "examine room" in action_lower:
            return self._process_examine_room(agent)
        
        if "examine" in action_lower:
            return self._process_examine_object(agent, action_lower)
        
        if "search" in action_lower:
            return self._process_search(agent, action_lower)
        
        if any(word in action_lower for word in ["speak", "talk", "say", "tell"]):
            return self._process_speak(agent, action, action_lower)
        
        if any(word in action_lower for word in ["take", "pick up", "grab", "get"]):
            return self._process_take_item(agent, action_lower)
        
        if "read" in action_lower:
            return self._process_read_item(agent, action_lower)
        
        return ActionResult(False, f"{agent.name} attempted an unclear action. Please choose from: examine room, examine [object], search [object], take [item], read [item], speak to [agent], or wait and observe.")
    
    def _is_wait_action(self, action_lower: str) -> bool:
        """Check if action is a wait/observe action."""
        return any(phrase in action_lower for phrase in ["wait", "do nothing", "observe", "wait and observe"])
    
    def _process_examine_room(self, agent) -> ActionResult:
        """Process examine room action."""
        if not self.environment.active_location:
            return ActionResult(True, f"{agent.name} looks around but sees nothing distinctive.")
        
        result = self.environment.process_interaction(
            agent_id=agent.name,
            action="examine",
            target_id=self.environment.active_location
        )
        return ActionResult(
            success=result.get("success", False),
            message=result.get("message", ""),
            observation=result.get("observation", "")
        )
    
    def _process_examine_object(self, agent, action_lower: str) -> ActionResult:
        """Process examine object action."""
        for obj_id in self._get_all_object_ids():
            obj_name = self._get_object_name(obj_id).lower()
            if obj_name in action_lower or obj_id.lower() in action_lower:
                result = self.environment.process_interaction(
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
    
    def _process_search(self, agent, action_lower: str) -> ActionResult:
        """Process search action."""
        for obj_id in self._get_all_object_ids():
            obj_name = self._get_object_name(obj_id).lower()
            if obj_name in action_lower or obj_id.lower() in action_lower:
                return self._handle_search_interaction(agent, obj_id)
        
        return ActionResult(False, "You don't see that object here.")
    
    def _handle_search_interaction(self, agent, obj_id: str) -> ActionResult:
        """Handle the actual search interaction logic."""
        if not self.environment.active_location:
            return self._process_generic_search(agent, obj_id)
        
        active_loc = self.environment.locations[self.environment.active_location]
        if obj_id not in active_loc.objects:
            return self._process_generic_search(agent, obj_id)
        
        obj = active_loc.objects[obj_id]
        contents = obj.properties.get("contents", [])
        hidden_items = []
        already_found = []
        
        # Check what's already visible vs what's still hidden
        for item_id in contents:
            if item_id in self.environment.items:
                item = self.environment.items[item_id]
                if item.properties.get("location", "hidden") == "hidden":
                    hidden_items.append(item_id)
                else:
                    already_found.append(item_id)
        
        # If all items already found, provide that feedback
        if not hidden_items and already_found:
            names = [self.environment.items[item_id].name for item_id in already_found]
            return ActionResult(
                success=True,
                message=f"You already searched this {obj.name} and found: {', '.join(names)}.",
                observation=f"You search the {obj.name} again, but find nothing new beyond what you already discovered."
            )
        
        # Process normal search interaction
        result = self.environment.process_interaction(
            agent_id=agent.name,
            action="search",
            target_id=obj_id
        )
        
        # Handle search results
        if result.get("success", False) and "found_items" in result:
            newly_found = []
            for item_id in result["found_items"]:
                if item_id in self.environment.items:
                    item = self.environment.items[item_id]
                    if item.properties.get("location", "hidden") == "hidden":
                        item.properties["location"] = "visible"
                        newly_found.append(item_id)
            
            # Update result message if some items were already found
            if not newly_found and already_found:
                names = [self.environment.items[item_id].name for item_id in already_found]
                result["observation"] = f"You search the {obj.name} again, but find nothing new beyond what you already discovered: {', '.join(names)}"
        
        return ActionResult(
            success=result.get("success", False),
            message=result.get("message", ""),
            observation=result.get("observation", ""),
            found_items=result.get("found_items", [])
        )
    
    def _process_generic_search(self, agent, obj_id: str) -> ActionResult:
        """Process search for non-location objects."""
        result = self.environment.process_interaction(
            agent_id=agent.name,
            action="search",
            target_id=obj_id
        )
        
        # Handle search results
        if result.get("success", False) and "found_items" in result:
            for item_id in result["found_items"]:
                if item_id in self.environment.items:
                    self.environment.items[item_id].properties["location"] = "visible"
        
        return ActionResult(
            success=result.get("success", False),
            message=result.get("message", ""),
            observation=result.get("observation", ""),
            found_items=result.get("found_items", [])
        )
    
    def _process_speak(self, agent, action: str, action_lower: str) -> ActionResult:
        """Process speak action."""
        # This would need access to other agents - simplified for now
        about_index = action_lower.find("about")
        if about_index != -1:
            topic = action[about_index + 5:].strip()
            message = f"Hello, I'd like to discuss {topic}"
        else:
            message = "Hello, let's talk."
        
        return ActionResult(True, f"{agent.name} says: {message}")
    
    def _process_take_item(self, agent, action_lower: str) -> ActionResult:
        """Process take item action."""
        for item_id in self.environment.items:
            if item_id.lower() in action_lower or self.environment.items[item_id].name.lower() in action_lower:
                item = self.environment.items[item_id]
                if item.properties.get("location", "visible") == "visible":
                    result = self.environment.process_interaction(
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
    
    def _process_read_item(self, agent, action_lower: str) -> ActionResult:
        """Process read item action."""
        for item_id in self.environment.items:
            if item_id.lower() in action_lower or self.environment.items[item_id].name.lower() in action_lower:
                item = self.environment.items[item_id]
                
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
    
    def _get_all_object_ids(self) -> List[str]:
        """Get all object IDs in the environment for action matching."""
        object_ids = []
        object_ids.extend(self.environment.items.keys())
        object_ids.extend(self.environment.locations.keys())
        
        for location in self.environment.locations.values():
            object_ids.extend(location.objects.keys())
        
        return object_ids
    
    def _get_object_name(self, obj_id: str) -> str:
        """Get the display name of an object by its ID."""
        if obj_id in self.environment.items:
            return self.environment.items[obj_id].name
        
        if obj_id in self.environment.locations:
            return self.environment.locations[obj_id].name
        
        for location in self.environment.locations.values():
            if obj_id in location.objects:
                return location.objects[obj_id].name
        
        return obj_id


class SimulationEngine:
    """Base class for simulation engines. Handles the simulation lifecycle and agent management."""
    
    def __init__(self, agents=None, state=None):
        self.agents = agents if agents is not None else []
        self.state = state if state is not None else {}
        self.scenario_state = self.state.copy()
        self.running = False

    def initialize(self):
        """Prepare the simulation (reset state, initialize agents, etc.)."""
        pass

    def step(self):
        """Advance the simulation by one step/tick."""
        pass

    def run(self, steps=None):
        """Run the simulation for a given number of steps, or until stopped."""
        self.running = True
        self.initialize()
        step_count = 0
        while self.running and (steps is None or step_count < steps):
            self.step()
            step_count += 1

    def shutdown(self):
        """Clean up resources and stop the simulation."""
        self.running = False


class SequentialEngine(SimulationEngine):
    """Concrete engine for sequential, turn-based agent interactions with optional archon observation."""
    
    def __init__(self, agents=None, archon=None, state=None, config=None):
        super().__init__(agents=agents, state=state)
        self.archon = archon
        self.turn = 0
        self.config = config or {}
        
        # Initialize configuration
        self._init_config()
        
        # Initialize environment
        self._init_environment()
        
        # Initialize components
        self.context_generator = EnvironmentContextGenerator(self.environment)
        self.action_processor = ActionProcessor(self.environment)

    def _init_config(self):
        """Initialize configuration settings."""
        self.response_word_limit = None
        engine_cfg = self.config.get('engine', {}) if self.config else {}
        if 'response_word_limit' in engine_cfg:
            self.response_word_limit = engine_cfg['response_word_limit']

    def _init_environment(self):
        """Initialize environment from config."""
        if "scenario" in self.config and "initial_state" in self.config["scenario"]:
            print("[Engine] Loading environment from config...")
            self.environment = create_environment_from_config(self.config)
            print(f"[Engine] Environment loaded. Active location: {self.environment.active_location}")
            print(f"[Engine] Available locations: {list(self.environment.locations.keys())}")
            print(f"[Engine] Available items: {list(self.environment.items.keys())}")
        else:
            print("[Engine] No scenario config found, creating empty environment")
            self.environment = Environment()
        
        # Add environment to scenario state
        self.scenario_state["environment"] = self.environment.to_dict()

    def initialize(self):
        """Initialize the simulation."""
        self.turn = 0
        print("[Engine] Initialized. Agents:", [a.name for a in self.agents])
        if self.archon:
            print(f"[Engine] Archon: {self.archon.name}")
        
        self._notify_agents_of_environment()

    def _notify_agents_of_environment(self):
        """Notify all agents and archon of the initial environment state."""
        if not self.environment.active_location:
            return
        
        active_location = self.environment.locations.get(self.environment.active_location)
        if not active_location:
            return
        
        desc = f"You are in the {active_location.name}. {active_location.description}"
        
        objects = [obj.name for obj in active_location.objects.values()]
        if objects:
            desc += f" You see the following objects: {', '.join(objects)}."
        else:
            desc += " There are no notable objects here."
        
        # Inform agents and archon
        for agent in self.agents:
            if hasattr(agent, 'observe'):
                agent.observe(desc)
        
        if self.archon and hasattr(self.archon, 'observe'):
            self.archon.observe(desc)

    def step(self):
        """Execute one simulation step."""
        print(f"\n[Engine] Step {self.turn+1}")
        
        # Initialize conversation log if needed
        if 'conversation_log' not in self.scenario_state:
            self.scenario_state['conversation_log'] = []
        
        # Process each agent's turn
        for agent in self.agents:
            self._process_agent_turn(agent)
        
        # Final archon analysis
        self._process_archon_analysis()
        
        self.turn += 1

    def _process_agent_turn(self, agent):
        """Process a single agent's turn."""
        # Create observation for agent
        inventory = self._get_agent_inventory(agent)
        observation = self.context_generator.get_agent_observation(
            agent.name, self.turn + 1, inventory
        )
        
        if hasattr(agent, 'observe'):
            agent.observe(observation)
        
        # Get agent action
        if hasattr(agent, 'act'):
            action = self._get_agent_action(agent)
            print(f"{agent.name} acts: {action}")
            
            # Process action through environment
            result = self.action_processor.process_action(agent, action)
            
            # Update environment state
            self.scenario_state["environment"] = self.environment.to_dict()
            
            # Handle environmental feedback
            self._handle_environmental_feedback(agent, action, result)
            
            # Log the action
            self._log_action(agent, action, result)

    def _get_agent_inventory(self, agent) -> Optional[List[str]]:
        """Get agent's inventory."""
        if "inventory" in self.scenario_state and agent.name in self.scenario_state["inventory"]:
            return self.scenario_state["inventory"][agent.name]
        return None

    def _get_agent_action(self, agent) -> str:
        """Get action from agent."""
        word_limit = getattr(agent, 'word_limit', self.response_word_limit)
        prompt = self.context_generator.get_action_prompt(agent, [a for a in self.agents if a != agent])
        
        if word_limit:
            prompt += f" (Respond in {word_limit} words or less.)"
        
        action_spec = ActionSpec(
            call_to_action=prompt,
            output_type=OutputType.FREE,
            tag="demo"
        )
        
        return agent.act(action_spec)

    def _handle_environmental_feedback(self, agent, action: str, result: ActionResult):
        """Handle environmental feedback for agent action."""
        if self.archon and hasattr(self.archon, 'act'):
            self._get_archon_feedback(agent, action)
        else:
            # Fallback: provide direct environment result
            result_message = result.observation or result.message
            print(f"[Environment] {result_message}")
            if hasattr(agent, 'observe'):
                agent.observe(f"Environmental result: {result_message}")

    def _get_archon_feedback(self, agent, action: str):
        """Get environmental feedback from archon."""
        if not self.archon:
            return
            
        word_limit_archon = getattr(self.archon, 'word_limit', self.response_word_limit)
        
        feedback_prompt = self._build_archon_feedback_prompt(agent, action)
        
        if word_limit_archon:
            feedback_prompt += f" (Respond in {word_limit_archon} words or less.)"
        
        environmental_feedback = self.archon.act(ActionSpec(
            call_to_action=feedback_prompt,
            output_type=OutputType.FREE,
            tag="environmental_feedback"
        ))
        
        print(f"[Environment] {environmental_feedback}")
        
        # Let the agent observe the environmental outcome
        if hasattr(agent, 'observe'):
            agent.observe(f"Environmental result: {environmental_feedback}")
        
        # Let other agents observe what happened
        for other_agent in self.agents:
            if other_agent != agent and hasattr(other_agent, 'observe'):
                other_agent.observe(f"{agent.name} {environmental_feedback}")

    def _build_archon_feedback_prompt(self, agent, action: str) -> str:
        """Build prompt for archon environmental feedback."""
        context = self.context_generator.get_archon_context(self.scenario_state)
        
        return (
            f"ENVIRONMENTAL NARRATOR ROLE: Process {agent.name}'s action: '{action}'\n\n"
            "CRITICAL: You must base your narrative ONLY on the actual environment state below:\n\n"
            f"{context}\n\n"
            f"Based ONLY on the above environment state, describe what {agent.name} experiences from their action: '{action}'\n\n"
            "RULES:\n"
            "- Only reference objects and locations that actually exist in the environment state\n"
            "- Do NOT invent new objects, rooms, or magical elements\n"
            "- If the action reveals a previously hidden item, describe it based on its actual properties\n"
            "- Only describe items as 'discovered' if they were truly hidden before this action\n"
            "- Keep descriptions grounded in the simple, mundane environment provided\n"
            "- If the action cannot be performed, explain why based on the actual environment\n"
            "- CONSISTENCY: Refer to objects by the same name and properties as in previous descriptions\n\n"
            "Describe the immediate, concrete result of the action. Focus on what changed or what was observed."
        )

    def _log_action(self, agent, action: str, result: ActionResult):
        """Log agent action in scenario state."""
        self.scenario_state['conversation_log'].append({
            'turn': self.turn + 1,
            'agent': agent.name,
            'action': action,
            'environment_result': result.observation or result.message
        })

    def _process_archon_analysis(self):
        """Process final archon analysis of the round."""
        if not self.archon or not hasattr(self.archon, 'act'):
            return
        
        word_limit = getattr(self.archon, 'word_limit', self.response_word_limit)
        context = self.context_generator.get_archon_context(self.scenario_state)
        
        analysis_prompt = (
            f"Step {self.turn+1}: Analyze this round's activities based on the actual environment state:\n\n"
            f"{context}\n\n"
            "Summarize what actually happened this round based on the real environment state above. "
            "Focus on concrete actions and discoveries, not fictional elements.\n\n"
            "RULES:\n"
            "- Only mention items as 'discovered' if they were NEWLY found this round (check what was already visible)\n"
            "- Be specific about which agent found what\n" 
            "- Refer consistently to objects with their proper names\n"
            "- Don't mention objects or details not present in the environment state"
        )
        
        if word_limit:
            analysis_prompt += f" (Respond in {word_limit} words or less.)"
        
        analysis = self.archon.act(ActionSpec(
            call_to_action=analysis_prompt,
            output_type=OutputType.FREE,
            tag="archon_analysis"
        ))
        
        print(f"[Archon] {self.archon.name} analyzes: {analysis}")
        self.scenario_state['conversation_log'].append({
            'turn': self.turn + 1,
            'agent': self.archon.name,
            'action': analysis,
            'type': 'analysis'
        })