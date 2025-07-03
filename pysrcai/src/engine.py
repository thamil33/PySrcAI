from pysrcai.src.agents import ActionSpec, OutputType
from pysrcai.src.environment.objects import Environment, create_environment_from_config

class SimulationEngine:
    """
    Base class for simulation engines. Handles the simulation lifecycle and agent management.
    """
    def __init__(self, agents=None, state=None):
        self.agents = agents if agents is not None else []
        self.state = state if state is not None else {}
        self.scenario_state = self.state.copy()  # scenario_state is the evolving state
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
    """
    Concrete engine for sequential, turn-based agent interactions with optional archon observation.
    """
    def __init__(self, agents=None, archon=None, state=None, config=None):
        super().__init__(agents=agents, state=state)
        self.archon = archon
        self.turn = 0
        self.config = config or {}
        # Read word limit from config if present
        self.response_word_limit = None
        engine_cfg = self.config.get('engine', {}) if self.config else {}
        if 'response_word_limit' in engine_cfg:
            self.response_word_limit = engine_cfg['response_word_limit']
        
        # Initialize environment from config
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
        self.turn = 0
        # Optionally, reset agent and archon state here
        print("[Engine] Initialized. Agents:", [a.name for a in self.agents])
        if self.archon:
            print(f"[Engine] Archon: {self.archon.name}")
        
        # Generate environment description for agents
        if self.environment.active_location and self.environment.active_location in self.environment.locations:
            active_location = self.environment.locations[self.environment.active_location]
            desc = f"You are in the {active_location.name}. {active_location.description}"
            
            # List objects in the location
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

    def _get_environment_context_for_archon(self):
        """Generate a context description of the current environment state for the Archon."""
        context = "CURRENT ENVIRONMENT STATE:\n"
        
        if self.environment.active_location and self.environment.active_location in self.environment.locations:
            active_loc = self.environment.locations[self.environment.active_location]
            context += f"Location: {active_loc.name}\n"
            context += f"Description: {active_loc.description}\n"
            
            # List visible objects in the location
            context += "\nObjects in location:\n"
            for obj_id, obj in active_loc.objects.items():
                context += f"- {obj_id}: {obj.name} - {obj.description}\n"
                if obj.properties.get("searchable", False):
                    contents = obj.properties.get('contents', [])
                    discovered_items = []
                    hidden_items = []
                    
                    # Check which items are visible vs still hidden
                    for item_id in contents:
                        if item_id in self.environment.items:
                            item = self.environment.items[item_id]
                            if item.properties.get("location", "hidden") == "visible":
                                discovered_items.append(item_id)
                            else:
                                hidden_items.append(item_id)
                    
                    if discovered_items:
                        context += f"  * Previously discovered items: {discovered_items}\n"
                    if hidden_items:
                        context += f"  * May contain hidden items: {len(hidden_items)} remaining\n"
                
                if obj.properties.get("examination_detail"):
                    context += f"  * Detailed examination reveals: {obj.properties['examination_detail']}\n"
            
            # List available items (visible ones)
            visible_items = [item_id for item_id, item in self.environment.items.items() 
                           if item.properties.get("location", "hidden") == "visible"]
            
            context += "\nAvailable items (CURRENTLY VISIBLE):\n"
            if visible_items:
                for item_id in visible_items:
                    item = self.environment.items[item_id]
                    context += f"- {item_id}: {item.name} - {item.description}\n"
                    if item.properties.get("readable", False):
                        context += f"  * Readable content: {item.properties.get('content', 'No content')}\n"
            else:
                context += "- No visible items in the current location\n"
                
            # List hidden items (for reference)
            hidden_items = [item_id for item_id, item in self.environment.items.items() 
                          if item.properties.get("location", "visible") == "hidden"]
            if hidden_items:
                context += "\nHidden items (NOT YET DISCOVERED):\n"
                context += f"- There are {len(hidden_items)} hidden items that can be found by searching\n"
                    
            # Show agent inventories if any
            if "inventory" in self.scenario_state:
                context += "\nAgent inventories:\n"
                for agent_name, inventory in self.scenario_state["inventory"].items():
                    if inventory:
                        item_names = [self.environment.items[item_id].name for item_id in inventory if item_id in self.environment.items]
                        context += f"- {agent_name}: {', '.join(item_names)}\n"
                    else:
                        context += f"- {agent_name}: empty\n"
                        
            # Add recent actions/discoveries for context
            if 'conversation_log' in self.scenario_state:
                recent_logs = self.scenario_state['conversation_log'][-3:] if self.scenario_state['conversation_log'] else []
                if recent_logs:
                    context += "\nRECENT DEVELOPMENTS:\n"
                    for entry in recent_logs:
                        if 'agent' in entry and 'action' in entry:
                            if entry.get('type') != 'analysis':  # Skip Archon analyses
                                context += f"- {entry['agent']} {entry['action'][:50]}...\n"
        else:
            context += "No active location set.\n"
            
        return context

    def _make_observation_for_agent(self, agent, step_info=""):
        """Create an observation for a specific agent based on current environment state."""
        if self.environment.active_location and self.environment.active_location in self.environment.locations:
            active_location = self.environment.locations[self.environment.active_location]
            observation = f"{step_info} You are in the {active_location.name}. {active_location.description}"
            return observation
        else:
            return f"{step_info} You are in an undefined location."

    def step(self):
        print(f"\n[Engine] Step {self.turn+1}")
        # Example: add a conversation log to scenario_state
        if 'conversation_log' not in self.scenario_state:
            self.scenario_state['conversation_log'] = []
            
        for agent in self.agents:
            # Create and send observation
            observation_parts = [f"Turn {self.turn+1}: It's your turn to act."]
            
            # Add inventory information if available
            if "inventory" in self.scenario_state and agent.name in self.scenario_state["inventory"]:
                inventory = self.scenario_state["inventory"][agent.name]
                if inventory:
                    item_names = [self.environment.items[item_id].name for item_id in inventory if item_id in self.environment.items]
                    observation_parts.append(f"Your inventory: {', '.join(item_names)}")
                else:
                    observation_parts.append("Your inventory is empty.")
            
            # Add environment observation
            env_observation = self._make_observation_for_agent(agent, "")
            observation_parts.append(env_observation)
            
            full_observation = " ".join(observation_parts)
            
            if hasattr(agent, 'observe'):
                agent.observe(full_observation)
                        
            if hasattr(agent, 'act'):
                # Per-agent word limit override
                word_limit = getattr(agent, 'word_limit', None)
                if word_limit is None:
                    word_limit = self.response_word_limit
                
                # Build environment-aware action prompt with actual objects
                prompt = f"Step {self.turn+1}: Choose ONE specific action from the available options based on your current environment:\n\n"
                
                # Add current environment context for the agent
                if self.environment.active_location and self.environment.active_location in self.environment.locations:
                    active_loc = self.environment.locations[self.environment.active_location]
                    prompt += f"CURRENT LOCATION: {active_loc.name}\n"
                    prompt += f"{active_loc.description}\n\n"
                    
                    prompt += "AVAILABLE ACTIONS:\n\n"
                    prompt += "**OBSERVATION:**\n"
                    prompt += "- examine room (look around the current location)\n"
                    
                    # List actual objects that can be examined
                    for obj_id, obj in active_loc.objects.items():
                        prompt += f"- examine {obj_id} (look closely at the {obj.name})\n"
                    
                    prompt += "\n**SEARCH ACTIONS:**\n"
                    # List searchable objects
                    searchable_objects = [(obj_id, obj) for obj_id, obj in active_loc.objects.items() 
                                        if obj.properties.get("searchable", False)]
                    if searchable_objects:
                        for obj_id, obj in searchable_objects:
                            prompt += f"- search {obj_id} (search the {obj.name} for hidden items)\n"
                    else:
                        prompt += "- No searchable objects available\n"
                    
                    prompt += "\n**ITEM INTERACTIONS:**\n"
                    # List visible items
                    visible_items = [(item_id, item) for item_id, item in self.environment.items.items() 
                                   if item.properties.get("location", "visible") == "visible"]
                    if visible_items:
                        for item_id, item in visible_items:
                            prompt += f"- take {item_id} (pick up the {item.name})\n"
                            if item.properties.get("readable", False):
                                prompt += f"- read {item_id} (read the {item.name})\n"
                    else:
                        prompt += "- No visible items to interact with\n"
                else:
                    prompt += "**MOVEMENT & OBSERVATION:**\n"
                    prompt += "- examine room (look around the current location)\n"
                
                prompt += "\n**SOCIAL INTERACTIONS:**\n"
                for other_agent in self.agents:
                    if other_agent != agent:
                        prompt += f"- speak to {other_agent.name} about [topic]\n"
                
                prompt += "- wait and observe (do nothing this turn)\n\n"
                prompt += "**IMPORTANT**: Choose exactly ONE action using the format shown above. "
                prompt += "Only interact with objects and items that are actually present in your environment."
                

                if word_limit:
                    prompt += f"(Respond in {word_limit} words or less.)"

                action_spec = ActionSpec(
                    call_to_action=prompt,
                    output_type=OutputType.FREE,
                    tag="demo"
                )
                action = agent.act(action_spec)
                print(f"{agent.name} acts: {action}")
                
                # PROCESS ACTION THROUGH ENVIRONMENT IMMEDIATELY
                environment_result = self.process_agent_action(agent, action)
                
                # Update scenario state with environment changes after each action
                self.scenario_state["environment"] = self.environment.to_dict()
                
                # Have Archon provide environmental feedback if available
                if self.archon and hasattr(self.archon, 'act'):
                    word_limit_archon = getattr(self.archon, 'word_limit', None)
                    if word_limit_archon is None:
                        word_limit_archon = self.response_word_limit
                    
                    # Build environment-grounded feedback prompt
                    feedback_prompt = f"ENVIRONMENTAL NARRATOR ROLE: Process {agent.name}'s action: '{action}'\n\n"
                    feedback_prompt += "CRITICAL: You must base your narrative ONLY on the actual environment state below:\n\n"
                    
                    # Add current environment state context
                    feedback_prompt += self._get_environment_context_for_archon()
                    
                    feedback_prompt += f"\nBased ONLY on the above environment state, describe what {agent.name} experiences from their action: '{action}'\n\n"
                    feedback_prompt += "RULES:\n"
                    feedback_prompt += "- Only reference objects and locations that actually exist in the environment state\n"
                    feedback_prompt += "- Do NOT invent new objects, rooms, or magical elements\n"
                    feedback_prompt += "- If the action reveals a previously hidden item, describe it based on its actual properties\n"
                    feedback_prompt += "- Only describe items as 'discovered' if they were truly hidden before this action\n"
                    feedback_prompt += "- Keep descriptions grounded in the simple, mundane environment provided\n"
                    feedback_prompt += "- If the action cannot be performed, explain why based on the actual environment\n"
                    feedback_prompt += "- CONSISTENCY: Refer to objects by the same name and properties as in previous descriptions\n\n"
                    feedback_prompt += "Describe the immediate, concrete result of the action. Focus on what changed or what was observed."
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
                else:
                    # Fallback: provide direct environment result if no archon
                    print(f"[Environment] {environment_result}")
                    if hasattr(agent, 'observe'):
                        agent.observe(f"Environmental result: {environment_result}")
                
                # Log the action and result in scenario_state
                self.scenario_state['conversation_log'].append({
                    'turn': self.turn+1,
                    'agent': agent.name,
                    'action': action,
                    'environment_result': environment_result
                })
                
        # Final Archon analysis of the round - grounded in actual environment state
        if self.archon and hasattr(self.archon, 'act'):
            word_limit = getattr(self.archon, 'word_limit', None)
            if word_limit is None:
                word_limit = self.response_word_limit
            
            # Build environment-grounded analysis prompt
            analysis_prompt = f"Step {self.turn+1}: Analyze this round's activities based on the actual environment state:\n\n"
            analysis_prompt += self._get_environment_context_for_archon()
            analysis_prompt += f"\nSummarize what actually happened this round based on the real environment state above. "
            analysis_prompt += "Focus on concrete actions and discoveries, not fictional elements.\n\n"
            analysis_prompt += "RULES:\n"
            analysis_prompt += "- Only mention items as 'discovered' if they were NEWLY found this round (check what was already visible)\n"
            analysis_prompt += "- Be specific about which agent found what\n" 
            analysis_prompt += "- Refer consistently to objects with their proper names\n"
            analysis_prompt += "- Don't mention objects or details not present in the environment state"
            if word_limit:
                analysis_prompt += f" (Respond in {word_limit} words or less.)"
            analysis = self.archon.act(ActionSpec(
                call_to_action=analysis_prompt,
                output_type=OutputType.FREE,
                tag="archon_analysis"
            ))
            print(f"[Archon] {self.archon.name} analyzes: {analysis}")
            self.scenario_state['conversation_log'].append({
                'turn': self.turn+1,
                'agent': self.archon.name,
                'action': analysis,
                'type': 'analysis'
            })
        self.turn += 1

    def process_agent_action(self, agent, action):
        """
        Process an agent's natural language action and translate it into 
        environment interactions via the Archon.
        
        Args:
            agent: The agent performing the action
            action: The natural language action description
            
        Returns:
            A result message describing the outcome
        """
        # Normalize action for parsing
        action_lower = action.lower()
        result = {"success": False, "message": "You must choose an action. Try: examine, search, take, read, speak, or do nothing."}
        
        # Track if an environmental interaction was found
        interaction_found = False

        # Handle wait/observe actions
        if any(phrase in action_lower for phrase in ["wait", "do nothing", "observe", "wait and observe"]):
            return f"{agent.name} chooses to do nothing and observe their surroundings."

        # Handle examine room action
        if "examine room" in action_lower:
            if self.environment.active_location:
                result = self.environment.process_interaction(
                    agent_id=agent.name,
                    action="examine",
                    target_id=self.environment.active_location
                )
                interaction_found = True
            else:
                return f"{agent.name} looks around but sees nothing distinctive."

        # Handle specific object examination
        elif "examine" in action_lower:
            # Parse "examine [object]" format
            for obj_id in self._get_all_object_ids():
                obj_name = self._get_object_name(obj_id).lower()
                if obj_name in action_lower or obj_id.lower() in action_lower:
                    result = self.environment.process_interaction(
                        agent_id=agent.name,
                        action="examine",
                        target_id=obj_id
                    )
                    interaction_found = True
                    break

        # Handle search actions with improved parsing
        elif "search" in action_lower:
            for obj_id in self._get_all_object_ids():
                obj_name = self._get_object_name(obj_id).lower()
                if obj_name in action_lower or obj_id.lower() in action_lower:
                    # Check if any items in this object are still hidden
                    if obj_id in self.environment.locations.get(self.environment.active_location, {}).objects:
                        obj = self.environment.locations[self.environment.active_location].objects[obj_id]
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
                            result = {
                                "success": True,
                                "message": f"You already searched this {obj.name} and found: {', '.join(names)}.",
                                "observation": f"You search the {obj.name} again, but find nothing new beyond what you already discovered."
                            }
                        else:
                            # Process normal search interaction
                            result = self.environment.process_interaction(
                                agent_id=agent.name,
                                action="search",
                                target_id=obj_id
                            )
                            
                            # Handle search results more explicitly
                            if result.get("success", False) and "found_items" in result:
                                newly_found = []
                                for item_id in result["found_items"]:
                                    if item_id in self.environment.items:
                                        item = self.environment.items[item_id]
                                        
                                        # Only announce if it was actually hidden before
                                        if item.properties.get("location", "hidden") == "hidden":
                                            # Move item from hidden to visible
                                            item.properties["location"] = "visible"
                                            newly_found.append(item_id)
                                            print(f"[Engine] Item '{item_id}' is now visible after search.")
                                            
                                # Update result message if some items were already found
                                if not newly_found and already_found:
                                    names = [self.environment.items[item_id].name for item_id in already_found]
                                    result["observation"] = f"You search the {obj.name} again, but find nothing new beyond what you already discovered: {', '.join(names)}"
                    else:
                        # Default search process for non-location objects
                        result = self.environment.process_interaction(
                            agent_id=agent.name,
                            action="search",
                            target_id=obj_id
                        )
                        
                        # Handle search results
                        if result.get("success", False) and "found_items" in result:
                            for item_id in result["found_items"]:
                                if item_id in self.environment.items:
                                    # Move item from hidden to visible
                                    self.environment.items[item_id].properties["location"] = "visible"
                                    print(f"[Engine] Item '{item_id}' is now visible after search.")
                                    
                    interaction_found = True
                    break

        # Handle speaking to other agents
        elif any(word in action_lower for word in ["speak", "talk", "say", "tell"]):
            for other_agent in self.agents:
                if other_agent != agent and other_agent.name.lower() in action_lower:
                    # Extract topic or message from "speak to [agent] about [topic]"
                    about_index = action_lower.find("about")
                    if about_index != -1:
                        topic = action[about_index + 5:].strip()
                        message = f"Hello {other_agent.name}, I'd like to discuss {topic}"
                    else:
                        message = f"Hello {other_agent.name}, let's talk."
                     
                    # Notify the other agent
                    if hasattr(other_agent, 'observe'):
                        other_agent.observe(f"{agent.name} says to you: {message}")
                    return f"{agent.name} says to {other_agent.name}: {message}"
                     
                    interaction_found = True
                    break
        
        # Handle take actions with improved parsing
        elif any(word in action_lower for word in ["take", "pick up", "grab", "get"]):
            for item_id in self.environment.items:
                if item_id.lower() in action_lower or self.environment.items[item_id].name.lower() in action_lower:
                    # Check if item is in a discoverable location
                    item = self.environment.items[item_id]
                    if item.properties.get("location", "visible") == "visible":
                        result = self.environment.process_interaction(
                            agent_id=agent.name,
                            action="pick_up",
                            target_id=item_id
                        )
                        
                        # If pickup successful, update agent's inventory
                        if result.get("success", False):
                            # Update the item's location to the agent's inventory
                            item.properties["location"] = f"inventory_{agent.name}"
                            
                            # Make sure the agent has an inventory
                            if "inventory" not in self.scenario_state:
                                self.scenario_state["inventory"] = {}
                            if agent.name not in self.scenario_state["inventory"]:
                                self.scenario_state["inventory"][agent.name] = []
                                
                            # Add item to inventory
                            self.scenario_state["inventory"][agent.name].append(item_id)
                    else:
                        result = {
                            "success": False,
                            "message": f"You don't see the {item.name} here.",
                            "observation": f"The {item.name} is not visible or accessible."
                        }
                        
                    interaction_found = True
                    break
        
        # Check for read actions
        elif "read" in action_lower:
            for item_id in self.environment.items:
                if item_id.lower() in action_lower or self.environment.items[item_id].name.lower() in action_lower:
                    item = self.environment.items[item_id]
                    
                    if item.properties.get("readable", False):
                        content = item.properties.get("content", "There's nothing written here.")
                        result = {
                            "success": True,
                            "message": f"You read the {item.name}.",
                            "observation": f"The {item.name} says: '{content}'"
                        }
                    else:
                        result = {
                            "success": False,
                            "message": f"The {item.name} is not readable.",
                            "observation": f"There's nothing to read on the {item.name}."
                        }
                    
                    interaction_found = True
                    break
        
        # Notify agent of the result through observation
        result_message = result.get("observation", result.get("message", "You performed an action."))
        if hasattr(agent, 'observe'):
            agent.observe(f"Action result: {result_message}")
        
        # If no specific interaction was found, provide general environmental context
        if not interaction_found and action.strip():
            if self.environment.active_location and self.environment.active_location in self.environment.locations:
                loc = self.environment.locations[self.environment.active_location]
                result = {
                    "success": True,
                    "message": f"You explore the {loc.name}.",
                    "observation": f"You are in the {loc.name}. {loc.description}"
                }
        
        # If no valid action was parsed, provide feedback
        if not interaction_found:
            return f"{agent.name} attempted an unclear action. Please choose from: examine room, examine [object], search [object], take [item], read [item], speak to [agent], or wait and observe."
        
        return result_message

    def _get_all_object_ids(self):
        """Get all object IDs in the environment for action matching."""
        object_ids = []
        
        # Add all item IDs
        object_ids.extend(self.environment.items.keys())
        
        # Add all location IDs
        object_ids.extend(self.environment.locations.keys())
        
        # Add all object IDs in locations
        for location in self.environment.locations.values():
            object_ids.extend(location.objects.keys())
            
        return object_ids

    def _get_object_name(self, obj_id):
        """Get the display name of an object by its ID."""
        # Check items first
        if obj_id in self.environment.items:
            return self.environment.items[obj_id].name
        
        # Check locations
        if obj_id in self.environment.locations:
            return self.environment.locations[obj_id].name
        
        # Check objects in locations
        for location in self.environment.locations.values():
            if obj_id in location.objects:
                return location.objects[obj_id].name
        
        return obj_id