"""Demonstration of the PySrcAI Agent Hierarchy.

This script demonstrates the basic functionality of the new Agent hierarchy:
- Agent as the base abstraction (equivalent to Concordia's Entity)
- Actor as specialized participant agents
- Archon as specialized moderator agents

Run this script to see the agent hierarchy in action.
"""

import sys
import os
import dotenv

dotenv.load_dotenv()


# Add the project root to the path so we can import from pysrcai
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from pysrcai.src.agents import (
    Actor, 
    ActorWithLogging,
    Archon, 
    ArchonWithLogging,
    ActionSpec,
    OutputType,
    free_action_spec,
    choice_action_spec,
)


def demonstrate_actor():
    """Demonstrate Actor functionality."""
    print("=== ACTOR DEMONSTRATION ===")
    
    # Create an Actor for a debate scenario
    actor = ActorWithLogging(
        agent_name="Alice",
        goals=["Argue for renewable energy", "Win the debate"],
        personality_traits={
            "assertiveness": 0.8,
            "knowledge_level": 0.9,
            "cooperation": 0.6
        }
    )
    
    print(f"Created Actor: {actor.name}")
    print(f"Role: {actor.role}")
    print(f"Goals: {actor.goals}")
    print(f"Personality: {actor.personality_traits}")
    
    # Test observation
    print("\n--- Testing Observation ---")
    observation = "The debate topic is: 'Should governments invest more in renewable energy?'"
    actor.observe(observation)
    print(f"Actor observed: {observation}")
    
    # Test free action
    print("\n--- Testing Free Action ---")
    action_spec = free_action_spec(
        call_to_action="What is your opening statement in this debate?",
        tag="opening_statement"
    )
    action = actor.act(action_spec)
    print(f"Actor action: {action}")
    
    # Test choice action
    print("\n--- Testing Choice Action ---")
    choice_spec = choice_action_spec(
        call_to_action="How would you like to proceed?",
        options=["Present evidence", "Ask a question", "Make a counterargument"],
        tag="strategy_choice"
    )
    choice = actor.act(choice_spec)
    print(f"Actor choice: {choice}")
    
    # Show logging information
    print("\n--- Actor Log Information ---")
    log_info = actor.get_last_log()
    print(f"Recent actions: {len(log_info['recent_actions'])}")
    print(f"Recent observations: {len(log_info['recent_observations'])}")
    print(f"Current phase: {log_info['current_phase']}")
    
    return actor


def demonstrate_archon():
    """Demonstrate Archon functionality."""
    print("\n=== ARCHON DEMONSTRATION ===")
    
    # Create an Archon for moderating the debate
    archon = ArchonWithLogging(
        agent_name="DebateModerator",
        moderation_rules=[
            "Enforce 2-minute speaking time limits",
            "Maintain civil discourse",
            "Ensure balanced participation"
        ],
        authority_level="high",
        managed_entities=["Alice", "Bob"]
    )
    
    print(f"Created Archon: {archon.name}")
    print(f"Role: {archon.role}")
    print(f"Authority Level: {archon.authority_level}")
    print(f"Moderation Rules: {archon.moderation_rules}")
    print(f"Managed Entities: {archon.managed_entities}")
    
    # Test observation
    print("\n--- Testing Observation ---")
    observation = "Alice has been speaking for 2 minutes and 15 seconds."
    archon.observe(observation)
    print(f"Archon observed: {observation}")
    
    # Test moderation action
    print("\n--- Testing Moderation Action ---")
    action_spec = ActionSpec(
        call_to_action="What moderation action should you take?",
        output_type=OutputType.MODERATE,
        tag="moderation"
    )
    action = archon.act(action_spec)
    print(f"Archon moderation: {action}")
    
    # Test interaction moderation
    print("\n--- Testing Interaction Moderation ---")
    moderation_result = archon.moderate_interaction(
        participants=["Alice", "Bob"],
        interaction_type="debate"
    )
    print(f"Interaction moderation: {moderation_result}")
    
    # Test performance evaluation
    print("\n--- Testing Performance Evaluation ---")
    evaluation = archon.evaluate_performance(
        entity_name="Alice",
        criteria={"argument_quality": 0.9, "time_management": 0.7}
    )
    print(f"Performance evaluation: {evaluation}")
    
    # Show logging information
    print("\n--- Archon Log Information ---")
    log_info = archon.get_last_log()
    print(f"Recent moderations: {len(log_info['recent_moderations'])}")
    print(f"Recent observations: {len(log_info['recent_observations'])}")
    print(f"Recent evaluations: {len(log_info['recent_evaluations'])}")
    print(f"Session state: {log_info['session_state']}")
    
    return archon


def demonstrate_interaction(actor, archon):
    """Demonstrate interaction between Actor and Archon."""
    print("\n=== ACTOR-ARCHON INTERACTION ===")
    
    # Archon starts the session
    archon.set_session_state("active")
    print(f"Archon {archon.name} started the session")
    
    # Archon observes the actor's behavior
    observation = f"{actor.name} is preparing to make an argument"
    archon.observe(observation)
    print(f"Archon observed: {observation}")
    
    # Actor makes a statement
    statement_spec = free_action_spec(
        call_to_action="Make your main argument",
        tag="main_argument"
    )
    statement = actor.act(statement_spec)
    print(f"Actor statement: {statement}")
    
    # Archon moderates the statement
    moderation = archon.moderate_interaction([actor.name], "statement")
    print(f"Archon moderation: {moderation}")
    
    # Archon evaluates the actor's performance
    evaluation = archon.evaluate_performance(
        actor.name,
        {"clarity": 0.8, "evidence": 0.7, "persuasiveness": 0.9}
    )
    print(f"Performance evaluation: {evaluation}")
    
    print(f"\nInteraction complete. Session state: {archon.session_state}")


def main():
    """Main demonstration function."""
    print("PySrcAI Agent Hierarchy Demonstration")
    print("=====================================")
    
    # Demonstrate individual agent types
    actor = demonstrate_actor()
    archon = demonstrate_archon()
    
    # Demonstrate interaction between agent types
    demonstrate_interaction(actor, archon)
    
    print("\n=== SUMMARY ===")
    print("✅ Agent base class provides common functionality")
    print("✅ Actor specializes for simulation participation")
    print("✅ Archon specializes for simulation moderation")
    print("✅ Both can observe, act, and maintain state")
    print("✅ Clear separation of participant vs. moderator roles")
    print("✅ Component system ready for extension")
    print("✅ Logging capabilities for debugging and monitoring")
    
    print("\nPySrcAI Agent Hierarchy is ready for development!")


if __name__ == "__main__":
    main()
