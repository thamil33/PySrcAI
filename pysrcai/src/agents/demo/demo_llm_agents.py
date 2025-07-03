"""Enhanced demonstration of PySrcAI with LLM Integration.

This script demonstrates the LLM-powered agent hierarchy:
- Actor agents using ActorLLMComponent for intelligent participation
- Archon agents using ArchonLLMComponent for smart moderation
- Real AI-powered decision making instead of placeholder responses

Run this script to see intelligent agents in action.
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

from pysrcai.src.language_model_client import (
    LMStudioLanguageModel,
    OpenRouterLanguageModel,
    NoLanguageModel,
)

DEMO_MODEL_TYPE="openrouter"

def create_language_model(model_type: str = "openrouter"):
    """Create a language model based on configuration.
    
    Args:
        model_type: Type of model to create ("lmstudio", "openrouter", or "mock")
        
    Returns:
        A language model instance.
    """
    if model_type == "lmstudio":
        return LMStudioLanguageModel(
            model_name="local-model",
            base_url="http://localhost:1234/v1",
            verbose_logging=True
        )
    elif model_type == "openrouter":
        return OpenRouterLanguageModel(
            OPENROUTER_API_KEY=os.getenv("OPENROUTER_API_KEY"),
            model_name="mistralai/mistral-small-3.1-24b-instruct:free",
            verbose_logging=True
        )
    else:
        # Use no-op model for testing without actual LLM calls
        return NoLanguageModel()


def demonstrate_llm_actor(language_model):
    """Demonstrate LLM-powered Actor functionality."""
    print("=== LLM-POWERED ACTOR DEMONSTRATION ===")
    
    # Create an Actor with LLM integration
    actor = ActorWithLogging(
        agent_name="Alice",
        goals=["Argue for renewable energy", "Win the debate with compelling evidence"],
        personality_traits={
            "assertiveness": 0.8,
            "knowledge_level": 0.9,
            "cooperation": 0.6,
            "environmental_passion": 0.95
        },
        language_model=language_model
    )
    
    print(f"Created LLM-powered Actor: {actor.name}")
    print(f"Role: {actor.role}")
    print(f"Goals: {actor.goals}")
    print(f"Personality: {actor.personality_traits}")
    
    # Test observation
    print("\n--- Testing LLM Observation Processing ---")
    observation = (
        "The debate topic is: 'Should governments invest more in renewable energy?' "
        "Your opponent has just argued that renewable energy is too expensive and unreliable. "
        "The moderator has given you 2 minutes to respond."
    )
    actor.observe(observation)
    print(f"Actor observed: {observation}")
    
    # Test intelligent speech generation
    print("\n--- Testing LLM Speech Generation ---")
    speech_spec = ActionSpec(
        call_to_action=(
            "Respond to your opponent's argument about renewable energy being "
            "expensive and unreliable. Make a compelling counter-argument."
        ),
        output_type=OutputType.SPEECH,
        tag="debate_response"
    )
    speech = actor.act(speech_spec)
    print(f"Alice's LLM-generated response: {speech}")
    
    # Test strategic decision making
    print("\n--- Testing LLM Strategic Decision ---")
    strategy_spec = choice_action_spec(
        call_to_action=(
            "What strategy should you use for your next argument? "
            "Consider your opponent's weaknesses and the audience."
        ),
        options=[
            "Present economic data showing renewable cost savings",
            "Share examples of successful renewable energy projects", 
            "Challenge opponent's reliability claims with evidence",
            "Appeal to long-term environmental benefits"
        ],
        tag="strategy_choice"
    )
    strategy = actor.act(strategy_spec)
    print(f"Alice's strategic choice: {strategy}")
    
    return actor


def demonstrate_llm_archon(language_model):
    """Demonstrate LLM-powered Archon functionality."""
    print("\n=== LLM-POWERED ARCHON DEMONSTRATION ===")
    
    # Create an Archon with LLM integration
    archon = ArchonWithLogging(
        agent_name="DebateModerator",
        moderation_rules=[
            "Enforce 2-minute speaking time limits",
            "Maintain civil discourse - no personal attacks",
            "Ensure balanced participation between debaters",
            "Keep discussion focused on the topic",
            "Provide fair evaluation based on argument quality"
        ],
        authority_level="high",
        managed_entities=["Alice", "Bob"],
        language_model=language_model
    )
    
    print(f"Created LLM-powered Archon: {archon.name}")
    print(f"Role: {archon.role}")
    print(f"Authority Level: {archon.authority_level}")
    print(f"Managed Entities: {archon.managed_entities}")
    
    # Test intelligent moderation
    print("\n--- Testing LLM Moderation Decision ---")
    moderation_observation = (
        "Alice has been speaking for 2 minutes and 30 seconds about renewable energy costs. "
        "She's making good points but is exceeding the time limit. Bob is waiting to respond "
        "and looks frustrated. The audience seems engaged but restless."
    )
    archon.observe(moderation_observation)
    print(f"Archon observed: {moderation_observation}")
    
    moderation_spec = ActionSpec(
        call_to_action=(
            "Alice has exceeded her 2-minute time limit. How should you moderate this situation "
            "to maintain fairness while allowing valuable discussion?"
        ),
        output_type=OutputType.MODERATE,
        tag="time_limit_enforcement"
    )
    moderation_action = archon.act(moderation_spec)
    print(f"Archon's LLM-generated moderation: {moderation_action}")
    
    # Test intelligent evaluation
    print("\n--- Testing LLM Performance Evaluation ---")
    evaluation_spec = ActionSpec(
        call_to_action=(
            "Evaluate Alice's debate performance so far. Consider argument quality, "
            "evidence presented, persuasiveness, and adherence to debate rules."
        ),
        output_type=OutputType.EVALUATE,
        tag="performance_evaluation"
    )
    evaluation = archon.act(evaluation_spec)
    print(f"Archon's LLM-generated evaluation: {evaluation}")
    
    # Test session management decision
    print("\n--- Testing LLM Session Management ---")
    session_spec = choice_action_spec(
        call_to_action=(
            "The debate has been running for 15 minutes. Both participants have made "
            "their main arguments. What should you do next?"
        ),
        options=[
            "Move to closing statements phase",
            "Allow one more round of rebuttals", 
            "Open floor for audience questions",
            "Conclude the debate and announce results"
        ],
        tag="session_management"
    )
    session_decision = archon.act(session_spec)
    print(f"Archon's session management decision: {session_decision}")
    
    return archon


def demonstrate_llm_interaction(actor, archon):
    """Demonstrate intelligent interaction between LLM-powered agents."""
    print("\n=== LLM-POWERED AGENT INTERACTION ===")
    
    # Archon starts an intelligent session
    archon.set_session_state("active")
    print(f"Archon {archon.name} started the debate session")
    
    # Scenario: Mid-debate intervention
    scenario_observation = (
        f"{actor.name} has made a passionate argument about renewable energy, "
        f"but is starting to get heated and may be veering into personal territory. "
        f"The opponent looks defensive. The audience is engaged but tension is rising."
    )
    
    # Both agents observe the same situation
    archon.observe(scenario_observation)
    actor.observe(f"The moderator is watching you carefully. The atmosphere is tense.")
    
    print(f"Scenario: {scenario_observation}")
    
    # Archon makes an intelligent moderation decision
    intervention_spec = ActionSpec(
        call_to_action=(
            "The debate is getting heated. Make a moderation intervention that "
            "maintains engagement while ensuring civil discourse."
        ),
        output_type=OutputType.MODERATE,
        tag="intervention"
    )
    intervention = archon.act(intervention_spec)
    print(f"Archon intervention: {intervention}")
    
    # Actor responds intelligently to moderation
    actor.observe(f"The moderator intervened: {intervention}")
    response_spec = ActionSpec(
        call_to_action=(
            "The moderator has just intervened. How do you adjust your approach "
            "while still pursuing your debate goals?"
        ),
        output_type=OutputType.SPEECH,
        tag="moderated_response"
    )
    actor_response = actor.act(response_spec)
    print(f"Actor's adjusted response: {actor_response}")
    
    # Archon evaluates the response
    evaluation_spec = ActionSpec(
        call_to_action=(
            f"Evaluate how well {actor.name} responded to your moderation. "
            f"Did they adjust appropriately while maintaining their position?"
        ),
        output_type=OutputType.EVALUATE,
        tag="response_evaluation"
    )
    response_evaluation = archon.act(evaluation_spec)
    print(f"Archon's evaluation of response: {response_evaluation}")
    
    print(f"\nIntelligent interaction complete. Session state: {archon.session_state}")


def main():
    """Main demonstration function."""
    print("PySrcAI LLM-Powered Agent Hierarchy Demonstration")
    print("=================================================")
    
    # Determine which language model to use
    model_type = os.getenv("DEMO_MODEL_TYPE", "mock")  # mock, lmstudio, or openrouter
    print(f"Using {model_type} language model")
    
    if model_type != "mock":
        print(f"Note: Using real LLM - responses will be more intelligent but may take longer")
    else:
        print(f"Note: Using mock LLM - responses will be placeholders for demonstration")
    
    # Create language model
    language_model = create_language_model(model_type)
    
    # Demonstrate LLM-powered agents
    actor = demonstrate_llm_actor(language_model)
    archon = demonstrate_llm_archon(language_model)
    
    # Demonstrate intelligent interaction
    demonstrate_llm_interaction(actor, archon)
    
    print("\n=== LLM INTEGRATION SUMMARY ===")
    print("✅ Language model client successfully integrated")
    print("✅ ActorLLMComponent provides intelligent participant behavior")
    print("✅ ArchonLLMComponent provides intelligent moderation") 
    print("✅ Agents can use LLMs for decision-making and responses")
    print("✅ Prompt engineering optimized for Actor vs Archon roles")
    print("✅ Choice and free-form action support")
    print("✅ Context integration from components")
    print("✅ Backwards compatibility with non-LLM components")
    
    if model_type == "mock":
        print("\n💡 To see real AI behavior:")
        print("   - Set DEMO_MODEL_TYPE=lmstudio (with LM Studio running)")
        print("   - Or set DEMO_MODEL_TYPE=openrouter (with OPENROUTER_API_KEY)")
    
    print("\nPySrcAI LLM Integration is ready for intelligent simulations!")


if __name__ == "__main__":
    main()
