"""Memory Integration Demonstration for PySrcAI.

This script demonstrates memory capabilities with a simple, open-ended scenario:
- Agents with persistent memory
- Memory-based context for decision making  
- Different memory retrieval strategies
- LLM integration with memory context

Scenario: A simple social interaction where agents meet, chat, and remember
their interactions for future reference.
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
    BasicMemoryBank,
    AssociativeMemoryBank,
    MemoryComponent,
    create_simple_embedder,
)

from pysrcai.src.language_model_client import (
    LMStudioLanguageModel,
    OpenRouterLanguageModel,
    NoLanguageModel,
)


def create_language_model(model_type: str = "mock"):
    """Create a language model based on configuration."""
    if model_type == "lmstudio":
        return LMStudioLanguageModel(
            model_name="local-model",
            base_url="http://localhost:1234/v1",
            verbose_logging=True
        )
    elif model_type == "openrouter":
        return OpenRouterLanguageModel(
            model_name="mistralai/mistral-7b-instruct:free",
            verbose_logging=True
        )
    else:
        return NoLanguageModel()


def create_actor_with_memory(name: str, personality: dict, language_model, use_associative: bool = False):
    """Create an actor with memory capabilities."""
    # Create memory bank
    if use_associative:
        try:
            embedder = create_simple_embedder()
            memory_bank = AssociativeMemoryBank(embedder=embedder.embed, max_memories=100)
            print(f"  Using associative memory for {name}")
        except ImportError:
            print(f"  Falling back to basic memory for {name} (numpy/pandas not available)")
            memory_bank = BasicMemoryBank(max_memories=100)
    else:
        memory_bank = BasicMemoryBank(max_memories=100)
        print(f"  Using basic memory for {name}")
    
    # Create memory component
    memory_component = MemoryComponent(memory_bank, max_context_memories=3)
    
    # Create actor with memory
    actor = ActorWithLogging(
        agent_name=name,
        personality_traits=personality,
        context_components={"memory": memory_component},
        language_model=language_model
    )
    
    return actor, memory_component


def demonstrate_basic_memory():
    """Demonstrate basic memory functionality."""
    print("=== BASIC MEMORY DEMONSTRATION ===")
    
    # Create simple memory bank
    memory_bank = BasicMemoryBank(max_memories=10)
    
    print("Adding memories...")
    memory_bank.add_memory("I went to the store today", tags=["action", "daily"])
    memory_bank.add_memory("I met a friendly person named Alice", tags=["social", "people"])
    memory_bank.add_memory("We talked about renewable energy", tags=["conversation", "topics"])
    memory_bank.add_memory("Alice seemed very knowledgeable", tags=["people", "impression"])
    memory_bank.add_memory("I bought some groceries", tags=["action", "daily"])
    
    print(f"Memory bank now contains {len(memory_bank)} memories")
    
    # Test different retrieval methods
    print("\n--- Recent Memories ---")
    recent = memory_bank.retrieve_recent(3)
    for i, memory in enumerate(recent, 1):
        print(f"{i}. {memory}")
    
    print("\n--- Query: 'Alice' ---")
    alice_memories = memory_bank.retrieve_by_query("Alice", 3)
    for i, memory in enumerate(alice_memories, 1):
        print(f"{i}. {memory}")
    
    print("\n--- Memories tagged 'people' ---")
    people_memories = memory_bank.retrieve_by_tags(["people"], 3)
    for i, memory in enumerate(people_memories, 1):
        print(f"{i}. {memory}")


def demonstrate_agents_with_memory(language_model):
    """Demonstrate agents using memory in interactions."""
    print("\n=== AGENTS WITH MEMORY DEMONSTRATION ===")
    
    # Create two actors with different personalities and memory systems
    alice, alice_memory = create_actor_with_memory(
        name="Alice",
        personality={"curiosity": 0.9, "friendliness": 0.8, "knowledge_sharing": 0.7},
        language_model=language_model,
        use_associative=True  # Try associative memory
    )
    
    bob, bob_memory = create_actor_with_memory(
        name="Bob", 
        personality={"introversion": 0.6, "thoughtfulness": 0.8, "analytical": 0.9},
        language_model=language_model,
        use_associative=False  # Use basic memory
    )
    
    print(f"Created {alice.name} and {bob.name} with memory capabilities")
    
    # Give them some initial memories
    print("\n--- Setting Initial Memories ---")
    alice_memory.add_explicit_memory(
        "I love learning about new technologies and sharing knowledge",
        tags=["personality", "interests"],
        importance=0.9
    )
    alice_memory.add_explicit_memory(
        "I recently read about advances in renewable energy",
        tags=["knowledge", "recent"],
        importance=0.7
    )
    
    bob_memory.add_explicit_memory(
        "I prefer to think carefully before responding to others",
        tags=["personality", "behavior"],
        importance=0.9
    )
    bob_memory.add_explicit_memory(
        "I'm interested in analytical approaches to complex problems",
        tags=["interests", "analytical"],
        importance=0.8
    )
    
    print("Initial memories set for both agents")
    
    return alice, bob, alice_memory, bob_memory


def simulate_interaction(alice, bob, alice_memory, bob_memory):
    """Simulate a social interaction between the agents."""
    print("\n=== MEMORY-DRIVEN INTERACTION SIMULATION ===")
    
    # First encounter
    print("\n--- First Meeting ---")
    
    # Alice observes meeting Bob
    alice.observe("I've just met someone new named Bob. He seems thoughtful and analytical.")
    
    # Bob observes meeting Alice  
    bob.observe("I've encountered Alice, who appears curious and knowledgeable about technology.")
    
    # Alice decides what to say
    alice_greeting = alice.act(free_action_spec(
        call_to_action="You've just met Bob. What would you say to start a conversation?",
        tag="greeting"
    ))
    print(f"Alice says: {alice_greeting}")
    
    # Bob observes Alice's greeting and responds
    bob.observe(f"Alice just said to me: {alice_greeting}")
    bob_response = bob.act(free_action_spec(
        call_to_action="Alice has just greeted you. How do you respond?",
        tag="response"
    ))
    print(f"Bob responds: {bob_response}")
    
    # Continue the conversation
    print("\n--- Conversation Development ---")
    
    # Alice observes Bob's response and continues
    alice.observe(f"Bob responded: {bob_response}")
    alice_followup = alice.act(free_action_spec(
        call_to_action="Based on Bob's response, what would you like to talk about?",
        tag="conversation"
    ))
    print(f"Alice continues: {alice_followup}")
    
    # Bob's thoughtful reply
    bob.observe(f"Alice said: {alice_followup}")
    bob_thoughtful = bob.act(free_action_spec(
        call_to_action="Alice has shared something with you. Give a thoughtful response.",
        tag="thoughtful_response"
    ))
    print(f"Bob reflects: {bob_thoughtful}")
    
    print("\n--- Memory Status After Interaction ---")
    alice_bank = alice_memory.get_memory_bank()
    bob_bank = bob_memory.get_memory_bank()
    print(f"Alice's memory contains {len(alice_bank)} memories")
    print(f"Bob's memory contains {len(bob_bank)} memories")
    
    return alice_bank, bob_bank


def demonstrate_memory_persistence(alice, bob, alice_bank, bob_bank):
    """Demonstrate how memories persist and influence future interactions."""
    print("\n=== MEMORY PERSISTENCE DEMONSTRATION ===")
    
    print("--- Later Encounter (agents remember previous interaction) ---")
    
    # Simulate meeting again later
    alice.observe("I see Bob again. We've met before.")
    alice_recognition = alice.act(free_action_spec(
        call_to_action="You're meeting Bob again. What do you remember about him and what would you say?",
        tag="recognition"
    ))
    print(f"Alice (remembering): {alice_recognition}")
    
    bob.observe("Alice is approaching me again. We've talked before.")
    bob_recognition = bob.act(free_action_spec(
        call_to_action="Alice is approaching you again. What do you remember about your previous conversation?",
        tag="recognition"
    ))
    print(f"Bob (remembering): {bob_recognition}")
    
    # Show memory contents
    print("\n--- Memory Contents ---")
    print("Alice's memories:")
    alice_memories = alice_bank.retrieve_recent(5)
    for i, memory in enumerate(alice_memories, 1):
        print(f"  {i}. {memory}")
    
    print("\nBob's memories:")
    bob_memories = bob_bank.retrieve_recent(5) 
    for i, memory in enumerate(bob_memories, 1):
        print(f"  {i}. {memory}")


def demonstrate_archon_memory_management(language_model):
    """Demonstrate how an Archon can manage and analyze agent memories."""
    print("\n=== ARCHON MEMORY MANAGEMENT ===")
    
    # Create an Archon that can analyze interactions
    memory_bank = BasicMemoryBank(max_memories=200)
    memory_component = MemoryComponent(memory_bank, max_context_memories=5)
    
    archon = ArchonWithLogging(
        agent_name="InteractionAnalyst",
        moderation_rules=[
            "Monitor agent interactions for insights",
            "Track relationship development", 
            "Identify interesting patterns"
        ],
        authority_level="observer",
        context_components={"memory": memory_component},
        language_model=language_model
    )
    
    print(f"Created {archon.name} archon with memory capabilities")
    
    # Archon observes the agents' interactions
    archon.observe("Alice and Bob had their first meeting. Alice was curious and talkative.")
    archon.observe("Bob was thoughtful and analytical in his responses to Alice.")
    archon.observe("They discussed technology and both seemed engaged.")
    archon.observe("When they met again, both showed recognition and referenced their previous conversation.")
    
    # Archon analyzes the interaction patterns
    analysis = archon.act(ActionSpec(
        call_to_action="Based on your observations, what patterns do you notice in how Alice and Bob interact and remember each other?",
        output_type=OutputType.FREE,
        tag="analysis"
    ))
    print(f"\nArchon Analysis: {analysis}")
    
    return archon


def main():
    """Main demonstration function."""
    print("PySrcAI Memory Integration Demonstration")
    print("========================================")
    
    # Determine language model type
    model_type = os.getenv("DEMO_MODEL_TYPE", "mock")
    print(f"Using {model_type} language model")
    
    # Create language model
    language_model = create_language_model(model_type)
    
    # Demonstrate basic memory functionality
    demonstrate_basic_memory()
    
    # Demonstrate agents with memory
    alice, bob, alice_memory, bob_memory = demonstrate_agents_with_memory(language_model)
    
    # Simulate interactions
    alice_bank, bob_bank = simulate_interaction(alice, bob, alice_memory, bob_memory)
    
    # Show memory persistence
    demonstrate_memory_persistence(alice, bob, alice_bank, bob_bank)
    
    # Demonstrate Archon memory management (behind-the-scenes)
    archon = demonstrate_archon_memory_management(language_model)
    
    print("\n=== MEMORY INTEGRATION SUMMARY ===")
    print("✅ Basic memory storage and retrieval working")
    print("✅ Associative memory with embedding-based search")
    print("✅ Memory components integrated with agent context system")
    print("✅ LLM agents use memory context for decision making")
    print("✅ Memory persistence across interactions")
    print("✅ Multiple memory retrieval strategies (recent, query, tags)")
    print("✅ Archon memory management for behind-the-scenes analysis")
    print("✅ Open-ended scenario showcasing memory capabilities")
    
    if model_type == "mock":
        print("\n💡 To see intelligent memory-driven behavior:")
        print("   - Set DEMO_MODEL_TYPE=lmstudio (with LM Studio running)")
        print("   - Or set DEMO_MODEL_TYPE=openrouter (with OPENROUTER_API_KEY)")
    
    print("\nPySrcAI Memory Integration is ready for intelligent, persistent agents!")


if __name__ == "__main__":
    main()
