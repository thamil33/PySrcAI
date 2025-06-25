"""Example demonstrating OpenRouter LLM integration with LangGraph."""

import os
from typing import TypedDict, Annotated
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

from pyscrai.agentica.config.config import ModelConfig, AgentConfig
from pyscrai.agentica.adapters.llm import OpenRouterLLM
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class State(TypedDict):
    """State for the LangGraph example."""
    messages: Annotated[list, add_messages]
    iterations: int


def llm_node(state: State) -> dict:
    """Node that calls the OpenRouter LLM."""
    # Get the last human message
    last_message = state["messages"][-1]
    
    # Create OpenRouter LLM with robust configuration
    llm = OpenRouterLLM(
        model="mistralai/mistral-small-24b-instruct-2501:free",
        temperature=0.7,
        max_tokens=500,
        app_name="pyscrai_example",
        site_url="https://github.com/tyler-richardson/pyscrai_workstation",
        # Simplified provider routing for free models (complex routing causes 404)
        provider={
            "sort": "price"  # Cost optimization works with free models
        }
    )
    
    # Generate response
    response = llm.invoke(last_message.content)
    
    return {
        "messages": [AIMessage(content=response)],
        "iterations": state["iterations"] + 1
    }


def should_continue(state: State) -> str:
    """Decide whether to continue the conversation."""
    if state["iterations"] >= 3:
        return END
    return "llm"


def create_agent_graph():
    """Create a LangGraph agent using OpenRouter."""
    # Create the graph
    workflow = StateGraph(State)
    
    # Add nodes
    workflow.add_node("llm", llm_node)
    
    # Set entry point
    workflow.set_entry_point("llm")
    
    # Add conditional edges
    workflow.add_conditional_edges(
        "llm",
        should_continue,
        {"llm": "llm", END: END}
    )
    
    # Compile the graph
    return workflow.compile()


def main():
    """Main example function."""
    # Ensure API key is set
    if not os.getenv("OPENROUTER_API_KEY"):
        print("Please set OPENROUTER_API_KEY environment variable")
        return
    
    # Create the agent
    agent = create_agent_graph()
    
    # Example conversation
    initial_state = {
        "messages": [HumanMessage(content="What are the benefits of using OpenRouter for AI applications?")],
        "iterations": 0
    }
    
    print("🤖 OpenRouter LangGraph Agent Example")
    print("=" * 50)
    
    try:
        # Run the agent
        result = agent.invoke(initial_state)
        
        # Display conversation
        for message in result["messages"]:
            if isinstance(message, HumanMessage):
                print(f"👤 Human: {message.content}")
            elif isinstance(message, AIMessage):
                print(f"🤖 AI: {message.content}")
                print()
        
        print(f"✅ Completed after {result['iterations']} iterations")
        
    except Exception as e:
        print(f"❌ Error: {e}")


def streaming_example():
    """Example demonstrating streaming responses."""
    if not os.getenv("OPENROUTER_API_KEY"):
        print("Please set OPENROUTER_API_KEY environment variable")
        return
    print("🚀 Streaming Example")
    print("=" * 30)
    llm = OpenRouterLLM(
        model="mistralai/mistral-small-24b-instruct-2501:free",
        temperature=0.7,
        max_tokens=200
    )
    
    try:
        print("💭 Prompt: Tell me about Python programming")
        print("🤖 Response: ", end="", flush=True)
        
        for chunk in llm.stream("Tell me about Python programming in 3 sentences"):
            # Handle both GenerationChunk and string types
            content = chunk.text if hasattr(chunk, 'text') else str(chunk)
            print(content, end="", flush=True)
        
        print("\n✅ Streaming completed")
        
    except Exception as e:
        print(f"❌ Streaming error: {e}")


def provider_routing_example():
    """Example demonstrating advanced provider routing."""
    if not os.getenv("OPENROUTER_API_KEY"):
        print("Please set OPENROUTER_API_KEY environment variable")
        return
    
    print("🎯 Provider Routing Example")
    print("=" * 35)
    
    # Example 1: Price-optimized routing
    llm_cheap = OpenRouterLLM(
        model="mistralai/mistral-small-24b-instruct-2501:free",
        provider={
            "sort": "price",
        }
    )
    
    # Example 2: Performance-optimized routing
    llm_fast = OpenRouterLLM(
        model="mistralai/mistral-small-3.1-24b-instruct",
        provider={
            "sort": "throughput",
            "allow_fallbacks": True
        }
    )
    # Example 3: Specific provider with provider fallbacks
    llm_specific = OpenRouterLLM(
        model="mistralai/mistral-small-3.1-24b-instruct",
        provider={
            "order": ["Mistral", "OpenAI"],
            "allow_fallbacks": True
        }
    )
    
    examples = [
        ("💰 Price-optimized", llm_cheap),
        ("⚡ Performance-optimized", llm_fast),
        ("🎯 Specific provider", llm_specific)
    ]
    
    for name, llm in examples:
        try:
            print(f"\n{name}:")
            response = llm.invoke("What is 2+2?")
            print(f"Response: {response[:100]}...")
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    # Run examples
    main()
    print("\n" + "=" * 60 + "\n")
    streaming_example()
    print("\n" + "=" * 60 + "\n")
    provider_routing_example()
