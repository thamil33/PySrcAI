"""
Example: Creating a Custom RAG Agent
=====================================

This example shows how to create a custom RAG agent using the builder pattern.
"""

from pyscrai.rag_agent import RAGAgentBuilder, quick_agent

# Example 1: Using the builder pattern
def create_documentation_agent():
    """Create a specialized documentation assistant."""
    
    agent = (RAGAgentBuilder()
             .with_name("DocumentationAssistant")
             .with_system_prompt("""
You are a helpful documentation assistant. You excel at:
- Explaining complex technical concepts clearly
- Providing step-by-step guides
- Finding relevant examples and code snippets
- Organizing information logically

When answering questions, always:
1. Start with a brief, clear answer
2. Provide detailed explanations with examples
3. Include relevant code snippets when helpful
4. Suggest related topics or next steps
""")
             .with_data_sources([
                 "docs/",
                 "README.md",
                 "examples/",
                 "tutorials/"
             ])
             .with_embedding_provider("huggingface_api")
             .with_llm_model("mistralai/mistral-small-3.1-24b-instruct:free")
             .with_vector_db_collection("documentation_kb")
             .with_rag_settings(top_k=7, similarity_threshold=0.6)
             .build())
    
    return agent


# Example 2: Using the quick_agent helper
def create_code_review_agent():
    """Create a code review assistant using the quick helper."""
    
    system_prompt = """
You are an expert code reviewer with deep knowledge of software development best practices.

Your responsibilities include:
- Identifying potential bugs and issues
- Suggesting performance improvements
- Recommending better architectural patterns
- Ensuring code follows best practices
- Providing constructive feedback

Always provide:
1. Specific, actionable feedback
2. Examples of improved code when suggesting changes
3. Explanations for why changes are recommended
4. Positive reinforcement for good practices found
"""
    
    agent = quick_agent(
        name="CodeReviewAssistant",
        system_prompt=system_prompt,
        data_sources=[
            "src/",
            "docs/coding_standards.md",
            "docs/architecture_guide.md"
        ]
    )
    
    return agent


# Example 3: Customizing with specific configuration
def create_research_agent():
    """Create a research assistant with custom configuration."""
    
    # Load custom config file
    agent = (RAGAgentBuilder()
             .with_config_file("research_config.yaml")
             .with_name("ResearchAssistant")
             .with_system_prompt("""
You are a research assistant specializing in academic and technical research.

Your capabilities include:
- Synthesizing information from multiple sources
- Identifying key insights and patterns
- Providing comprehensive literature reviews
- Suggesting research directions and methodologies
- Creating structured summaries and reports

Approach each query by:
1. Analyzing the research question thoroughly
2. Gathering relevant information from multiple sources
3. Synthesizing findings into coherent insights
4. Providing citations and references
5. Suggesting follow-up research questions
""")
             .with_data_sources([
                 "research_papers/",
                 "literature_reviews/",
                 "datasets/",
                 "methodologies/"
             ])
             .build())
    
    return agent


# Example usage
if __name__ == "__main__":
    # Create and use a custom agent
    print("Creating documentation agent...")
    doc_agent = create_documentation_agent()
    
    # Ingest documents
    doc_agent.ingest_documents([
        "docs/getting_started.md",
        "docs/api_reference.md",
        "README.md"
    ])
    
    # Query the agent
    response = doc_agent.query("How do I get started with this framework?")
    print(f"Agent response: {response}")
    
    # Get collection info
    info = doc_agent.get_collection_info()
    print(f"Collection info: {info}")
