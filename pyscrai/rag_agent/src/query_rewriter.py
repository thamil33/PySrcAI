"""Query rewriting module for Concordia Assistant (Phase 3)."""

class QueryRewriter:
    """Rewrites user queries for improved retrieval."""
    def __init__(self, llm_adapter=None):
        self.llm_adapter = llm_adapter

    def rewrite(self, query: str, context: str = None) -> str:
        """Rewrite the user query for clarity and context."""
        # Placeholder: In production, use LLM or prompt template
        if self.llm_adapter:
            prompt = f"Rewrite the following user query to be as clear and specific as possible for a retrieval system.\n\nQuery: {query}\n"
            if context:
                prompt += f"\nContext: {context}\n"
            prompt += "\nRewritten Query:"
            return self.llm_adapter.generate(prompt).strip()
        return query
