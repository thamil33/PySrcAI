"""Base RAG Agent class - abstract foundation for all RAG agents."""

import os
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from .config_loader import AgentConfig


class BaseRAGAgent(ABC):
    """
    Abstract base class for all RAG agents.

    Provides the common interface and basic functionality that all RAG agents
    must implement, while allowing for customization of specific behaviors.
    """

    def __init__(self, config: AgentConfig):
        """Initialize the base RAG agent with configuration."""
        self.config = config
        self.embedding_adapter = None
        self.vector_db = None
        self.chunker = None
        self.llm_adapter = None
        self._initialized = False

    @abstractmethod
    def get_system_prompt(self) -> str:
        """Return the system prompt specific to this agent type."""
        pass

    @abstractmethod
    def get_agent_name(self) -> str:
        """Return the name of this agent type."""
        pass

    @abstractmethod
    def get_default_data_sources(self) -> List[str]:
        """Return default data sources for this agent type."""
        pass

    def initialize(self):
        """Initialize all components of the RAG agent."""
        if self._initialized:
            return

        try:
            print(f"Initializing {self.get_agent_name()}...")

            # Initialize components using adapters
            self._setup_embedding_adapter()
            self._setup_vector_db()
            self._setup_chunker()
            self._setup_llm_adapter()

            self._initialized = True
            print(f"{self.get_agent_name()} initialized successfully!")

        except Exception as e:
            print(f"Error initializing {self.get_agent_name()}: {e}")
            raise

    def _setup_embedding_adapter(self):
        """Setup the embedding adapter based on configuration."""
        # Import the adapter from the package root
        from ..adapters.embedding_adapter import EmbeddingAdapter

        self.embedding_adapter = EmbeddingAdapter(self.config)

    def _setup_vector_db(self):
        """Setup the vector database based on configuration."""
        from ..adapters.vector_db_adapter import VectorDBAdapter

        self.vector_db = VectorDBAdapter(self.config)

    def _setup_chunker(self):
        """Setup the document chunker based on configuration."""
        from .chunking import DocumentChunker

        self.chunker = DocumentChunker(self.config)

    def _setup_llm_adapter(self):
        """Setup the language model adapter based on configuration."""
        from ..adapters.llm_adapter import LLMAdapter

        self.llm_adapter = LLMAdapter(self.config)

    def ingest_documents(self, file_paths: List[str], force_rebuild: bool = False):
        """
        Ingest documents into the vector database.

        Args:
            file_paths: List of file paths to ingest
            force_rebuild: Whether to rebuild the collection even if it exists
        """
        if not self._initialized:
            self.initialize()

        return self.vector_db.ingest_documents(
            file_paths,
            self.chunker,
            self.embedding_adapter,
            force_rebuild=force_rebuild,
        )

    def query(self, question: str, top_k: int = None) -> str:
        """
        Query the RAG agent with a question.

        Args:
            question: The question to ask
            top_k: Number of top results to retrieve (uses config default if None)

        Returns:
            The agent's response
        """
        if not self._initialized:
            self.initialize()

        # Retrieve relevant documents
        top_k = top_k or self.config.rag.top_k
        relevant_docs = self.vector_db.query(
            question, self.embedding_adapter, top_k=top_k
        )

        # Generate response using LLM
        return self._generate_response(question, relevant_docs)

    def _generate_response(
        self, question: str, relevant_docs: List[Dict[str, Any]]
    ) -> str:
        """Generate a response using the LLM and retrieved documents."""
        # Format the context from retrieved documents
        context = self._format_context(relevant_docs)

        # Create the full prompt
        prompt = self._build_prompt(question, context)

        # Generate response
        return self.llm_adapter.generate(prompt)

    def _format_context(self, relevant_docs: List[Dict[str, Any]]) -> str:
        """Format retrieved documents into context string."""
        if not relevant_docs:
            return "No relevant information found."

        context_parts = []
        for i, doc in enumerate(relevant_docs, 1):
            content = doc.get("content", "")
            metadata = doc.get("metadata", {})
            source = metadata.get("source", "Unknown")

            context_parts.append(f"[{i}] {content}\n(Source: {source})")

        return "\n\n".join(context_parts)

    def _build_prompt(self, question: str, context: str) -> str:
        """Build the complete prompt for the LLM."""
        system_prompt = self.get_system_prompt()

        return f"""{system_prompt}

Context Information:
{context}

User Question: {question}

Please provide a helpful and accurate response based on the context above."""

    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the vector database collection."""
        if not self._initialized:
            self.initialize()
        return self.vector_db.get_collection_info()

    def clear_database(self):
        """Clear the vector database."""
        if not self._initialized:
            self.initialize()
        self.vector_db.clear_collection()
