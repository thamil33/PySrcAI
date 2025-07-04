"""Memory components for PySrcAI agents.

This module provides memory capabilities for agents, including:
- Basic memory storage and retrieval
- Associative memory with embedding-based search
- Context components that provide memory-based context
- Integration with the agent component system
"""

import abc
import threading
from collections.abc import Callable, Sequence
from typing import Any, Dict, List, Union
import time
import json

try:
    import numpy as np
    import pandas as pd
    HAS_EMBEDDINGS = True
except ImportError:
    HAS_EMBEDDINGS = False

from ..agent import ContextComponent, ActionSpec, ComponentState


class MemoryBank(abc.ABC):
    """Abstract base class for memory storage systems."""
    
    @abc.abstractmethod
    def add_memory(self, text: str, tags: list[str] | None = None, importance: float = 1.0) -> None:
        """Add a memory to the bank."""
        raise NotImplementedError()
    
    @abc.abstractmethod
    def retrieve_recent(self, k: int = 5) -> Sequence[str]:
        """Retrieve the k most recent memories."""
        raise NotImplementedError()
    
    @abc.abstractmethod
    def retrieve_by_query(self, query: str, k: int = 5) -> Sequence[str]:
        """Retrieve memories relevant to a query."""
        raise NotImplementedError()
    
    @abc.abstractmethod
    def get_all_memories(self) -> Sequence[str]:
        """Get all memories."""
        raise NotImplementedError()
    
    @abc.abstractmethod
    def get_state(self) -> ComponentState:
        """Get serializable state."""
        raise NotImplementedError()
    
    @abc.abstractmethod
    def set_state(self, state: ComponentState) -> None:
        """Set state from serialized data."""
        raise NotImplementedError()


class BasicMemoryBank(MemoryBank):
    """Simple memory bank with chronological storage and text-based search."""
    
    def __init__(self, max_memories: int = 1000):
        """Initialize the basic memory bank.
        
        Args:
            max_memories: Maximum number of memories to store.
        """
        self._max_memories: int = max_memories
        self._memories: list[dict[str, Any]] = []
        self._lock = threading.Lock()
    
    def add_memory(self, text: str, tags: list[str] | None = None, importance: float = 1.0) -> None:
        """Add a memory to the bank."""
        memory = {
            'text': text.replace('\n', ' '),  # Normalize newlines
            'timestamp': time.time(),
            'tags': tags or [],
            'importance': importance,
        }
        
        with self._lock:
            self._memories.append(memory)
            # Keep only the most recent memories
            if len(self._memories) > self._max_memories:
                self._memories = self._memories[-self._max_memories:]
    
    def retrieve_recent(self, k: int = 5) -> Sequence[str]:
        """Retrieve the k most recent memories."""
        with self._lock:
            recent_memories = self._memories[-k:] if self._memories else []
            return [mem['text'] for mem in reversed(recent_memories)]
    
    def retrieve_by_query(self, query: str, k: int = 5) -> Sequence[str]:
        """Retrieve memories that contain query terms (simple text search)."""
        query_lower = query.lower()
        relevant_memories = []
        
        with self._lock:
            for memory in reversed(self._memories):  # Most recent first
                if query_lower in memory['text'].lower():
                    relevant_memories.append(memory['text'])
                    if len(relevant_memories) >= k:
                        break
        
        return relevant_memories
    
    def retrieve_by_tags(self, tags: list[str], k: int = 5) -> Sequence[str]:
        """Retrieve memories that have any of the specified tags."""
        relevant_memories = []
        
        with self._lock:
            for memory in reversed(self._memories):  # Most recent first
                if any(tag in memory['tags'] for tag in tags):
                    relevant_memories.append(memory['text'])
                    if len(relevant_memories) >= k:
                        break
        
        return relevant_memories
    
    def get_all_memories(self) -> Sequence[str]:
        """Get all memories."""
        with self._lock:
            return [mem['text'] for mem in self._memories]
    
    def get_state(self) -> ComponentState:
        """Get serializable state."""
        with self._lock:
            return {
                'memories': self._memories.copy(),
                'max_memories': self._max_memories,
            }
    
    def set_state(self, state: ComponentState) -> None:
        """Set state from serialized data."""
        with self._lock:
            memories = state.get('memories', [])
            if isinstance(memories, list):
                self._memories = memories
            else:
                self._memories = []
            
            max_memories = state.get('max_memories', 1000)
            if isinstance(max_memories, int):
                self._max_memories = max_memories
            else:
                self._max_memories = 1000
    
    def __len__(self) -> int:
        """Return the number of memories."""
        with self._lock:
            return len(self._memories)


class AssociativeMemoryBank(MemoryBank):
    """Advanced memory bank with embedding-based associative retrieval."""
    
    def __init__(
        self, 
        embedder: Callable[[str], Any] | None = None,
        max_memories: int = 1000,
    ):
        """Initialize the associative memory bank.
        
        Args:
            embedder: Function to create embeddings from text.
            max_memories: Maximum number of memories to store.
        """
        if not HAS_EMBEDDINGS:
            raise ImportError("AssociativeMemoryBank requires numpy and pandas")
        
        self._embedder = embedder
        self._max_memories = max_memories
        self._lock = threading.Lock()
        self._memories = pd.DataFrame(columns=['text', 'timestamp', 'tags', 'importance', 'embedding'])
        self._stored_hashes = set()
    
    def set_embedder(self, embedder: Callable[[str], Any]) -> None:
        """Set the embedder function."""
        self._embedder = embedder
    
    def add_memory(self, text: str, tags: list[str] | None = None, importance: float = 1.0) -> None:
        """Add a memory to the bank."""
        if not self._embedder:
            raise ValueError("Embedder must be set before adding memories")
        
        # Normalize text
        text = text.replace('\n', ' ')
        
        # Create memory record
        memory_data = {
            'text': text,
            'timestamp': time.time(),
            'tags': json.dumps(tags or []),  # Store as JSON string
            'importance': importance,
        }
        
        # Check for duplicates
        content_hash = hash(tuple(memory_data.values()))
        
        with self._lock:
            if content_hash in self._stored_hashes:
                return
            
            # Create embedding
            embedding = self._embedder(text)
            memory_data['embedding'] = embedding
            
            # Add to dataframe
            new_memory = pd.Series(memory_data).to_frame().T.infer_objects()
            
            # Handle concatenation to avoid deprecation warnings
            if not new_memory.empty:
                if self._memories.empty:
                    self._memories = new_memory
                else:
                    self._memories = pd.concat([self._memories, new_memory], ignore_index=True)
            self._stored_hashes.add(content_hash)
            
            # Limit memory size
            if len(self._memories) > self._max_memories:
                oldest_index = self._memories['timestamp'].idxmin()
                self._memories = self._memories.drop(oldest_index)
    
    def retrieve_recent(self, k: int = 5) -> Sequence[str]:
        """Retrieve the k most recent memories."""
        with self._lock:
            if self._memories.empty:
                return []
            recent = self._memories.nlargest(k, 'timestamp')
            return recent['text'].tolist()
    
    def retrieve_by_query(self, query: str, k: int = 5) -> Sequence[str]:
        """Retrieve memories most similar to the query using embeddings."""
        if not self._embedder:
            raise ValueError("Embedder must be set before retrieving memories")
        
        with self._lock:
            if self._memories.empty:
                return []
            
            # Get query embedding
            query_embedding = self._embedder(query)
            
            # Calculate cosine similarities
            similarities = self._memories['embedding'].apply(
                lambda emb: np.dot(query_embedding, emb) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(emb)
                ) if np.linalg.norm(emb) > 0 else 0
            )
            
            # Get top k most similar
            top_indices = similarities.nlargest(k).index
            return self._memories.loc[top_indices, 'text'].tolist()
    
    def retrieve_by_tags(self, tags: list[str], k: int = 5) -> Sequence[str]:
        """Retrieve memories that have any of the specified tags."""
        with self._lock:
            if self._memories.empty:
                return []
            
            # Filter by tags (stored as JSON strings)
            relevant_memories = []
            for _, memory in self._memories.iterrows():
                memory_tags = json.loads(memory['tags'])
                if any(tag in memory_tags for tag in tags):
                    relevant_memories.append((memory['timestamp'], memory['text']))
            
            # Sort by recency and return top k
            relevant_memories.sort(reverse=True)  # Most recent first
            return [text for _, text in relevant_memories[:k]]
    
    def get_all_memories(self) -> Sequence[str]:
        """Get all memories."""
        with self._lock:
            if self._memories.empty:
                return []
            return self._memories['text'].tolist()
    
    def get_state(self) -> ComponentState:
        """Get serializable state."""
        with self._lock:
            return {
                'memories_json': self._memories.to_json() if not self._memories.empty else '{}',
                'stored_hashes': list(self._stored_hashes),
                'max_memories': self._max_memories,
            }
    
    def set_state(self, state: ComponentState) -> None:
        """Set state from serialized data."""
        with self._lock:
            memories_json = state.get('memories_json', '{}')
            if memories_json != '{}':
                self._memories = pd.read_json(memories_json)
            else:
                self._memories = pd.DataFrame(columns=['text', 'timestamp', 'tags', 'importance', 'embedding'])
            
            self._stored_hashes = set(state.get('stored_hashes', []))
            self._max_memories = state.get('max_memories', 1000)
    
    def __len__(self) -> int:
        """Return the number of memories."""
        with self._lock:
            return len(self._memories)


class MemoryComponent(ContextComponent):
    """Context component that provides memory-based context to agents."""
    
    def __init__(
        self,
        memory_bank: MemoryBank,
        memory_importance_threshold: float = 0.5,
        max_context_memories: int = 5,
    ):
        """Initialize the memory component.
        
        Args:
            memory_bank: The memory bank to use for storage and retrieval.
            memory_importance_threshold: Minimum importance for memories to include in context.
            max_context_memories: Maximum number of memories to include in context.
        """
        super().__init__()
        self._memory_bank = memory_bank
        self._importance_threshold = memory_importance_threshold
        self._max_context_memories = max_context_memories
    
    def pre_act(self, action_spec: ActionSpec) -> str:
        """Provide relevant memories as context for acting."""
        # Get recent memories
        recent_memories = self._memory_bank.retrieve_recent(self._max_context_memories)
        
        # Try to get memories relevant to the action
        query_memories = []
        if hasattr(self._memory_bank, 'retrieve_by_query'):
            query = action_spec.call_to_action
            query_memories = self._memory_bank.retrieve_by_query(query, self._max_context_memories // 2)
        
        # Combine and deduplicate
        all_memories = list(dict.fromkeys(query_memories + recent_memories))[:self._max_context_memories]
        
        if not all_memories:
            return "No relevant memories found."
        
        return f"Relevant memories:\n" + "\n".join(f"- {memory}" for memory in all_memories)
    
    def post_act(self, action_attempt: str) -> str:
        """Store the action as a memory."""
        agent = self.get_agent()
        memory_text = f"{agent.name} acted: {action_attempt}"
        self._memory_bank.add_memory(memory_text, tags=['action'], importance=0.7)
        return f"Stored action in memory: {action_attempt[:50]}..."
    
    def pre_observe(self, observation: str) -> str:
        """Store the observation and provide related memories."""
        # Store the observation
        agent = self.get_agent()
        memory_text = f"{agent.name} observed: {observation}"
        self._memory_bank.add_memory(memory_text, tags=['observation'], importance=0.8)
        
        # Find related memories
        related_memories = []
        if hasattr(self._memory_bank, 'retrieve_by_query'):
            related_memories = self._memory_bank.retrieve_by_query(observation, 3)
        
        if related_memories:
            return f"Related past experiences:\n" + "\n".join(f"- {memory}" for memory in related_memories)
        else:
            return "No related past experiences found."
    
    def post_observe(self) -> str:
        """Provide summary of current memory state."""
        memory_count = len(self._memory_bank)
        return f"Memory bank contains {memory_count} memories."
    
    def get_state(self) -> ComponentState:
        """Get the component's state."""
        return {
            'memory_bank_state': self._memory_bank.get_state(),
            'importance_threshold': self._importance_threshold,
            'max_context_memories': self._max_context_memories,
        }
    
    def set_state(self, state: ComponentState) -> None:
        """Set the component's state."""
        if 'memory_bank_state' in state:
            self._memory_bank.set_state(state['memory_bank_state'])
        self._importance_threshold = state.get('importance_threshold', 0.5)
        self._max_context_memories = state.get('max_context_memories', 5)
    
    def get_memory_bank(self) -> MemoryBank:
        """Get the underlying memory bank."""
        return self._memory_bank
    
    def add_explicit_memory(self, text: str, tags: list[str] | None = None, importance: float = 1.0) -> None:
        """Explicitly add a memory (useful for initialization or external events)."""
        agent = self.get_agent()
        memory_text = f"{agent.name}: {text}"
        self._memory_bank.add_memory(memory_text, tags=tags, importance=importance)
