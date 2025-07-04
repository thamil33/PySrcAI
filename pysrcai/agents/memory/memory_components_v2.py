"""Memory components for PySrcAI agents - Version 2 with improved typing."""

import abc
import threading
from collections.abc import Callable, Sequence
from typing import Any, Dict, List, Optional, Union
import time
import json

try:
    import numpy as np
    HAS_EMBEDDINGS = True
except ImportError:
    HAS_EMBEDDINGS = False

from ..agent import ContextComponent, ActionSpec, ComponentState


class MemoryBank(abc.ABC):
    """Abstract base class for memory storage systems."""
    
    @abc.abstractmethod
    def add_memory(self, text: str, tags: Optional[List[str]] = None, importance: float = 1.0) -> None:
        """Add a memory to the bank."""
        raise NotImplementedError()
    
    @abc.abstractmethod
    def retrieve_recent(self, k: int = 5) -> List[str]:
        """Retrieve the k most recent memories."""
        raise NotImplementedError()
    
    @abc.abstractmethod
    def retrieve_by_query(self, query: str, k: int = 5) -> List[str]:
        """Retrieve memories relevant to a query."""
        raise NotImplementedError()
    
    @abc.abstractmethod
    def get_all_memories(self) -> List[str]:
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
        self._memories: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
    
    def add_memory(self, text: str, tags: Optional[List[str]] = None, importance: float = 1.0) -> None:
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
    
    def retrieve_recent(self, k: int = 5) -> List[str]:
        """Retrieve the k most recent memories."""
        with self._lock:
            recent_memories = self._memories[-k:] if self._memories else []
            return [mem['text'] for mem in reversed(recent_memories)]
    
    def retrieve_by_query(self, query: str, k: int = 5) -> List[str]:
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
    
    def retrieve_by_tags(self, tags: List[str], k: int = 5) -> List[str]:
        """Retrieve memories that have any of the specified tags."""
        relevant_memories = []
        
        with self._lock:
            for memory in reversed(self._memories):  # Most recent first
                if any(tag in memory['tags'] for tag in tags):
                    relevant_memories.append(memory['text'])
                    if len(relevant_memories) >= k:
                        break
        
        return relevant_memories
    
    def get_all_memories(self) -> List[str]:
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
                # Type cast to ensure compatibility
                self._memories = [mem for mem in memories if isinstance(mem, dict)]
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
        embedder: Optional[Callable[[str], Any]] = None,
        max_memories: int = 1000,
    ):
        """Initialize the associative memory bank.
        
        Args:
            embedder: Function to create embeddings from text.
            max_memories: Maximum number of memories to store.
        """
        if not HAS_EMBEDDINGS:
            raise ImportError("AssociativeMemoryBank requires numpy")
        
        self._embedder = embedder
        self._max_memories: int = max_memories
        self._lock = threading.Lock()
        self._memories: List[Dict[str, Any]] = []
        self._stored_hashes = set()
    
    def set_embedder(self, embedder: Callable[[str], Any]) -> None:
        """Set the embedder function."""
        self._embedder = embedder
    
    def add_memory(self, text: str, tags: Optional[List[str]] = None, importance: float = 1.0) -> None:
        """Add a memory to the bank."""
        if not self._embedder:
            raise ValueError("Embedder must be set before adding memories")
        
        # Normalize text
        text = text.replace('\n', ' ')
        
        # Create memory record
        memory_data = {
            'text': text,
            'timestamp': time.time(),
            'tags': tags or [],
            'importance': importance,
        }
        
        # Check for duplicates
        content_hash = hash(tuple(str(v) for v in memory_data.values()))
        
        with self._lock:
            if content_hash in self._stored_hashes:
                return
            
            # Create embedding
            embedding = self._embedder(text)
            memory_data['embedding'] = embedding
            
            # Add to memory list
            self._memories.append(memory_data)
            self._stored_hashes.add(content_hash)
            
            # Limit memory size
            if len(self._memories) > self._max_memories:
                # Remove oldest memory
                oldest_memory = min(self._memories, key=lambda m: m['timestamp'])
                self._memories.remove(oldest_memory)
    
    def retrieve_recent(self, k: int = 5) -> List[str]:
        """Retrieve the k most recent memories."""
        with self._lock:
            if not self._memories:
                return []
            # Sort by timestamp and get most recent
            sorted_memories = sorted(self._memories, key=lambda m: m['timestamp'], reverse=True)
            return [mem['text'] for mem in sorted_memories[:k]]
    
    def retrieve_by_query(self, query: str, k: int = 5) -> List[str]:
        """Retrieve memories most similar to the query using embeddings."""
        if not self._embedder:
            raise ValueError("Embedder must be set before retrieving memories")
        
        with self._lock:
            if not self._memories:
                return []
            
            # Get query embedding
            query_embedding = self._embedder(query)
            
            # Calculate cosine similarities
            similarities = []
            for memory in self._memories:
                emb = memory['embedding']
                if isinstance(emb, (list, np.ndarray)):
                    # Calculate cosine similarity
                    dot_product = np.dot(query_embedding, emb)
                    norm_query = np.linalg.norm(query_embedding)
                    norm_emb = np.linalg.norm(emb)
                    if norm_query > 0 and norm_emb > 0:
                        similarity = dot_product / (norm_query * norm_emb)
                    else:
                        similarity = 0
                    similarities.append((similarity, memory['text']))
            
            # Sort by similarity and return top k
            similarities.sort(reverse=True)  # Highest similarity first
            return [text for _, text in similarities[:k]]
    
    def retrieve_by_tags(self, tags: List[str], k: int = 5) -> List[str]:
        """Retrieve memories that have any of the specified tags."""
        with self._lock:
            if not self._memories:
                return []
            
            # Filter by tags
            relevant_memories = []
            for memory in self._memories:
                memory_tags = memory['tags']
                if any(tag in memory_tags for tag in tags):
                    relevant_memories.append((memory['timestamp'], memory['text']))
            
            # Sort by recency and return top k
            relevant_memories.sort(reverse=True)  # Most recent first
            return [text for _, text in relevant_memories[:k]]
    
    def get_all_memories(self) -> List[str]:
        """Get all memories."""
        with self._lock:
            return [mem['text'] for mem in self._memories]
    
    def get_state(self) -> ComponentState:
        """Get serializable state."""
        with self._lock:
            return {
                'memories': self._memories.copy(),
                'stored_hashes': list(self._stored_hashes),
                'max_memories': self._max_memories,
            }
    
    def set_state(self, state: ComponentState) -> None:
        """Set state from serialized data."""
        with self._lock:
            memories = state.get('memories', [])
            if isinstance(memories, list):
                # Type cast to ensure compatibility
                self._memories = [mem for mem in memories if isinstance(mem, dict)]
            else:
                self._memories = []
            
            stored_hashes = state.get('stored_hashes', [])
            if isinstance(stored_hashes, list):
                self._stored_hashes = set(stored_hashes)
            else:
                self._stored_hashes = set()
            
            max_memories = state.get('max_memories', 1000)
            if isinstance(max_memories, int):
                self._max_memories = max_memories
            else:
                self._max_memories = 1000
    
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
        try:
            memory_count = len(self._memory_bank)
        except TypeError:
            memory_count = 0
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
        memory_bank_state = state.get('memory_bank_state')
        if memory_bank_state and isinstance(memory_bank_state, dict):
            self._memory_bank.set_state(memory_bank_state)
        
        importance_threshold = state.get('importance_threshold', 0.5)
        if isinstance(importance_threshold, (int, float)):
            self._importance_threshold = float(importance_threshold)
        
        max_context_memories = state.get('max_context_memories', 5)
        if isinstance(max_context_memories, int):
            self._max_context_memories = max_context_memories
    
    def get_memory_bank(self) -> MemoryBank:
        """Get the underlying memory bank."""
        return self._memory_bank
    
    def add_explicit_memory(self, text: str, tags: Optional[List[str]] = None, importance: float = 1.0) -> None:
        """Explicitly add a memory (useful for initialization or external events)."""
        agent = self.get_agent()
        memory_text = f"{agent.name}: {text}"
        self._memory_bank.add_memory(memory_text, tags=tags, importance=importance) 