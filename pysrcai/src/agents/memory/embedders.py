"""Simple embedding functions for memory systems.

This module provides basic text embedding capabilities that don't require
external dependencies like sentence transformers or OpenAI embeddings.
"""

import hashlib
import re
from collections import Counter
from typing import Any
import math

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


class SimpleEmbedder:
    """Simple text embedder using TF-IDF-like features."""
    
    def __init__(self, vocab_size: int = 1000):
        """Initialize the simple embedder.
        
        Args:
            vocab_size: Size of the vocabulary to track.
        """
        self.vocab_size = vocab_size
        self.word_counts = Counter()
        self.doc_counts = Counter()
        self.total_docs = 0
        
    def _tokenize(self, text: str) -> list[str]:
        """Simple tokenization."""
        # Convert to lowercase and extract words
        text = text.lower()
        words = re.findall(r'\b\w+\b', text)
        return words
    
    def _update_vocab(self, text: str) -> None:
        """Update vocabulary with new text."""
        words = self._tokenize(text)
        unique_words = set(words)
        
        # Update document frequency
        for word in unique_words:
            self.doc_counts[word] += 1
        
        # Update word frequency
        for word in words:
            self.word_counts[word] += 1
        
        self.total_docs += 1
    
    def embed(self, text: str) -> list[float]:
        """Create a simple embedding for text.
        
        Args:
            text: Text to embed.
            
        Returns:
            A list of floats representing the embedding.
        """
        # Update vocabulary
        self._update_vocab(text)
        
        # Get words and their frequencies in this document
        words = self._tokenize(text)
        word_freq = Counter(words)
        doc_length = len(words)
        
        if doc_length == 0:
            return [0.0] * min(self.vocab_size, 100)  # Return zero vector
        
        # Get top words by global frequency for vocab
        top_words = [word for word, _ in self.word_counts.most_common(self.vocab_size)]
        
        # Create TF-IDF-like features
        features = []
        for word in top_words[:100]:  # Limit to first 100 features for simplicity
            if word in word_freq:
                # Term frequency
                tf = word_freq[word] / doc_length
                
                # Inverse document frequency
                idf = math.log(self.total_docs / (self.doc_counts[word] + 1))
                
                # TF-IDF score
                tfidf = tf * idf
                features.append(tfidf)
            else:
                features.append(0.0)
        
        # Pad or truncate to consistent length
        target_length = min(self.vocab_size, 100)
        if len(features) < target_length:
            features.extend([0.0] * (target_length - len(features)))
        else:
            features = features[:target_length]
        
        return features


class HashEmbedder:
    """Simple hash-based embedder for basic similarity."""
    
    def __init__(self, dimensions: int = 100):
        """Initialize hash embedder.
        
        Args:
            dimensions: Number of dimensions in the embedding.
        """
        self.dimensions = dimensions
    
    def embed(self, text: str) -> list[float]:
        """Create a hash-based embedding.
        
        Args:
            text: Text to embed.
            
        Returns:
            A list of floats representing the embedding.
        """
        # Normalize text
        text = text.lower().strip()
        
        # Create multiple hash features
        features = []
        
        # Use different hash functions by adding salt
        for i in range(self.dimensions):
            salted_text = f"{text}_{i}"
            hash_value = hashlib.md5(salted_text.encode()).hexdigest()
            
            # Convert hex to float between -1 and 1
            hash_int = int(hash_value[:8], 16)  # Use first 8 hex chars
            normalized = (hash_int / (16**8)) * 2 - 1  # Normalize to [-1, 1]
            features.append(normalized)
        
        return features


def create_simple_embedder() -> Any:
    """Create a simple embedder that works without external dependencies."""
    if HAS_NUMPY:
        # Use numpy version with better linear algebra
        class NumpySimpleEmbedder(SimpleEmbedder):
            def embed(self, text: str) -> Any:
                features = super().embed(text)
                return np.array(features, dtype=np.float32)
        
        return NumpySimpleEmbedder()
    else:
        # Use pure Python version
        return SimpleEmbedder()


def create_hash_embedder() -> Any:
    """Create a hash-based embedder."""
    if HAS_NUMPY:
        class NumpyHashEmbedder(HashEmbedder):
            def embed(self, text: str) -> Any:
                features = super().embed(text)
                return np.array(features, dtype=np.float32)
        
        return NumpyHashEmbedder()
    else:
        return HashEmbedder()
