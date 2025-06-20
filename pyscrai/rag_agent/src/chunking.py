"""Document chunking utilities for the Concordia Assistant."""

import json
import re
from typing import List, Dict, Any, Tuple
from pathlib import Path
from ..config_loader import AgentConfig 


class DocumentChunker:
    """Document chunker for different file types and strategies."""
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.chunk_size = config.chunking.chunk_size
        self.overlap = config.chunking.overlap
        self.json_strategy = config.chunking.json_strategy
        self.text_strategy = config.chunking.text_strategy
    
    def chunk_text_file(self, file_path: str) -> List[Tuple[str, Dict[str, Any]]]:
        """Chunk a text file into smaller pieces."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if self.text_strategy == "semantic":
                chunks = self._semantic_chunk_text(content)
            else:
                chunks = self._fixed_size_chunk_text(content)
            
            # Add metadata to each chunk
            result = []
            for i, chunk in enumerate(chunks):
                metadata = {
                    "source": file_path,
                    "chunk_id": i,
                    "chunk_type": "text",
                    "strategy": self.text_strategy
                }
                result.append((chunk, metadata))
            
            return result
            
        except Exception as e:
            print(f"Error chunking text file {file_path}: {e}")
            return []
    
    def chunk_json_file(self, file_path: str) -> List[Tuple[str, Dict[str, Any]]]:
        """Chunk a JSON file based on its structure."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if self.json_strategy == "hierarchical":
                chunks = self._hierarchical_chunk_json(data, file_path)
            else:
                # Fallback to string representation chunking
                json_str = json.dumps(data, indent=2)
                text_chunks = self._fixed_size_chunk_text(json_str)
                chunks = []
                for i, chunk in enumerate(text_chunks):
                    metadata = {
                        "source": file_path,
                        "chunk_id": i,
                        "chunk_type": "json",
                        "strategy": "string_based"
                    }
                    chunks.append((chunk, metadata))
            
            return chunks
            
        except Exception as e:
            print(f"Error chunking JSON file {file_path}: {e}")
            return []
    
    def _semantic_chunk_text(self, text: str) -> List[str]:
        """Chunk text based on semantic boundaries (paragraphs, sections)."""
        # Split by double newlines (paragraphs) first
        paragraphs = re.split(r'\n\s*\n', text)
        
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            # If adding this paragraph would exceed chunk size, start a new chunk
            if len(current_chunk) + len(paragraph) + 2 > self.chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = paragraph
            else:
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
        
        # Add the last chunk
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        # If we still have chunks that are too large, fall back to fixed-size chunking
        final_chunks = []
        for chunk in chunks:
            if len(chunk) <= self.chunk_size:
                final_chunks.append(chunk)
            else:
                final_chunks.extend(self._fixed_size_chunk_text(chunk))
        
        return final_chunks
    
    def _fixed_size_chunk_text(self, text: str) -> List[str]:
        """Chunk text into fixed-size pieces with overlap."""
        if len(text) <= self.chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            # If this is not the last chunk, try to break at a word boundary
            if end < len(text):
                # Look for the last space within the chunk
                last_space = text.rfind(' ', start, end)
                if last_space > start:
                    end = last_space
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Move start position with overlap
            start = end - self.overlap
            if start >= len(text):
                break
        
        return chunks
    
    def _hierarchical_chunk_json(self, data: Any, source_path: str, prefix: str = "") -> List[Tuple[str, Dict[str, Any]]]:
        """Chunk JSON data hierarchically based on structure."""
        chunks = []
        
        if isinstance(data, dict):
            for key, value in data.items():
                current_prefix = f"{prefix}.{key}" if prefix else key
                
                if isinstance(value, (dict, list)):
                    # Recursively chunk nested structures
                    chunks.extend(self._hierarchical_chunk_json(value, source_path, current_prefix))
                else:
                    # Leaf node - create a chunk
                    chunk_text = f"{current_prefix}: {json.dumps(value, ensure_ascii=False)}"
                    metadata = {
                        "source": source_path,
                        "json_path": current_prefix,
                        "chunk_type": "json",
                        "strategy": "hierarchical",
                        "data_type": type(value).__name__
                    }
                    chunks.append((chunk_text, metadata))
        
        elif isinstance(data, list):
            for i, item in enumerate(data):
                current_prefix = f"{prefix}[{i}]" if prefix else f"[{i}]"
                
                if isinstance(item, (dict, list)):
                    chunks.extend(self._hierarchical_chunk_json(item, source_path, current_prefix))
                else:
                    chunk_text = f"{current_prefix}: {json.dumps(item, ensure_ascii=False)}"
                    metadata = {
                        "source": source_path,
                        "json_path": current_prefix,
                        "chunk_type": "json",
                        "strategy": "hierarchical",
                        "data_type": type(item).__name__
                    }
                    chunks.append((chunk_text, metadata))
        
        else:
            # Simple value
            chunk_text = f"{prefix}: {json.dumps(data, ensure_ascii=False)}" if prefix else json.dumps(data, ensure_ascii=False)
            metadata = {
                "source": source_path,
                "json_path": prefix,
                "chunk_type": "json",
                "strategy": "hierarchical",
                "data_type": type(data).__name__
            }
            chunks.append((chunk_text, metadata))
        
        return chunks
    
    def chunk_directory(self, directory_path: str, file_type: str = "auto") -> List[Tuple[str, Dict[str, Any]]]:
        """Chunk all files in a directory."""
        directory = Path(directory_path)
        all_chunks = []
        
        if not directory.exists():
            print(f"Directory not found: {directory_path}")
            return []
        
        for file_path in directory.rglob("*"):
            if file_path.is_file():
                file_ext = file_path.suffix.lower()
                
                if file_type == "auto":
                    if file_ext == ".json":
                        chunks = self.chunk_json_file(str(file_path))
                    elif file_ext in [".txt", ".md", ".rst"]:
                        chunks = self.chunk_text_file(str(file_path))
                    else:
                        # Skip unknown file types
                        continue
                elif file_type == "json" and file_ext == ".json":
                    chunks = self.chunk_json_file(str(file_path))
                elif file_type == "text" and file_ext in [".txt", ".md", ".rst"]:
                    chunks = self.chunk_text_file(str(file_path))
                else:
                    continue
                
                all_chunks.extend(chunks)
        
        return all_chunks
