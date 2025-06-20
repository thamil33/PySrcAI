"""Document chunking strategies."""

import json
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from langchain.schema import Document
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter
)

from ..config.config import ChunkingConfig


class BaseChunker(ABC):
    """Base class for document chunkers."""
    
    def __init__(self, config: ChunkingConfig):
        """Initialize the chunker with configuration.
        
        Args:
            config: Chunking configuration
        """
        self.config = config
    
    @abstractmethod
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Chunk a list of documents.
        
        Args:
            documents: List of documents to chunk
            
        Returns:
            List of chunked documents
        """
        pass


class TextChunker(BaseChunker):
    """Simple text chunker using LangChain's text splitters."""
    
    def __init__(self, config: ChunkingConfig):
        """Initialize the text chunker."""
        super().__init__(config)
        
        # Use RecursiveCharacterTextSplitter for better chunking
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Chunk documents using text splitter.
        
        Args:
            documents: List of documents to chunk
            
        Returns:
            List of chunked documents
        """
        chunked_docs = []
        
        for doc in documents:
            # Split the document
            chunks = self.text_splitter.split_documents([doc])
            
            # Add chunk metadata
            for i, chunk in enumerate(chunks):
                chunk.metadata.update({
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "chunk_size": len(chunk.page_content),
                    "chunking_strategy": "text"
                })
            
            chunked_docs.extend(chunks)
        
        return chunked_docs


class SemanticChunker(BaseChunker):
    """Semantic chunker that tries to maintain semantic boundaries."""
    
    def __init__(self, config: ChunkingConfig):
        """Initialize the semantic chunker."""
        super().__init__(config)
        
        # Use different separators for semantic chunking
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.overlap,
            length_function=len,
            separators=[
                "\n# ",      # Markdown headers
                "\n## ",     # Markdown subheaders
                "\n### ",    # Markdown sub-subheaders
                "\n\n",      # Paragraph breaks
                "\n",        # Line breaks
                ". ",        # Sentence endings
                " ",         # Word boundaries
                ""           # Character level
            ]
        )
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Chunk documents semantically.
        
        Args:
            documents: List of documents to chunk
            
        Returns:
            List of chunked documents
        """
        chunked_docs = []
        
        for doc in documents:
            # For markdown files, try to preserve structure
            if doc.metadata.get("file_type") == "markdown":
                chunks = self._chunk_markdown(doc)
            else:
                chunks = self.text_splitter.split_documents([doc])
            
            # Add chunk metadata
            for i, chunk in enumerate(chunks):
                chunk.metadata.update({
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "chunk_size": len(chunk.page_content),
                    "chunking_strategy": "semantic"
                })
            
            chunked_docs.extend(chunks)
        
        return chunked_docs
    
    def _chunk_markdown(self, doc: Document) -> List[Document]:
        """Special handling for markdown documents."""
        content = doc.page_content
        
        # Split by headers first to maintain structure
        sections = []
        current_section = ""
        
        for line in content.split('\n'):
            if line.startswith('#'):
                if current_section.strip():
                    sections.append(current_section.strip())
                current_section = line + '\n'
            else:
                current_section += line + '\n'
        
        if current_section.strip():
            sections.append(current_section.strip())
        
        # If sections are too large, further split them
        chunks = []
        for section in sections:
            if len(section) <= self.config.chunk_size:
                chunks.append(Document(
                    page_content=section,
                    metadata=doc.metadata.copy()
                ))
            else:
                # Split large sections
                section_doc = Document(page_content=section, metadata=doc.metadata.copy())
                sub_chunks = self.text_splitter.split_documents([section_doc])
                chunks.extend(sub_chunks)
        
        return chunks


class HierarchicalChunker(BaseChunker):
    """Hierarchical chunker for JSON documents."""
    
    def __init__(self, config: ChunkingConfig):
        """Initialize the hierarchical chunker."""
        super().__init__(config)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.overlap
        )
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Chunk documents hierarchically.
        
        Args:
            documents: List of documents to chunk
            
        Returns:
            List of chunked documents
        """
        chunked_docs = []
        
        for doc in documents:
            if doc.metadata.get("file_type") == "json":
                chunks = self._chunk_json(doc)
            else:
                chunks = self.text_splitter.split_documents([doc])
            
            # Add chunk metadata
            for i, chunk in enumerate(chunks):
                chunk.metadata.update({
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "chunk_size": len(chunk.page_content),
                    "chunking_strategy": "hierarchical"
                })
            
            chunked_docs.extend(chunks)
        
        return chunked_docs
    
    def _chunk_json(self, doc: Document) -> List[Document]:
        """Special handling for JSON documents."""
        try:
            # Try to parse as JSON
            data = json.loads(doc.page_content)
            chunks = []
            
            if isinstance(data, dict):
                chunks.extend(self._chunk_json_object(data, doc.metadata, []))
            elif isinstance(data, list):
                for i, item in enumerate(data):
                    if isinstance(item, dict):
                        chunks.extend(self._chunk_json_object(item, doc.metadata, [str(i)]))
                    else:
                        chunk_content = json.dumps(item, indent=2)
                        if len(chunk_content) <= self.config.chunk_size:
                            chunk_metadata = doc.metadata.copy()
                            chunk_metadata["json_path"] = f"[{i}]"
                            chunks.append(Document(
                                page_content=chunk_content,
                                metadata=chunk_metadata
                            ))
            
            return chunks if chunks else [doc]
            
        except json.JSONDecodeError:
            # If not valid JSON, fall back to text chunking
            return self.text_splitter.split_documents([doc])
    
    def _chunk_json_object(
        self, 
        obj: Dict[str, Any], 
        base_metadata: Dict[str, Any], 
        path: List[str]
    ) -> List[Document]:
        """Recursively chunk a JSON object."""
        chunks = []
        
        for key, value in obj.items():
            current_path = path + [key]
            path_str = ".".join(current_path)
            
            if isinstance(value, dict):
                # Recursively handle nested objects
                chunks.extend(self._chunk_json_object(value, base_metadata, current_path))
            elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], dict):
                # Handle arrays of objects
                for i, item in enumerate(value):
                    item_path = current_path + [str(i)]
                    chunks.extend(self._chunk_json_object(item, base_metadata, item_path))
            else:
                # Handle primitive values or simple arrays
                content = json.dumps({key: value}, indent=2)
                if len(content) <= self.config.chunk_size:
                    chunk_metadata = base_metadata.copy()
                    chunk_metadata["json_path"] = path_str
                    chunks.append(Document(
                        page_content=content,
                        metadata=chunk_metadata
                    ))
        
        return chunks


def create_chunker(config: ChunkingConfig, strategy: Optional[str] = None) -> BaseChunker:
    """Create a chunker based on configuration and strategy.
    
    Args:
        config: Chunking configuration
        strategy: Chunking strategy ('text', 'semantic', 'hierarchical', or None for auto-detect)
        
    Returns:
        Chunker instance
    """
    if strategy:
        chunkers = {
            'text': TextChunker,
            'semantic': SemanticChunker,
            'hierarchical': HierarchicalChunker
        }
        if strategy in chunkers:
            return chunkers[strategy](config)
    
    # Auto-detect based on config
    if config.text_strategy == "semantic":
        return SemanticChunker(config)
    elif config.json_strategy == "hierarchical":
        return HierarchicalChunker(config)
    else:
        return TextChunker(config)
