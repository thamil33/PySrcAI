"""Document loaders for various file formats."""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional

from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
    JSONLoader
)
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from pysrcai.agentica.config.config import ChunkingConfig


class TextLoader:
    """Loader for plain text files."""

    def __init__(self, file_path: str, encoding: str = "utf-8"):
        """Initialize the text loader.

        Args:
            file_path: Path to the text file
            encoding: File encoding (default: utf-8)
        """
        self.file_path = file_path
        self.encoding = encoding

    def load(self) -> List[Document]:
        """Load the text file as a document.

        Returns:
            List containing a single Document
        """
        try:
            with open(self.file_path, 'r', encoding=self.encoding) as f:
                content = f.read()
        except UnicodeDecodeError as e:
            print(f"Warning: Skipping {self.file_path}: {e}")
            return []

        metadata = {
            "source": self.file_path,
            "file_type": "text",
            "size": len(content)
        }

        return [Document(page_content=content, metadata=metadata)]


class MarkdownLoader:
    """Loader for Markdown files."""

    def __init__(self, file_path: str, encoding: str = "utf-8"):
        """Initialize the markdown loader.

        Args:
            file_path: Path to the markdown file
            encoding: File encoding (default: utf-8)
        """
        self.file_path = file_path
        self.encoding = encoding

    def load(self) -> List[Document]:
        """Load the markdown file as a document.

        Returns:
            List containing a single Document
        """
        try:
            # Try using LangChain's markdown loader if available
            loader = UnstructuredMarkdownLoader(self.file_path)
            return loader.load()
        except ImportError:
            # Fallback to simple text loading
            try:
                with open(self.file_path, 'r', encoding=self.encoding) as f:
                    content = f.read()
            except UnicodeDecodeError as e:
                print(f"Warning: Skipping {self.file_path}: {e}")
                return []

            metadata = {
                "source": self.file_path,
                "file_type": "markdown",
                "size": len(content)
            }

            return [Document(page_content=content, metadata=metadata)]


class JSONLoader:
    """Loader for JSON files."""

    def __init__(
        self,
        file_path: str,
        content_key: Optional[str] = None,
        metadata_keys: Optional[List[str]] = None
    ):
        """Initialize the JSON loader.

        Args:
            file_path: Path to the JSON file
            content_key: Key in JSON containing the main content (if None, uses entire JSON as string)
            metadata_keys: List of keys to include in metadata
        """
        self.file_path = file_path
        self.content_key = content_key
        self.metadata_keys = metadata_keys or []

    def load(self) -> List[Document]:
        """Load the JSON file as document(s).

        Returns:
            List of Documents
        """
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except UnicodeDecodeError as e:
            print(f"Warning: Skipping {self.file_path}: {e}")
            return []
        except json.JSONDecodeError as e:
            print(f"Warning: Skipping {self.file_path} (invalid JSON): {e}")
            return []

        documents = []

        if isinstance(data, list):
            # Handle JSON arrays
            for i, item in enumerate(data):
                content, metadata = self._extract_content_and_metadata(item, i)
                documents.append(Document(page_content=content, metadata=metadata))
        else:
            # Handle single JSON object
            content, metadata = self._extract_content_and_metadata(data)
            documents.append(Document(page_content=content, metadata=metadata))

        return documents

    def _extract_content_and_metadata(self, item: Dict[str, Any], index: Optional[int] = None) -> tuple:
        """Extract content and metadata from a JSON item.

        Args:
            item: JSON item (dict)
            index: Index if part of an array

        Returns:
            Tuple of (content, metadata)
        """
        if self.content_key and self.content_key in item:
            content = str(item[self.content_key])
        else:
            content = json.dumps(item, indent=2)

        metadata = {
            "source": self.file_path,
            "file_type": "json",
            "size": len(content)
        }

        if index is not None:
            metadata["index"] = index

        # Add specified metadata keys
        for key in self.metadata_keys:
            if key in item:
                metadata[key] = item[key]

        return content, metadata


class DirectoryLoader:
    """Loader for directories containing multiple files."""

    def __init__(
        self,
        directory_path: str,
        glob_pattern: str = "**/*",
        file_extensions: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None
    ):
        """Initialize the directory loader.

        Args:
            directory_path: Path to the directory
            glob_pattern: Glob pattern for file matching (default: all files)
            file_extensions: List of file extensions to include (e.g., ['.txt', '.md'])
            exclude_patterns: List of patterns to exclude
        """
        self.directory_path = Path(directory_path)
        self.glob_pattern = glob_pattern
        self.file_extensions = file_extensions or []
        self.exclude_patterns = exclude_patterns or []

    def load(self) -> List[Document]:
        """Load all matching files in the directory.

        Returns:
            List of Documents from all loaded files
        """
        documents = []

        file_count = 0
        doc_count = 0
        for file_path in self.directory_path.glob(self.glob_pattern):
            if not file_path.is_file():
                continue
            # Check file extension filter
            if self.file_extensions and file_path.suffix.lower() not in self.file_extensions:
                continue
            # Check exclude patterns
            if any(pattern in str(file_path) for pattern in self.exclude_patterns):
                continue
            # Load the file based on its extension
            loader = self._get_loader_for_file(file_path)
            if loader:
                try:
                    loaded_docs = loader.load()
                    if loaded_docs:
                        print(f"✅ Ingested {len(loaded_docs)} document(s) from: {file_path}")
                        file_count += 1
                        doc_count += len(loaded_docs)
                    documents.extend(loaded_docs)
                except Exception as e:
                    print(f"Warning: Failed to load {file_path}: {e}")
        print(f"--- Ingestion complete: {file_count} file(s), {doc_count} document(s) loaded ---")
        return documents

    def _get_loader_for_file(self, file_path: Path):
        """Get the appropriate loader for a file based on its extension.

        Args:
            file_path: Path to the file

        Returns:
            Loader instance or None if unsupported
        """
        extension = file_path.suffix.lower()

        if extension in ['.txt', '.py', '.yaml', '.yml', '.toml', '.cfg', '.ini']:
            return TextLoader(str(file_path))
        elif extension in ['.md', '.markdown']:
            return MarkdownLoader(str(file_path))
        elif extension == '.json':
            return JSONLoader(str(file_path))
        else:
            # Default to text loader for unknown extensions
            return TextLoader(str(file_path))


def create_loader(
    path: str,
    loader_type: Optional[str] = None,
    **kwargs
):
    """Create a loader based on path and type.

    Args:
        path: File or directory path
        loader_type: Type of loader ('text', 'markdown', 'json', 'directory', or None for auto-detect)
        **kwargs: Additional arguments for the loader

    Returns:
        Loader instance
    """
    path_obj = Path(path)

    if path_obj.is_dir():
        return DirectoryLoader(path, **kwargs)

    if loader_type:
        loaders = {
            'text': TextLoader,
            'markdown': MarkdownLoader,
            'json': JSONLoader,
        }
        if loader_type in loaders:
            return loaders[loader_type](path, **kwargs)

    # Auto-detect based on file extension
    extension = path_obj.suffix.lower()
    if extension in ['.md', '.markdown']:
        return MarkdownLoader(path, **kwargs)
    elif extension == '.json':
        return JSONLoader(path, **kwargs)
    else:
        return TextLoader(path, **kwargs)
