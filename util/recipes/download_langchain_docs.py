"""
Download LangChain Documentation for RAG Setup

This script downloads LangChain documentation from various sources to create
a comprehensive knowledge base for RAG demonstrations.

Usage:
    python recipes/download_langchain_docs.py [--output-dir docs/langchain]
"""

import argparse
import requests
import zipfile
import shutil
from pathlib import Path
from typing import List
import tempfile


def download_file(url: str, output_path: Path) -> bool:
    """Download a file from URL to output path."""
    try:
        print(f"📥 Downloading: {url}")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"✅ Downloaded: {output_path}")
        return True
        
    except Exception as e:
        print(f"❌ Error downloading {url}: {e}")
        return False


def download_github_docs(repo: str, output_dir: Path, paths: List[str] = None) -> bool:
    """Download documentation from a GitHub repository."""
    try:
        # Download as ZIP
        zip_url = f"https://github.com/{repo}/archive/refs/heads/master.zip"
        
        with tempfile.TemporaryDirectory() as temp_dir:
            zip_path = Path(temp_dir) / "repo.zip"
            
            if not download_file(zip_url, zip_path):
                return False
            
            # Extract ZIP
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            
            # Find extracted directory
            extracted_dirs = [d for d in Path(temp_dir).iterdir() if d.is_dir()]
            if not extracted_dirs:
                print("❌ No directories found in ZIP")
                return False
            
            repo_dir = extracted_dirs[0]
            
            # Copy documentation files
            if paths is None:
                paths = ["docs", "README.md"]
            
            for path in paths:
                source_path = repo_dir / path
                if source_path.exists():
                    if source_path.is_file():
                        target_path = output_dir / source_path.name
                        target_path.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(source_path, target_path)
                        print(f"📄 Copied: {source_path.name}")
                    else:
                        target_path = output_dir / source_path.name
                        if target_path.exists():
                            shutil.rmtree(target_path)
                        shutil.copytree(source_path, target_path)
                        print(f"📁 Copied directory: {source_path.name}")
                else:
                    print(f"⚠️  Path not found: {path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error downloading from {repo}: {e}")
        return False


def create_sample_docs(output_dir: Path):
    """Create comprehensive sample documentation if downloads fail."""
    print("📝 Creating sample documentation...")
    
    # Ensure the output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # LangChain Core Concepts
    (output_dir / "core_concepts.md").write_text("""
# LangChain Core Concepts

## Overview
LangChain is a framework for developing applications powered by language models.

## Key Components

### 1. LLMs and Chat Models
- **LLMs**: Text completion models
- **Chat Models**: Conversational models with message-based APIs

### 2. Prompt Templates
Templates for structuring inputs to language models:

```python
from langchain.prompts import PromptTemplate

template = "What is a good name for a company that makes {product}?"
prompt = PromptTemplate.from_template(template)
```

### 3. Chains
Sequences of calls to LLMs or other utilities:

```python
from langchain.chains import LLMChain

chain = LLMChain(llm=llm, prompt=prompt)
result = chain.run(product="colorful socks")
```

### 4. Memory
Storing and retrieving conversation history:
- ConversationBufferMemory
- ConversationSummaryMemory
- ConversationBufferWindowMemory
""")
    
    # RAG Documentation
    (output_dir / "rag_tutorial.md").write_text("""
# Retrieval-Augmented Generation (RAG) Tutorial

## What is RAG?
RAG combines retrieval of relevant documents with text generation to provide accurate, context-aware responses.

## RAG Pipeline Steps

### 1. Document Loading
```python
from langchain.document_loaders import TextLoader

loader = TextLoader("document.txt")
documents = loader.load()
```

### 2. Text Splitting
```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
splits = text_splitter.split_documents(documents)
```

### 3. Embeddings and Vector Store
```python
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(splits, embeddings)
```

### 4. Retrieval and Generation
```python
from langchain.chains import RetrievalQA

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever()
)

result = qa_chain.run("What is the main topic of the document?")
```

## Best Practices

1. **Chunk Size**: Balance between context and relevance
2. **Overlap**: Maintain continuity between chunks
3. **Retrieval Strategy**: Use similarity search with filtering
4. **Context Window**: Respect model token limits
5. **Evaluation**: Measure retrieval accuracy and generation quality
""")
    
    # Vector Stores Guide
    (output_dir / "vector_stores.md").write_text("""
# Vector Stores in LangChain

## Overview
Vector stores enable efficient storage and retrieval of document embeddings.

## Supported Vector Stores

### Chroma
Local, persistent vector database:

```python
from langchain.vectorstores import Chroma

vectorstore = Chroma(
    collection_name="my_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)
```

### Pinecone
Cloud-based vector database:

```python
from langchain.vectorstores import Pinecone
import pinecone

pinecone.init(api_key="your-api-key", environment="us-west1-gcp")
vectorstore = Pinecone.from_documents(documents, embeddings, index_name="my-index")
```

### FAISS
Facebook AI Similarity Search:

```python
from langchain.vectorstores import FAISS

vectorstore = FAISS.from_documents(documents, embeddings)
vectorstore.save_local("./faiss_index")
```

## Vector Store Operations

### Adding Documents
```python
# Add new documents
vectorstore.add_documents(new_documents)

# Add texts directly
vectorstore.add_texts(texts, metadatas=metadatas)
```

### Searching
```python
# Similarity search
docs = vectorstore.similarity_search("query", k=5)

# Similarity search with scores
docs_with_scores = vectorstore.similarity_search_with_score("query", k=5)

# Maximum marginal relevance search
docs = vectorstore.max_marginal_relevance_search("query", k=5)
```

### Filtering
```python
# Search with metadata filters
docs = vectorstore.similarity_search(
    "query",
    filter={"source": "document.pdf"}
)
```
""")
    
    # Embeddings Guide
    (output_dir / "embeddings_guide.md").write_text("""
# Embeddings in LangChain

## What are Embeddings?
Embeddings are numerical representations of text that capture semantic meaning.

## Embedding Providers

### OpenAI Embeddings
```python
from langchain.embeddings import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(
    model="text-embedding-ada-002",
    openai_api_key="your-api-key"
)
```

### HuggingFace Embeddings
```python
from langchain.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
```

### HuggingFace Hub
```python
from langchain.embeddings import HuggingFaceHubEmbeddings

embeddings = HuggingFaceHubEmbeddings(
    repo_id="sentence-transformers/all-MiniLM-L6-v2",
    huggingfacehub_api_token="your-token"
)
```

## Usage Examples

### Embedding Documents
```python
# Embed multiple documents
doc_embeddings = embeddings.embed_documents([
    "This is a document",
    "This is another document"
])
```

### Embedding Queries
```python
# Embed a query
query_embedding = embeddings.embed_query("What is machine learning?")
```

## Best Practices

1. **Model Selection**: Choose models appropriate for your domain
2. **Consistency**: Use the same embedding model for indexing and querying
3. **Normalization**: Some models benefit from L2 normalization
4. **Batch Processing**: Process multiple texts together for efficiency
5. **Caching**: Cache embeddings to avoid recomputation
""")
    
    # Agents Documentation
    (output_dir / "agents_guide.md").write_text("""
# LangChain Agents

## Overview
Agents use language models to determine which actions to take and in what order.

## Agent Types

### Zero-shot ReAct
Determines actions based on tool descriptions:

```python
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)
```

### Conversational ReAct
Designed for conversational scenarios:

```python
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    memory=memory,
    verbose=True
)
```

## Tools

### Built-in Tools
- Search tools (Google, DuckDuckGo)
- Math tools (Calculator, WolframAlpha)
- Code execution tools (Python REPL)

### Custom Tools
```python
from langchain.tools import BaseTool

class CustomTool(BaseTool):
    name = "custom_tool"
    description = "Useful for custom operations"
    
    def _run(self, query: str) -> str:
        # Tool logic here
        return f"Result for: {query}"
```

## Agent Execution

### Running Agents
```python
# Single execution
result = agent.run("What's the weather like today?")

# With intermediate steps
result = agent.run(
    "What's the weather like today?",
    return_intermediate_steps=True
)
```

### Agent Callbacks
```python
from langchain.callbacks import StdOutCallbackHandler

agent.run(
    "Question here",
    callbacks=[StdOutCallbackHandler()]
)
```
""")
    
    print(f"✅ Created sample documentation in {output_dir}")


def main():
    """Main function to download LangChain documentation."""
    parser = argparse.ArgumentParser(description="Download LangChain documentation")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/langchain"),
        help="Output directory for documentation"
    )
    
    args = parser.parse_args()
    output_dir = args.output_dir
    
    print("📚 LangChain Documentation Downloader")
    print("=" * 40)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # List of documentation sources to download
    doc_sources = [
        {
            "name": "LangChain Core",
            "repo": "langchain-ai/langchain",
            "paths": ["docs/docs", "README.md"]
        },
        {
            "name": "LangChain Community",
            "repo": "langchain-ai/langchain-community", 
            "paths": ["docs", "README.md"]
        }
    ]
    
    success_count = 0
    
    # Try to download from GitHub repos
    for source in doc_sources:
        print(f"\n📦 Downloading {source['name']}...")
        source_dir = output_dir / source['name'].lower().replace(' ', '_')
        
        if download_github_docs(source['repo'], source_dir, source['paths']):
            success_count += 1
        else:
            print(f"⚠️  Failed to download {source['name']}")
    
    # Create sample docs regardless
    sample_dir = output_dir / "samples"
    create_sample_docs(sample_dir)
    
    # Summary
    print(f"\n📊 Summary:")
    print(f"   Downloaded: {success_count}/{len(doc_sources)} repositories")
    print(f"   Sample docs: ✅ Created")
    print(f"   Output directory: {output_dir}")
    
    # Verification instructions
    print(f"\n✅ Setup complete!")
    print(f"   Documents available in: {output_dir}")
    print(f"   To test RAG setup: python recipes/default_rag_setup.py")
    
    # Count total files
    total_files = len(list(output_dir.rglob("*.md"))) + len(list(output_dir.rglob("*.txt")))
    print(f"   Total documentation files: {total_files}")


if __name__ == "__main__":
    main()
