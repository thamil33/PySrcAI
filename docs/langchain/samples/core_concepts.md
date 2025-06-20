
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
