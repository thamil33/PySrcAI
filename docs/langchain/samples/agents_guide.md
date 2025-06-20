
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
