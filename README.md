# Oepn — Unified AI Agent Framework

> Combining the best patterns from CrewAI, AutoGPT, LangGraph, PydanticAI, and Agno.

```
oepn/
├── core/          — BaseAgent, BaseModel, BaseTool, Message types
├── agent/         — Agent (tool-calling loop) + Team (multi-agent)
├── models/        — OpenAI, Anthropic, Gemini, LiteLLM providers
├── graph/         — StateGraph execution engine (from LangGraph)
├── blocks/        — Input/Output/LLM blocks (from AutoGPT)
├── capabilities/  — Middleware/hooks system (from PydanticAI)
├── memory/        — InMemory + extensible memory (from Agno)
├── knowledge/     — Knowledge base / RAG (from CrewAI)
├── workflow/      — DAG-based workflow engine (from Agno/CrewAI)
├── storage/       — Session persistence
├── orchestration/ — Execution engine
└── utils/         — Re-exports
```

## Quick Start

```python
from oepn import Agent

agent = Agent(
    name="Assistant",
    model="openai:gpt-4o",
    instructions="You are a helpful assistant.",
)

result = agent.run("What is the capital of France?")
print(result.text)
```

### With Tools

```python
from oepn import Agent, tool

@tool(description="Get the weather for a city")
def get_weather(city: str) -> str:
    return f"The weather in {city} is sunny, 22°C."

agent = Agent(
    name="WeatherBot",
    model="openai:gpt-4o",
    instructions="You help users check the weather.",
    tools=[get_weather],
)

result = agent.run("What's the weather in Tokyo?")
print(result.text)
```

### Multi-Agent Team

```python
from oepn import Agent, Team, TeamMode

writer = Agent(name="Writer", model="openai:gpt-4o", instructions="Write content.")
reviewer = Agent(name="Reviewer", model="openai:gpt-4o", instructions="Review and improve content.")

team = Team(
    name="ContentTeam",
    members=[writer, reviewer],
    mode=TeamMode.SEQUENTIAL,
)

result = team.run("Write a short blog post about AI agents.")
print(result.text)
```

### StateGraph

```python
from oepn import StateGraph

graph = StateGraph()
graph.add_node("process", lambda s: {**s, "result": f"Hello {s['name']}"})
graph.add_node("format", lambda s: {**s, "output": s["result"].upper()})
graph.add_edge("process", "format")

compiled = graph.compile()
result = compiled.invoke({"name": "Oepn"})
print(result)  # {"name": "Oepn", "result": "Hello Oepn", "output": "HELLO OEPN"}
```

### Streaming

```python
agent = Agent(name="Streamer", model="openai:gpt-4o")
for chunk in agent.run_stream("Tell me a short story"):
    print(chunk, end="", flush=True)
```

## Architecture

Oepn's architecture is inspired by the five frameworks:

| Concept | Inspired By |
|---------|-------------|
| Graph-based execution | LangGraph (StateGraph) |
| Block composability | AutoGPT (Block system) |
| Multi-agent orchestration | CrewAI (Team/Supervisor) |
| Capability middleware | PydanticAI (hooks chain) |
| Rich memory & knowledge | Agno (Memory/Knowledge) |
| Clean provider abstraction | PydanticAI + Agno |

## Installation

```bash
pip install -e ".[all]"
# or minimal:
pip install -e "."
```

Requires Python 3.11+.
