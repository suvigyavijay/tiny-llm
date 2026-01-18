# Week 3 Day 6: AI Agents & Tool Calling

In this chapter, we will implement an **AI Agent** capable of using tools. We will build the **ReAct** (Reasoning + Acting) loop that allows the model to "think", call external functions, and observe results.

Agents use LLMs as reasoning engines to interact with the world. The core loop is **ReAct** (Reasoning + Acting).

**📚 Readings**

- [LangChain Agents - LangChain Docs](https://python.langchain.com/docs/modules/agents/)
- [ReAct: Reasoning and Acting with LLMs - Prompt Engineering Guide](https://www.promptingguide.ai/techniques/react)

## The Loop

1.  **Thought**: LLM analyzes the request.
2.  **Action**: LLM emits a special token or JSON blob to call a tool (e.g., `Calculator(2+2)`).
3.  **Observation**: The environment executes the tool and returns `4`.
4.  **Answer**: LLM sees the observation and produces the final answer.

## Task 1: Tool Registry

```
src/tiny_llm/agent.py
```

Implement a `ToolRegistry` to manage available tools.

```python
class ToolRegistry:
    def register(self, tool):
        pass
        
    def get_tool(self, name):
        pass
```

## Task 2: ReAct Loop

Implement the `Agent` class that uses the registry.

```python
class Agent:
    def run(self, query):
        # 1. Thought
        # 2. Action (from registry)
        # 3. Observation
        # 4. Answer
        pass
```

## Testing

```bash
pdm run test-refsol tests_refsol/test_week_3_day_6.py
```

{{#include copyright.md}}
