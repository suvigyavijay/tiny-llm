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

## Task 1: ReAct Loop

```
src/tiny_llm/agent.py
```

Implement a simple `Agent` class that:
- Takes a system prompt defining available tools.
- Runs the LLM.
- Parses output for "Action:".
- Executes the tool.
- Feeds output back.

### Implementation Pattern

```python
while steps < max_steps:
    response = model(history)
    if "Final Answer" in response:
        return response
    
    if "Action" in response:
        # 1. Parse tool name and input
        # 2. Execute tool function
        # 3. Append "Observation: {result}" to history
    else:
        # Model is just thinking
        pass
```

## Testing

```bash
pdm run test-refsol tests_refsol/test_week_3_day_6.py
```

{{#include copyright.md}}
