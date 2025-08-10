# Week 3 Day 6: AI Agent / Tool Calling

AI agents are systems that can reason, plan, and take actions to achieve a goal. A key capability of many AI agents is the ability to use external tools and services. This allows them to extend their capabilities beyond their built-in knowledge and skills, enabling them to perform a wide range of tasks, from looking up information on the web to controlling a smart home.

In this chapter, you will implement a simple AI agent that can use a set of predefined tools. This will give you a hands-on understanding of the core concepts behind tool-calling agents.

[📚 Reading: How LLM-powered agents can learn to use tools](https://www.youtube.com/watch?v=y_pC0o4zH_w)

## Task 1: Implement an AI Agent

Your task is to implement an AI agent that can select and execute tools based on a user's query.

```
src/tiny_llm/agent.py
```

The implementation will consist of two main components:
- **`Tool`**: A class that represents an external tool that the agent can use. Each tool will have a name, a description, and a function to execute.
- **`Agent`**: The main class that orchestrates the agent's behavior. It will take a user's query, use an LLM to select the most appropriate tool, and then execute the tool.

You can run the following tests to verify your implementation:

```
pdm run test --week 3 --day 6
```

This simplified agent will serve as a starting point for building more sophisticated agents. In a real-world application, you would need to handle more complex scenarios, such as parsing arguments for tools, managing conversational state, and handling errors.

{{#include copyright.md}}
