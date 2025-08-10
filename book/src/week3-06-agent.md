# Week 3 Day 6: AI Agent & Tool Calling

AI agents extend LLMs beyond text generation to autonomous problem-solving by enabling them to use external tools, APIs, and perform multi-step reasoning. This creates systems that can take actions in the world, not just generate text.

## Agent Architecture

**ReAct Pattern** (Reasoning and Acting):
```
Thought: I need to find current weather data
Action: call_weather_api(location="San Francisco")
Observation: {"temperature": 72, "condition": "sunny"}
Thought: Based on the weather data, I can provide a recommendation
Action: respond_to_user(message="It's 72°F and sunny in SF...")
```

**Core Components**:
1. **Reasoning Engine**: Plans actions based on goals
2. **Tool Registry**: Available functions/APIs  
3. **Execution Engine**: Safely calls tools
4. **Memory System**: Tracks conversation and state

**Readings**

- [ReAct: Reasoning and Acting](https://arxiv.org/abs/2210.03629)
- [Toolformer](https://arxiv.org/abs/2302.04761)
- [LangChain Agents](https://docs.langchain.com/docs/components/agents/)

## Task 1: Tool System Foundation

Create the tool calling infrastructure:

```python
@dataclass
class Tool:
    name: str
    description: str
    parameters: dict        # JSON schema for parameters
    function: Callable      # Actual function to execute
    
class ToolRegistry:
    def __init__(self):
        """TODO: Initialize tool registry"""
        pass
    
    def register_tool(self, tool: Tool):
        """TODO: Add tool to registry"""
        pass
    
    def get_tool_descriptions(self) -> str:
        """TODO: Format tools for LLM consumption"""
        pass
    
    def execute_tool(self, tool_name: str, parameters: dict) -> dict:
        """
        TODO: Safely execute tool
        - Validate parameters against schema
        - Execute with error handling
        - Return structured result
        """
        pass
```

## Task 2: Agent Reasoning Engine

Implement the core agent logic:

```python
class ReActAgent:
    def __init__(self, llm_model, tool_registry: ToolRegistry):
        """TODO: Initialize ReAct agent"""
        pass
    
    def parse_llm_response(self, response: str) -> dict:
        """
        TODO: Parse LLM output for actions
        - Extract Thought/Action/Observation structure
        - Parse tool calls and parameters
        - Handle malformed responses gracefully
        """
        pass
    
    def step(self, user_message: str, conversation_history: list) -> dict:
        """
        TODO: Execute one reasoning step
        - Format prompt with tools and history
        - Generate LLM response
        - Parse and execute any tool calls
        - Return updated state
        """
        pass
```

## Task 3: Multi-Step Planning

Implement complex multi-step reasoning:

```python
class PlanningAgent(ReActAgent):
    def __init__(self, llm_model, tool_registry: ToolRegistry, max_steps: int = 10):
        """TODO: Initialize planning agent"""
        super().__init__(llm_model, tool_registry)
        pass
    
    def solve(self, problem: str) -> dict:
        """
        TODO: Solve multi-step problems
        - Create initial plan
        - Execute steps iteratively
        - Adapt plan based on observations
        - Track progress toward goal
        """
        pass
```

## Task 4: Built-in Tool Library

Create useful tools for agents:

```python
# Web search tool
def web_search(query: str, num_results: int = 5) -> dict:
    """TODO: Implement web search capability"""
    pass

# Calculator tool  
def calculator(expression: str) -> dict:
    """TODO: Safe mathematical computation"""
    pass

# File system tools
def read_file(filepath: str) -> dict:
    """TODO: Read file contents safely"""
    pass

def write_file(filepath: str, content: str) -> dict:
    """TODO: Write file with validation"""
    pass

# API calling tool
def api_call(url: str, method: str = "GET", data: dict = None) -> dict:
    """TODO: Make HTTP API calls"""
    pass
```

## Task 5: Memory and State Management

Implement persistent agent memory:

```python
class AgentMemory:
    def __init__(self, vector_store):
        """TODO: Initialize agent memory system"""
        pass
    
    def store_interaction(self, interaction: dict):
        """TODO: Store conversation for later retrieval"""
        pass
    
    def retrieve_relevant(self, query: str, top_k: int = 5) -> list[dict]:
        """TODO: Find relevant past interactions"""
        pass
```

## Task 6: Safety and Sandboxing

Implement safe tool execution:

```python
class SafeExecutor:
    def __init__(self, allowed_domains: list[str], timeout: int = 30):
        """TODO: Initialize safe execution environment"""
        pass
    
    def execute_with_limits(self, tool_call: dict) -> dict:
        """
        TODO: Execute tools with safety constraints
        - Timeout protection
        - Resource limits
        - Domain restrictions
        - Input validation
        """
        pass
```

{{#include copyright.md}}
