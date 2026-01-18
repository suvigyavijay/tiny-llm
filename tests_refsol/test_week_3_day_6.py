import pytest
from .tiny_llm_base import *
from .utils import *


def test_tool_registry_register_and_get():
    """Test tool registration and retrieval."""
    registry = ToolRegistry()
    tool = Tool("Calculator", lambda x: x, "Math operations")
    
    registry.register(tool)
    retrieved = registry.get("Calculator")
    
    assert retrieved == tool


def test_tool_registry_missing_tool():
    """Test error when tool not found."""
    registry = ToolRegistry()
    
    with pytest.raises(ValueError):
        registry.get("NonexistentTool")


def test_agent_executes_tool():
    """Test agent can execute a tool and return result."""
    def calculator(expr):
        return str(eval(expr))
    
    registry = ToolRegistry()
    registry.register(Tool("Calculator", calculator, "Math"))
    
    step = 0
    def mock_model(prompt):
        nonlocal step
        step += 1
        if step == 1:
            return "Thought: I need to calculate.\nAction: Calculator\nAction Input: 2+2"
        return "Final Answer: 4"
        
    agent = Agent(mock_model, registry)
    result = agent.run("What is 2+2?")
    
    assert result == "4"


def test_agent_max_steps():
    """Test agent stops after max steps."""
    registry = ToolRegistry()
    
    def infinite_loop_model(prompt):
        return "Thought: Still thinking..."
        
    agent = Agent(infinite_loop_model, registry)
    result = agent.run("Question", max_steps=3)
    
    assert result == "Max steps reached"
