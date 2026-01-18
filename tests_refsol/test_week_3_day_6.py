import pytest
from tiny_llm_ref.agent import Agent, Tool


def test_react_loop_with_tool():
    """Test ReAct loop correctly uses tools."""
    def calculator(expr: str) -> str:
        return str(eval(expr))
    
    tools = [Tool("Calculator", calculator, "Calculate math expressions")]
    
    step = 0
    def mock_model(prompt: str) -> str:
        nonlocal step
        step += 1
        if step == 1:
            return "Thought: I need to calculate 2+2\nAction: Calculator\nAction Input: 2+2"
        elif step == 2:
            if "Observation: 4" in prompt:
                return "Thought: I have the answer\nFinal Answer: 4"
            return "Error: No observation"
        return "Final Answer: Unknown"
    
    agent = Agent(mock_model, tools)
    result = agent.run("What is 2+2?")
    
    assert result == "4"


def test_react_loop_final_answer_direct():
    """Test ReAct loop when model provides final answer directly."""
    def mock_model(prompt: str) -> str:
        return "Final Answer: The sky is blue."
    
    agent = Agent(mock_model, [])
    result = agent.run("What color is the sky?")
    
    assert result == "The sky is blue."


def test_react_loop_max_steps():
    """Test ReAct loop respects max_steps."""
    call_count = 0
    def mock_model(prompt: str) -> str:
        nonlocal call_count
        call_count += 1
        return "Thought: Still thinking..."
    
    agent = Agent(mock_model, [])
    result = agent.run("Endless question", max_steps=3)
    
    assert result == "Max steps reached"
    assert call_count == 3
