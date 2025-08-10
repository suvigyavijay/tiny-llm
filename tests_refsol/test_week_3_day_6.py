import pytest
from tiny_llm_ref.agent import Tool, Agent

class MockModel:
    def __init__(self, response):
        self.response = response
        
    def generate(self, prompt):
        return self.response

def get_weather():
    return "The weather is sunny."

def get_time():
    return "The time is 3:00 PM."

class TestAgent:
    def test_agent_tool_selection(self):
        tools = [
            Tool("weather", "Get the current weather", get_weather),
            Tool("time", "Get the current time", get_time)
        ]
        
        model = MockModel("weather")
        agent = Agent(model, tools)
        
        response = agent.run("What's the weather like?")
        assert "Executed tool 'weather'" in response
        assert "The weather is sunny" in response

    @pytest.mark.parametrize("prompt, expected_tool", [
        ("What's the weather like?", "weather"),
        ("what time is it", "time"),
        ("current time please", "time"),
    ])
    def test_agent_tool_selection_parametrized(self, prompt, expected_tool):
        tools = [
            Tool("weather", "Get the current weather", get_weather),
            Tool("time", "Get the current time", get_time)
        ]
        
        model = MockModel(expected_tool)
        agent = Agent(model, tools)
        
        response = agent.run(prompt)
        assert f"Executed tool '{expected_tool}'" in response

    def test_agent_no_tools(self):
        model = MockModel("weather")
        agent = Agent(model, [])
        response = agent.run("What's the weather like?")
        assert "I don't have any tools" in response

    def test_tool_execution_error(self):
        def failing_tool():
            raise ValueError("Tool failed")

        tools = [
            Tool("failing", "A tool that always fails", failing_tool)
        ]
        
        model = MockModel("failing")
        agent = Agent(model, tools)
        
        response = agent.run("Run the failing tool")
        assert "Error executing tool 'failing': Tool failed" in response

    def test_agent_unknown_tool(self):
        tools = [
            Tool("weather", "Get the current weather", get_weather),
            Tool("time", "Get the current time", get_time)
        ]
        
        model = MockModel("calendar")
        agent = Agent(model, tools)
        
        response = agent.run("Show me my calendar.")
        assert "I don't have a tool for that" in response

    def test_agent_complex_prompt(self):
        tools = [
            Tool("weather", "Get the current weather", get_weather),
            Tool("time", "Get the current time", get_time)
        ]
        
        # In a real scenario, the model might choose 'time' even if it's not a perfect match.
        model = MockModel("time")
        agent = Agent(model, tools)
        
        response = agent.run("What will the weather be like tomorrow at 3pm?")
        assert "Executed tool 'time'" in response
