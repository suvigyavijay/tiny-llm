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

    def test_agent_unknown_tool(self):
        tools = [
            Tool("weather", "Get the current weather", get_weather),
            Tool("time", "Get the current time", get_time)
        ]
        
        model = MockModel("calendar")
        agent = Agent(model, tools)
        
        response = agent.run("Show me my calendar.")
        assert "I don't have a tool for that" in response
