from typing import Callable, List, Dict


class Tool:
    def __init__(self, name: str, func: Callable, description: str):
        self.name = name
        self.func = func
        self.description = description


class ToolRegistry:
    def __init__(self):
        self.tools: Dict[str, Tool] = {}

    def register(self, tool: Tool):
        self.tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        if name not in self.tools:
            raise ValueError(f"Tool {name} not found")
        return self.tools[name]


class Agent:
    def __init__(self, model, registry: ToolRegistry):
        self.model = model
        self.registry = registry

    def run(self, query: str, max_steps: int = 5) -> str:
        history = f"Question: {query}\n"
        
        for _ in range(max_steps):
            response = self.model(history)
            history += response + "\n"
            
            if "Final Answer:" in response:
                return response.split("Final Answer:")[1].strip()
            
            if "Action:" in response:
                try:
                    lines = response.split('\n')
                    action_line = [l for l in lines if l.startswith("Action:")][0]
                    tool_name = action_line.split("Action:")[1].strip()
                    
                    input_line = [l for l in lines if l.startswith("Action Input:")][0]
                    tool_input = input_line.split("Action Input:")[1].strip()
                    
                    tool = self.registry.get(tool_name)
                    result = tool.func(tool_input)
                    history += f"Observation: {result}\n"
                except Exception:
                    history += "Observation: Error executing tool\n"
                
        return "Max steps reached"
