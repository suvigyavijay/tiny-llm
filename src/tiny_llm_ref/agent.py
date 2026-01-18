from typing import Callable, List


class Tool:
    def __init__(self, name: str, func: Callable, description: str):
        self.name = name
        self.func = func
        self.description = description


class Agent:
    """
    A simple ReAct agent that can use tools.
    """
    def __init__(self, model, tools: List[Tool]):
        self.model = model  # Function: prompt -> response string
        self.tools = {t.name: t for t in tools}

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
                    
                    if tool_name in self.tools:
                        result = self.tools[tool_name].func(tool_input)
                        history += f"Observation: {result}\n"
                    else:
                        history += f"Observation: Tool {tool_name} not found\n"
                except Exception:
                    history += "Observation: Invalid Action format\n"
                
        return "Max steps reached"
