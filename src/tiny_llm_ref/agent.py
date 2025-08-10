from typing import List, Dict, Callable

class Tool:
    def __init__(self, name: str, description: str, func: Callable):
        self.name = name
        self.description = description
        self.func = func

class Agent:
    def __init__(self, model, tools: List[Tool]):
        self.model = model
        self.tools = {tool.name: tool for tool in tools}

    def run(self, query: str) -> str:
        # This is a simplified implementation of an agent.
        # A real implementation would involve more sophisticated prompt engineering,
        # tool selection, and execution logic.
        if not self.tools:
            return "I don't have any tools."

        prompt = "You have the following tools available:\n"
        for tool in self.tools.values():
            prompt += f"- {tool.name}: {tool.description}\n"
        prompt += f"\nUser query: {query}\n"
        prompt += "Which tool should I use? Respond with the tool name only."

        tool_name = self.model.generate(prompt).strip()

        if tool_name in self.tools:
            # For simplicity, we assume the tool doesn't require any arguments.
            # A real implementation would need to parse the arguments from the
            # user's query.
            try:
                result = self.tools[tool_name].func()
                return f"Executed tool '{tool_name}' and got the result: {result}"
            except Exception as e:
                return f"Error executing tool '{tool_name}': {e}"
        else:
            return "I don't have a tool for that."
