from typing import Callable, List


class Tool:
    def __init__(self, name: str, func: Callable, description: str):
        self.name = name
        self.func = func
        self.description = description


class ToolRegistry:
    """
    Manages a collection of tools.
    """
    def __init__(self):
        self.tools = {}

    def register(self, tool: Tool):
        pass

    def get(self, name: str) -> Tool:
        pass


class Agent:
    """
    A simple ReAct agent that can use tools.
    """
    def __init__(self, model, registry: ToolRegistry):
        self.model = model
        self.registry = registry

    def run(self, query: str, max_steps: int = 5) -> str:
        """
        Run the ReAct loop.
        """
        pass
