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
        self.model = model
        self.tools = {t.name: t for t in tools}

    def run(self, query: str, max_steps: int = 5) -> str:
        """
        Run the ReAct loop.
        
        Args:
            query: The user's question.
            max_steps: Maximum number of thought-action-observation cycles.
            
        Returns:
            The final answer string.
        """
        pass
