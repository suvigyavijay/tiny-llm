from typing import List, Callable
from tiny_llm.qwen2_week2 import Qwen2_2w

class Tool:
    def __init__(self, name: str, func: Callable, description: str):
        # TODO: implement
        pass

class Agent:
    def __init__(self, model: Qwen2_2w, tokenizer, tools: List[Tool]):
        # TODO: implement
        pass

    def run(self, query: str, max_tokens=100) -> str:
        # TODO: implement
        pass
