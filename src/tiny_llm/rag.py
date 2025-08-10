from typing import List
from tiny_llm.qwen2_week2 import Qwen2_2w

class KnowledgeBase:
    def __init__(self, documents: List[str]):
        # TODO: implement
        pass

    def retrieve(self, query: str) -> List[str]:
        # TODO: implement
        pass

class RagPipeline:
    def __init__(self, model: Qwen2_2w, tokenizer, knowledge_base: KnowledgeBase):
        # TODO: implement
        pass

    def generate(self, query: str, max_tokens=100) -> str:
        # TODO: implement
        pass
