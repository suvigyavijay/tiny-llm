from typing import List, Generator, Tuple
import mlx.core as mx
from tiny_llm.kv_cache import TinyKvCache
from tiny_llm.qwen2_week2 import Qwen2_2w

class Request:
    def __init__(
        self,
        prompt: str,
        tokenizer,
        prefill_max_step: int = 32,
    ):
        # TODO: implement
        pass

    def try_prefill(self, model: Qwen2_2w) -> Generator[None, None, None]:
        # TODO: implement
        pass

def batch_generate(
    prompts: List[str],
    model: Qwen2_2w,
    tokenizer,
    max_tokens: int = 100,
    prefill_max_step: int = 32,
    max_active_requests: int = 10,
) -> Generator[Tuple[int, str], None, None]:
    # TODO: implement
    pass
