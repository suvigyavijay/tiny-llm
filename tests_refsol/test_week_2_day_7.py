import pytest
from .utils import *
from tiny_llm_ref.batch import Request
from tiny_llm_ref.qwen2_week2 import Qwen2ModelWeek2
from mlx_lm import load


@pytest.mark.skipif(
    not qwen_2_05b_model_exists(), reason="Qwen2-0.5B-Instruct-MLX model not found"
)
class TestTask1:
    def test_chunked_prefill(self):
        mlx_model, tokenizer = load("Qwen/Qwen2-0.5B-Instruct-MLX")
        model = Qwen2ModelWeek2(mlx_model)
        prompt = "This is a long prompt that needs to be chunked for prefilling."
        request = Request(model, tokenizer, prompt, prefill_max_step=10)
        
        while not request.is_prefill_done:
            request.try_prefill()
            
        assert request.is_prefill_done
        assert request.offset == len(tokenizer.encode(prompt, add_special_tokens=False))
        assert request.next_token is not None

    def test_chunked_prefill_multiple_steps(self):
        mlx_model, tokenizer = load("Qwen/Qwen2-0.5B-Instruct-MLX")
        model = Qwen2ModelWeek2(mlx_model)
        prompt = "This is another long prompt that will definitely require multiple prefill steps."
        request = Request(model, tokenizer, prompt, prefill_max_step=5)
        
        prefill_steps = 0
        while not request.is_prefill_done:
            request.try_prefill()
            prefill_steps += 1
            
        assert prefill_steps > 1
        assert request.is_prefill_done
        assert request.offset == len(tokenizer.encode(prompt, add_special_tokens=False))
        assert request.next_token is not None
