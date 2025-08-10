import pytest
from .utils import *
from tiny_llm_ref.batch import Request
from tiny_llm_ref.qwen2_week2 import Qwen2ModelWeek2
from mlx_lm import load


@pytest.mark.skipif(
    not qwen_2_05b_model_exists(), reason="Qwen2-0.5B-Instruct-MLX model not found"
)
class TestChunkedPrefill:
    @pytest.fixture(scope="class")
    def model_and_tokenizer(self):
        mlx_model, tokenizer = load("Qwen/Qwen2-0.5B-Instruct-MLX")
        model = Qwen2ModelWeek2(mlx_model)
        return model, tokenizer

    @pytest.mark.parametrize("prompt_length", [10, 50, 100])
    @pytest.mark.parametrize("prefill_max_step", [5, 10, 20])
    def test_chunked_prefill_various_lengths(self, model_and_tokenizer, prompt_length, prefill_max_step):
        model, tokenizer = model_and_tokenizer
        prompt = " ".join(["word"] * prompt_length)
        request = Request(model, tokenizer, prompt, prefill_max_step=prefill_max_step)
        
        while not request.is_prefill_done:
            request.try_prefill()
            
        assert request.is_prefill_done
        assert request.offset == len(tokenizer.encode(prompt, add_special_tokens=False))
        assert request.next_token is not None

    def test_chunked_prefill_single_step(self, model_and_tokenizer):
        model, tokenizer = model_and_tokenizer
        prompt = "This is a short prompt."
        request = Request(model, tokenizer, prompt, prefill_max_step=100)
        
        request.try_prefill()
        
        assert request.is_prefill_done
        assert request.offset == len(tokenizer.encode(prompt, add_special_tokens=False))
        assert request.next_token is not None

    def test_chunked_prefill_empty_prompt(self, model_and_tokenizer):
        model, tokenizer = model_and_tokenizer
        prompt = ""
        request = Request(model, tokenizer, prompt, prefill_max_step=10)
        
        request.try_prefill()
        
        assert request.is_prefill_done
        assert request.offset == 1
        assert request.next_token is not None
