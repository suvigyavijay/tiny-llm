import pytest
import mlx.core as mx
from .tiny_llm_base import *
from .utils import *
from tiny_llm_ref.batch import batch_generate, Request
from tiny_llm_ref.qwen2_week2 import Qwen2ModelWeek2
from mlx_lm import load
import time


def attention_helper(
    stream: mx.Stream, H_q, H, L, E, S, BATCH, use_flash_attention: bool = False
):
    precision = mx.float32
    with mx.stream(stream):
        q_shape = (BATCH, H_q, L, E)
        kv_shape = (BATCH, H, S, E)
        scale = 0.8
        for _ in range(100):
            query = mx.random.uniform(shape=q_shape, dtype=precision)
            key = mx.random.uniform(shape=kv_shape, dtype=precision)
            value = mx.random.uniform(shape=kv_shape, dtype=precision)
            mask = mx.random.uniform(shape=(BATCH, 1, L, S), dtype=precision)

            reference_output_1 = mx.fast.scaled_dot_product_attention(
                q=query,
                k=key,
                v=value,
                scale=scale,
                mask=mask,
            )
            reference_output_2 = mx.fast.scaled_dot_product_attention(
                q=query,
                k=key,
                v=value,
                scale=scale,
            )
            if use_flash_attention:
                user_output_1 = flash_attention(
                    query,
                    key,
                    value,
                    scale=scale,
                    mask=mask,
                )
                user_output_2 = flash_attention(
                    query,
                    key,
                    value,
                    scale=scale,
                )
            else:
                user_output_1 = scaled_dot_product_attention_grouped(
                    query,
                    key,
                    value,
                    scale=scale,
                    mask=mask,
                )
                user_output_2 = scaled_dot_product_attention_grouped(
                    query,
                    key,
                    value,
                    scale=scale,
                )
            mx.eval(user_output_1)
            mx.eval(user_output_2)
            assert_allclose(
                user_output_2,
                reference_output_2,
                precision=mx.float16,
                message="no mask",
            )
            assert_allclose(
                user_output_1,
                reference_output_1,
                precision=mx.float16,
                message="with mask",
            )


def test_flash_attention_with_mask_cpu_small():
    attention_helper(mx.cpu, 6, 3, 2, 5, 3, 1, use_flash_attention=True)


def test_flash_attention_with_mask_cpu():
    attention_helper(mx.cpu, 18, 6, 7, 5, 3, 10, use_flash_attention=True)


def test_flash_attention_with_mask_cpu_large():
    attention_helper(mx.cpu, 28, 4, 16, 128, 16, 3, use_flash_attention=True)


def test_flash_attention_with_mask_gpu_extra_small():
    attention_helper(mx.gpu, 1, 1, 5, 7, 4, 1, use_flash_attention=True)


def test_flash_attention_with_mask_gpu_small():
    attention_helper(mx.gpu, 6, 3, 2, 5, 3, 1, use_flash_attention=True)


def test_flash_attention_with_mask_gpu():
    attention_helper(mx.gpu, 18, 6, 7, 5, 3, 10, use_flash_attention=True)


def test_flash_attention_with_mask_gpu_large():
    attention_helper(mx.gpu, 28, 4, 16, 128, 16, 3, use_flash_attention=True)


def test_attention_with_mask_cpu_small():
    attention_helper(mx.cpu, 6, 3, 2, 5, 3, 1, use_flash_attention=False)


def test_attention_with_mask_cpu():
    attention_helper(mx.cpu, 18, 6, 7, 5, 3, 10, use_flash_attention=False)


def test_attention_with_mask_cpu_large():
    attention_helper(mx.cpu, 28, 4, 16, 128, 16, 3, use_flash_attention=False)


def test_attention_with_mask_gpu_extra_small():
    attention_helper(mx.gpu, 1, 1, 5, 7, 4, 1, use_flash_attention=False)


def test_attention_with_mask_gpu_small():
    attention_helper(mx.gpu, 6, 3, 2, 5, 3, 1, use_flash_attention=False)


def test_attention_with_mask_gpu():
    attention_helper(mx.gpu, 18, 6, 7, 5, 3, 10, use_flash_attention=False)


def test_attention_with_mask_gpu_large():
    attention_helper(mx.gpu, 28, 4, 16, 128, 16, 3, use_flash_attention=False)


@pytest.mark.skipif(
    not qwen_2_05b_model_exists(), reason="Qwen2-0.5B-Instruct-MLX model not found"
)
class TestBatchGenerate:
    @pytest.fixture(scope="class")
    def model_and_tokenizer(self):
        mlx_model, tokenizer = load("Qwen/Qwen2-0.5B-Instruct-MLX")
        model = Qwen2ModelWeek2(mlx_model)
        return model, tokenizer

    @pytest.mark.parametrize("num_prompts", [1, 2, 5])
    def test_batch_generate_num_prompts(self, model_and_tokenizer, num_prompts):
        model, tokenizer = model_and_tokenizer
        prompts = ["hello"] * num_prompts
        result = batch_generate(model, tokenizer, prompts, max_seq_len=10)
        assert len(result) == num_prompts
        for i in range(num_prompts):
            assert "hello" in result[i][1]

    @pytest.mark.parametrize("prompt_length", [5, 20, 50])
    def test_batch_generate_prompt_length(self, model_and_tokenizer, prompt_length):
        model, tokenizer = model_and_tokenizer
        prompt = " ".join(["word"] * prompt_length)
        result = batch_generate(model, tokenizer, [prompt], max_seq_len=prompt_length + 10)
        assert len(result) == 1
        assert prompt in result[0][1]

    def test_batch_generate_empty_prompt(self, model_and_tokenizer):
        model, tokenizer = model_and_tokenizer
        prompts = [""]
        result = batch_generate(model, tokenizer, prompts, max_seq_len=10)
        assert len(result) == 1
        assert result[0][1] is not None

    def test_batch_generate_max_requests(self, model_and_tokenizer):
        model, tokenizer = model_and_tokenizer
        prompts = ["hello"] * 10
        with pytest.raises(ValueError):
            batch_generate(model, tokenizer, prompts, max_seq_len=10, max_active_requests=5)

    def test_request_prefill(self, model_and_tokenizer):
        model, tokenizer = model_and_tokenizer
        prompt = "this is a test prompt"
        request = Request(model, tokenizer, prompt, prefill_max_step=5)
        
        request.try_prefill()
        assert not request.is_finished()
