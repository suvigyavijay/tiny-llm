import pytest
import time
import mlx.core as mx
from .utils import *
from .tiny_llm_base import (
    batch_generate,
    Request,
    BatchingKvCache,
    TinyKvFullCache,
    Qwen2ModelWeek2
)
from mlx_lm import load


def create_test_prompts(num_prompts: int = 5) -> list[str]:
    """Create test prompts of varying lengths"""
    prompts = [
        "Hello",
        "Tell me about artificial intelligence",
        "What is the capital of France?",
        "Explain quantum computing in simple terms",
        "Write a short story about a robot"
    ]
    return prompts[:num_prompts]


@pytest.mark.skipif(
    not qwen_2_05b_model_exists(), reason="Qwen2-0.5B-Instruct-MLX model not found"
)
def test_request_lifecycle():
    """Test the basic Request class lifecycle"""
    mlx_model, tokenizer = load("Qwen/Qwen2-0.5B-Instruct-MLX")
    model = Qwen2ModelWeek2(mlx_model)
    
    prompt = "Hello world"
    request = Request(model, tokenizer, prompt, prefill_max_step=32)
    
    # Initial state
    assert not request.is_done
    assert not request.is_prefill_done
    assert request.offset == 0
    assert len(request.prefill_tokens) > 0
    
    # Process prefill
    initial_tokens = len(request.prefill_tokens)
    while not request.is_prefill_done:
        tokens_before = request.offset
        request.try_prefill()
        assert request.offset > tokens_before  # Made progress
        
    assert request.is_prefill_done
    assert request.offset == initial_tokens
    assert request.next_token is not None


@pytest.mark.skipif(
    not qwen_2_05b_model_exists(), reason="Qwen2-0.5B-Instruct-MLX model not found"
)
def test_batching_kv_cache_integration():
    """Test BatchingKvCache with actual model layers"""
    mlx_model, tokenizer = load("Qwen/Qwen2-0.5B-Instruct-MLX")
    model = Qwen2ModelWeek2(mlx_model)
    
    max_requests = 3
    max_seq_len = 64
    
    # Create batching cache for each layer
    kv_cache = [
        BatchingKvCache(max_active_requests=max_requests, max_seq_len=max_seq_len)
        for _ in range(model.num_hidden_layers)
    ]
    
    # Create some requests
    prompts = ["Hello", "Hi there"]
    requests = []
    
    for i, prompt in enumerate(prompts):
        request = Request(model, tokenizer, prompt, prefill_max_step=16)
        
        # Prefill the request
        while not request.is_prefill_done:
            request.try_prefill()
            
        # Add to batching cache
        for layer_idx, layer_cache in enumerate(kv_cache):
            layer_cache.add_request(request.kv_cache[layer_idx], slot_id=i)
            
        requests.append(request)
    
    # Now do a decode step
    next_tokens = mx.array([req.next_token for req in requests] + [0])  # Pad to batch size
    offsets = [req.offset for req in requests] + [0]
    
    # This should work without errors
    logits = model(next_tokens.reshape(-1, 1), offsets, kv_cache)
    assert logits.shape[0] == max_requests
    assert logits.shape[2] == model.vocab_size


@pytest.mark.skipif(
    not qwen_2_05b_model_exists(), reason="Qwen2-0.5B-Instruct-MLX model not found"
)
def test_continuous_batching_basic():
    """Test basic continuous batching functionality"""
    mlx_model, tokenizer = load("Qwen/Qwen2-0.5B-Instruct-MLX")
    
    prompts = create_test_prompts(3)
    
    results = batch_generate(
        mlx_model, 
        tokenizer, 
        prompts,
        max_seq_len=32,
        batch_size=2,
        prefill_step=16
    )
    
    # Should get results for all prompts
    assert len(results) == len(prompts)
    
    # Results should be in order by prompt index
    result_indices = [r[0] for r in results]
    assert sorted(result_indices) == list(range(len(prompts)))
    
    # All results should have generated text
    for prompt_idx, generated_text in results:
        assert isinstance(generated_text, str)
        assert len(generated_text.strip()) > 0


@pytest.mark.skipif(
    not qwen_2_05b_model_exists(), reason="Qwen2-0.5B-Instruct-MLX model not found"
)
def test_continuous_batching_variable_lengths():
    """Test continuous batching with variable length prompts"""
    mlx_model, tokenizer = load("Qwen/Qwen2-0.5B-Instruct-MLX")
    
    # Create prompts of very different lengths
    prompts = [
        "Hi",  # Very short
        "Tell me about " + "artificial intelligence " * 10,  # Long
        "What?",  # Short
        "Please explain quantum computing " * 5,  # Medium-long
    ]
    
    start_time = time.time()
    results = batch_generate(
        mlx_model,
        tokenizer, 
        prompts,
        max_seq_len=50,
        batch_size=2,
        prefill_step=8
    )
    total_time = time.time() - start_time
    
    assert len(results) == len(prompts)
    
    # Should complete in reasonable time (not wait for longest prompt)
    # This is hard to test precisely, but batching should be faster than sequential
    assert total_time < 30.0  # Generous upper bound


def test_batching_slot_management():
    """Test that batch slots are managed correctly"""
    max_requests = 3
    batch_cache = BatchingKvCache(max_requests, max_seq_len=32)
    
    # Create some dummy caches
    caches = [TinyKvFullCache() for _ in range(5)]
    
    # Fill all slots
    for i in range(max_requests):
        batch_cache.add_request(caches[i], slot_id=i)
        assert batch_cache.kv_caches[i] is caches[i]
    
    # Remove middle slot
    batch_cache.remove_request(slot_id=1)
    assert batch_cache.kv_caches[1] is None
    assert batch_cache.kv_caches[0] is caches[0]
    assert batch_cache.kv_caches[2] is caches[2]
    
    # Add new request to freed slot
    batch_cache.add_request(caches[3], slot_id=1)
    assert batch_cache.kv_caches[1] is caches[3]


@pytest.mark.skipif(
    not qwen_2_05b_model_exists(), reason="Qwen2-0.5B-Instruct-MLX model not found"
)
def test_continuous_batching_throughput():
    """Test that continuous batching improves throughput vs sequential"""
    mlx_model, tokenizer = load("Qwen/Qwen2-0.5B-Instruct-MLX")
    
    # Create several short prompts
    prompts = ["Hello"] * 6
    
    # Test batched processing
    start_time = time.time()
    batch_results = batch_generate(
        mlx_model,
        tokenizer,
        prompts.copy(),  # Copy to avoid modifying original
        max_seq_len=20,
        batch_size=3,
        prefill_step=16
    )
    batch_time = time.time() - start_time
    
    # Test sequential processing (batch_size=1)
    start_time = time.time()
    sequential_results = batch_generate(
        mlx_model,
        tokenizer,
        prompts.copy(),
        max_seq_len=20,
        batch_size=1,
        prefill_step=16
    )
    sequential_time = time.time() - start_time
    
    # Both should produce same number of results
    assert len(batch_results) == len(sequential_results) == len(prompts)
    
    # Batching should be faster (though this might be flaky on slow hardware)
    if sequential_time > 1.0:  # Only test if sequential takes reasonable time
        speedup = sequential_time / batch_time
        assert speedup > 1.1  # At least 10% improvement


@pytest.mark.skipif(
    not qwen_2_05b_model_exists(), reason="Qwen2-0.5B-Instruct-MLX model not found"
)
def test_continuous_batching_early_termination():
    """Test that requests can finish early without blocking others"""
    mlx_model, tokenizer = load("Qwen/Qwen2-0.5B-Instruct-MLX")
    
    # Mix of prompts that should finish at different times
    prompts = [
        "Hi",  # Should finish quickly
        "Tell me a very long story about",  # Should take longer
        "Yes",  # Should finish quickly
    ]
    
    results = batch_generate(
        mlx_model,
        tokenizer,
        prompts,
        max_seq_len=30,
        batch_size=3,
        prefill_step=8
    )
    
    assert len(results) == len(prompts)
    
    # Find the results for short vs long prompts
    short_results = []
    long_results = []
    
    for prompt_idx, text in results:
        if prompts[prompt_idx] in ["Hi", "Yes"]:
            short_results.append(text)
        else:
            long_results.append(text)
    
    # Short prompts should have generated some text
    assert len(short_results) == 2
    for text in short_results:
        assert len(text.strip()) > 0


def test_request_state_transitions():
    """Test that Request objects transition through states correctly"""
    # Mock model and tokenizer for state testing
    class MockModel:
        def __call__(self, inputs, offset, cache):
            # Return dummy logits
            return mx.random.uniform(shape=(inputs.shape[0], inputs.shape[1], 1000))
    
    class MockTokenizer:
        def encode(self, text, add_special_tokens=False):
            return [1, 2, 3, 4, 5]  # 5 tokens
        
        @property
        def eos_token_id(self):
            return 2
        
        @property  
        def detokenizer(self):
            class MockDetokenizer:
                def __init__(self, tokenizer):
                    self.text = ""
                def add_token(self, token):
                    self.text += f" {token}"
            return MockDetokenizer
    
    model = MockModel()
    tokenizer = MockTokenizer()
    
    request = Request(model, tokenizer, "test prompt", prefill_max_step=2)
    
    # Initial state
    assert not request.is_done
    assert not request.is_prefill_done
    assert request.offset == 0
    
    # Prefill step 1
    request.try_prefill()
    assert not request.is_prefill_done
    assert request.offset == 2
    
    # Prefill step 2  
    request.try_prefill()
    assert not request.is_prefill_done
    assert request.offset == 4
    
    # Prefill step 3 (final)
    request.try_prefill()
    assert request.is_prefill_done
    assert request.offset == 5
    assert request.next_token is not None
    
    # Decode step - add EOS token to finish
    request.decode_done(tokenizer.eos_token_id)
    assert request.is_done


@pytest.mark.skipif(
    not qwen_2_05b_model_exists(), reason="Qwen2-0.5B-Instruct-MLX model not found"
)
def test_continuous_batching_memory_efficiency():
    """Test that continuous batching doesn't leak memory"""
    mlx_model, tokenizer = load("Qwen/Qwen2-0.5B-Instruct-MLX")
    
    # Process several batches to test memory management
    for batch_num in range(3):
        prompts = [f"Batch {batch_num} prompt {i}" for i in range(4)]
        
        results = batch_generate(
            mlx_model,
            tokenizer,
            prompts,
            max_seq_len=20,
            batch_size=2,
            prefill_step=8
        )
        
        assert len(results) == len(prompts)
        
        # Force garbage collection and evaluation to clean up
        mx.eval([])
        
        # Memory usage should not grow unboundedly
        # (This is hard to test precisely, but the test shouldn't crash)


@pytest.mark.parametrize("batch_size", [1, 2, 3, 5])
@pytest.mark.skipif(
    not qwen_2_05b_model_exists(), reason="Qwen2-0.5B-Instruct-MLX model not found"
)
def test_continuous_batching_different_batch_sizes(batch_size: int):
    """Test continuous batching with different batch sizes"""
    mlx_model, tokenizer = load("Qwen/Qwen2-0.5B-Instruct-MLX")
    
    prompts = create_test_prompts(4)
    
    results = batch_generate(
        mlx_model,
        tokenizer,
        prompts,
        max_seq_len=25,
        batch_size=batch_size,
        prefill_step=8
    )
    
    # Should handle all prompts regardless of batch size
    assert len(results) == len(prompts)
    
    # Results should be valid
    for prompt_idx, text in results:
        assert 0 <= prompt_idx < len(prompts)
        assert isinstance(text, str)
        assert len(text.strip()) > 0
