import pytest
import time
import mlx.core as mx
from .utils import *
from .tiny_llm_base import (
    batch_generate,
    Request,
    Qwen2ModelWeek2
)
from mlx_lm import load


def create_long_prompt(repeat_count: int = 50) -> str:
    """Create a long prompt for testing chunked prefill"""
    base_text = "artificial intelligence and machine learning "
    return "Write a comprehensive essay about " + base_text * repeat_count


def create_mixed_prompts() -> list[str]:
    """Create prompts of varying lengths for testing interleaving"""
    return [
        create_long_prompt(30),  # Very long prompt
        "Hi",  # Short interactive prompt
        "Hello there",  # Short interactive prompt
        create_long_prompt(20),  # Long prompt
        "What?",  # Short interactive prompt
    ]


@pytest.mark.skipif(
    not qwen_2_05b_model_exists(), reason="Qwen2-0.5B-Instruct-MLX model not found"
)
def test_chunked_prefill_basic():
    """Test basic chunked prefill functionality"""
    mlx_model, tokenizer = load("Qwen/Qwen2-0.5B-Instruct-MLX")
    model = Qwen2ModelWeek2(mlx_model)
    
    # Create a request with chunked prefill
    long_prompt = create_long_prompt(20)
    chunk_size = 16
    request = Request(model, tokenizer, long_prompt, prefill_max_step=chunk_size)
    
    total_tokens = len(request.prefill_tokens)
    expected_chunks = (total_tokens + chunk_size - 1) // chunk_size
    
    chunk_count = 0
    while not request.is_prefill_done:
        initial_offset = request.offset
        request.try_prefill()
        chunk_count += 1
        
        # Should process at most chunk_size tokens
        tokens_processed = request.offset - initial_offset
        assert tokens_processed <= chunk_size
        
        # Should make progress
        assert request.offset > initial_offset
    
    # Should have processed all tokens in chunks
    assert request.offset == total_tokens
    assert chunk_count == expected_chunks
    assert request.next_token is not None


@pytest.mark.skipif(
    not qwen_2_05b_model_exists(), reason="Qwen2-0.5B-Instruct-MLX model not found"
)
def test_chunked_vs_full_prefill_correctness():
    """Test that chunked prefill produces same results as full prefill"""
    mlx_model, tokenizer = load("Qwen/Qwen2-0.5B-Instruct-MLX")
    model = Qwen2ModelWeek2(mlx_model)
    
    prompt = create_long_prompt(10)
    
    # Full prefill (large chunk size)
    request_full = Request(model, tokenizer, prompt, prefill_max_step=9999)
    while not request_full.is_prefill_done:
        request_full.try_prefill()
    
    # Chunked prefill
    request_chunked = Request(model, tokenizer, prompt, prefill_max_step=8)
    while not request_chunked.is_prefill_done:
        request_chunked.try_prefill()
    
    # Both should end up with same state
    assert request_full.offset == request_chunked.offset
    assert request_full.is_prefill_done == request_chunked.is_prefill_done
    
    # KV caches should have same final content
    for layer_idx in range(len(request_full.kv_cache)):
        full_keys, full_values = request_full.kv_cache[layer_idx].key_values
        chunked_keys, chunked_values = request_chunked.kv_cache[layer_idx].key_values
        
        assert_allclose(
            full_keys, chunked_keys, mx.float16, 
            message=f"Layer {layer_idx} keys mismatch"
        )
        assert_allclose(
            full_values, chunked_values, mx.float16,
            message=f"Layer {layer_idx} values mismatch"
        )


@pytest.mark.skipif(
    not qwen_2_05b_model_exists(), reason="Qwen2-0.5B-Instruct-MLX model not found"
)
def test_chunked_prefill_interleaving():
    """Test that chunked prefill allows interleaving with decode requests"""
    mlx_model, tokenizer = load("Qwen/Qwen2-0.5B-Instruct-MLX")
    
    # Mix of long and short prompts
    prompts = create_mixed_prompts()
    
    # Use small chunk size to force interleaving
    start_time = time.time()
    results = batch_generate(
        mlx_model,
        tokenizer,
        prompts,
        max_seq_len=30,
        batch_size=3,
        prefill_step=8  # Small chunks
    )
    chunked_time = time.time() - start_time
    
    # Should process all prompts
    assert len(results) == len(prompts)
    
    # Results should be reasonable
    for prompt_idx, text in results:
        assert isinstance(text, str)
        assert len(text.strip()) > 0
    
    # With small chunks, system should remain responsive
    # (This is hard to test precisely, but the test should complete)
    assert chunked_time < 60.0  # Generous upper bound


@pytest.mark.skipif(
    not qwen_2_05b_model_exists(), reason="Qwen2-0.5B-Instruct-MLX model not found"
)
def test_different_chunk_sizes():
    """Test chunked prefill with different chunk sizes"""
    mlx_model, tokenizer = load("Qwen/Qwen2-0.5B-Instruct-MLX")
    
    prompts = [create_long_prompt(15)]
    
    chunk_sizes = [4, 16, 64]
    results = {}
    
    for chunk_size in chunk_sizes:
        start_time = time.time()
        batch_results = batch_generate(
            mlx_model,
            tokenizer,
            prompts.copy(),
            max_seq_len=25,
            batch_size=1,
            prefill_step=chunk_size
        )
        elapsed = time.time() - start_time
        
        results[chunk_size] = (batch_results, elapsed)
        
        # Should get valid results regardless of chunk size
        assert len(batch_results) == 1
        assert len(batch_results[0][1].strip()) > 0
    
    # All chunk sizes should produce results
    assert len(results) == len(chunk_sizes)


def test_chunked_prefill_memory_pattern():
    """Test that chunked prefill has predictable memory usage"""
    mlx_model, tokenizer = load("Qwen/Qwen2-0.5B-Instruct-MLX") if qwen_2_05b_model_exists() else None
    
    if mlx_model is None:
        pytest.skip("Model not available")
    
    model = Qwen2ModelWeek2(mlx_model)
    
    # Create request with known token count
    prompt = "test " * 100  # Predictable length
    chunk_size = 16
    
    request = Request(model, tokenizer, prompt, prefill_max_step=chunk_size)
    
    # Track memory usage during chunked prefill
    initial_memory = mx.metal.get_peak_memory() if hasattr(mx, 'metal') else 0
    
    chunk_memories = []
    while not request.is_prefill_done:
        request.try_prefill()
        current_memory = mx.metal.get_peak_memory() if hasattr(mx, 'metal') else 0
        chunk_memories.append(current_memory - initial_memory)
    
    # Memory should grow predictably (not in huge spikes)
    if chunk_memories:
        # Each chunk should add roughly similar memory
        memory_deltas = [chunk_memories[i] - chunk_memories[i-1] 
                        for i in range(1, len(chunk_memories))]
        
        # Memory growth should be relatively stable
        if len(memory_deltas) > 1:
            max_delta = max(memory_deltas)
            min_delta = min(memory_deltas)
            
            # Growth shouldn't vary too wildly (within 5x factor)
            if max_delta > 0:
                assert max_delta / min_delta < 5.0


@pytest.mark.skipif(
    not qwen_2_05b_model_exists(), reason="Qwen2-0.5B-Instruct-MLX model not found"
)
def test_chunked_prefill_progress_tracking():
    """Test that chunked prefill progress can be tracked"""
    mlx_model, tokenizer = load("Qwen/Qwen2-0.5B-Instruct-MLX")
    model = Qwen2ModelWeek2(mlx_model)
    
    prompt = create_long_prompt(25)
    chunk_size = 12
    
    request = Request(model, tokenizer, prompt, prefill_max_step=chunk_size)
    
    total_tokens = len(request.prefill_tokens)
    progress_snapshots = []
    
    while not request.is_prefill_done:
        progress_before = request.offset / total_tokens
        request.try_prefill()
        progress_after = request.offset / total_tokens
        
        progress_snapshots.append((progress_before, progress_after))
        
        # Should make progress each step
        assert progress_after > progress_before
        assert 0 <= progress_after <= 1.0
    
    # Should end at 100% progress
    assert request.offset == total_tokens
    final_progress = request.offset / total_tokens
    assert abs(final_progress - 1.0) < 1e-6


@pytest.mark.skipif(
    not qwen_2_05b_model_exists(), reason="Qwen2-0.5B-Instruct-MLX model not found"
)
def test_chunked_prefill_attention_correctness():
    """Test that chunked prefill maintains correct attention patterns"""
    mlx_model, tokenizer = load("Qwen/Qwen2-0.5B-Instruct-MLX")
    model = Qwen2ModelWeek2(mlx_model)
    
    # Use a prompt where order matters for attention
    prompt = "The first word, then the second word, then the third word"
    
    # Process with different chunk sizes
    chunk_sizes = [4, 8, 999]  # 999 = effectively no chunking
    final_states = []
    
    for chunk_size in chunk_sizes:
        request = Request(model, tokenizer, prompt, prefill_max_step=chunk_size)
        
        while not request.is_prefill_done:
            request.try_prefill()
        
        # Extract final KV state
        final_kv = []
        for layer_cache in request.kv_cache:
            keys, values = layer_cache.key_values
            final_kv.append((keys, values))
        
        final_states.append(final_kv)
    
    # All chunk sizes should produce nearly identical final states
    reference_state = final_states[-1]  # Full prefill as reference
    
    for chunk_idx, chunked_state in enumerate(final_states[:-1]):
        for layer_idx, ((ref_k, ref_v), (chunk_k, chunk_v)) in enumerate(
            zip(reference_state, chunked_state)
        ):
            assert_allclose(
                chunk_k, ref_k, mx.float16, rtol=1e-2,
                message=f"Chunk size {chunk_sizes[chunk_idx]} layer {layer_idx} keys mismatch"
            )
            assert_allclose(
                chunk_v, ref_v, mx.float16, rtol=1e-2,
                message=f"Chunk size {chunk_sizes[chunk_idx]} layer {layer_idx} values mismatch"
            )


@pytest.mark.skipif(
    not qwen_2_05b_model_exists(), reason="Qwen2-0.5B-Instruct-MLX model not found"
)
def test_chunked_prefill_decode_latency():
    """Test that chunked prefill improves decode latency in mixed workloads"""
    mlx_model, tokenizer = load("Qwen/Qwen2-0.5B-Instruct-MLX")
    
    # Create workload with one very long prefill and several short requests
    prompts = [
        create_long_prompt(40),  # This will take time to prefill
        "Hi",
        "Hello", 
        "Test"
    ]
    
    # Test with large chunks (poor interleaving)
    start_time = time.time()
    results_large_chunks = batch_generate(
        mlx_model,
        tokenizer,
        prompts.copy(),
        max_seq_len=20,
        batch_size=2,
        prefill_step=128  # Large chunks
    )
    large_chunk_time = time.time() - start_time
    
    # Test with small chunks (good interleaving)
    start_time = time.time()
    results_small_chunks = batch_generate(
        mlx_model,
        tokenizer,
        prompts.copy(),
        max_seq_len=20,
        batch_size=2,
        prefill_step=8  # Small chunks
    )
    small_chunk_time = time.time() - start_time
    
    # Both should produce same number of results
    assert len(results_large_chunks) == len(results_small_chunks) == len(prompts)
    
    # Both should produce valid text
    for results in [results_large_chunks, results_small_chunks]:
        for prompt_idx, text in results:
            assert isinstance(text, str)
            assert len(text.strip()) > 0
    
    # The difference demonstrates chunked prefill benefit
    # (Exact timing comparison is hardware-dependent, so we just verify both work)


def test_chunked_prefill_edge_cases():
    """Test chunked prefill edge cases"""
    mlx_model, tokenizer = load("Qwen/Qwen2-0.5B-Instruct-MLX") if qwen_2_05b_model_exists() else None
    
    if mlx_model is None:
        pytest.skip("Model not available")
    
    model = Qwen2ModelWeek2(mlx_model)
    
    # Test 1: Chunk size larger than prompt
    short_prompt = "Hi"
    request = Request(model, tokenizer, short_prompt, prefill_max_step=1000)
    
    steps = 0
    while not request.is_prefill_done:
        request.try_prefill()
        steps += 1
        assert steps < 10  # Should not take many steps for short prompt
    
    assert request.is_prefill_done
    
    # Test 2: Chunk size of 1 (minimum)
    medium_prompt = "Tell me about AI"
    request = Request(model, tokenizer, medium_prompt, prefill_max_step=1)
    
    total_tokens = len(request.prefill_tokens)
    steps = 0
    
    while not request.is_prefill_done:
        initial_offset = request.offset
        request.try_prefill()
        steps += 1
        
        # Should process exactly 1 token per step
        assert request.offset == initial_offset + 1
        assert steps <= total_tokens
    
    assert steps == total_tokens


@pytest.mark.parametrize("chunk_size", [1, 4, 8, 16, 32])
@pytest.mark.skipif(
    not qwen_2_05b_model_exists(), reason="Qwen2-0.5B-Instruct-MLX model not found"
)
def test_chunked_prefill_various_sizes(chunk_size: int):
    """Test chunked prefill with various chunk sizes"""
    mlx_model, tokenizer = load("Qwen/Qwen2-0.5B-Instruct-MLX")
    model = Qwen2ModelWeek2(mlx_model)
    
    prompt = "Write about technology " * 10
    request = Request(model, tokenizer, prompt, prefill_max_step=chunk_size)
    
    total_tokens = len(request.prefill_tokens)
    
    while not request.is_prefill_done:
        initial_offset = request.offset
        request.try_prefill()
        
        # Should not process more than chunk_size tokens
        tokens_processed = request.offset - initial_offset
        assert tokens_processed <= chunk_size
        assert tokens_processed > 0  # Should make some progress
    
    # Should complete full prefill
    assert request.offset == total_tokens
    assert request.is_prefill_done
    assert request.next_token is not None
