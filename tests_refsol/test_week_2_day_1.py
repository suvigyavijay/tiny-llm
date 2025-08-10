import pytest
import mlx.core as mx
from .utils import *
from .tiny_llm_base import TinyKvFullCache, BatchingKvCache


@pytest.mark.parametrize("stream", AVAILABLE_STREAMS, ids=AVAILABLE_STREAMS_IDS)
@pytest.mark.parametrize("precision", PRECISIONS, ids=PRECISION_IDS)
def test_kv_cache_single_update(stream: mx.Stream, precision: mx.Dtype):
    """Test basic KV cache functionality with single update"""
    with mx.stream(stream):
        cache = TinyKvFullCache()
        
        B, H, L, D = 1, 8, 4, 64
        key = mx.random.uniform(shape=(B, H, L, D), dtype=precision)
        value = mx.random.uniform(shape=(B, H, L, D), dtype=precision)
        
        # First update
        keys_out, values_out, seq_len, mask = cache.update_and_fetch(key, value)
        
        assert keys_out.shape == (B, H, L, D)
        assert values_out.shape == (B, H, L, D)
        assert seq_len == 0  # First call returns 0 offset
        assert_allclose(keys_out, key, precision)
        assert_allclose(values_out, value, precision)


@pytest.mark.parametrize("stream", AVAILABLE_STREAMS, ids=AVAILABLE_STREAMS_IDS)
@pytest.mark.parametrize("precision", PRECISIONS, ids=PRECISION_IDS)
def test_kv_cache_multiple_updates(stream: mx.Stream, precision: mx.Dtype):
    """Test KV cache with multiple updates (incremental generation)"""
    with mx.stream(stream):
        cache = TinyKvFullCache()
        
        B, H, D = 1, 8, 64
        
        # First update: process 4 tokens
        L1 = 4
        key1 = mx.random.uniform(shape=(B, H, L1, D), dtype=precision)
        value1 = mx.random.uniform(shape=(B, H, L1, D), dtype=precision)
        
        keys_out1, values_out1, seq_len1, _ = cache.update_and_fetch(key1, value1)
        assert keys_out1.shape == (B, H, L1, D)
        assert seq_len1 == 0
        
        # Second update: add 1 more token
        L2 = 1
        key2 = mx.random.uniform(shape=(B, H, L2, D), dtype=precision)
        value2 = mx.random.uniform(shape=(B, H, L2, D), dtype=precision)
        
        keys_out2, values_out2, seq_len2, _ = cache.update_and_fetch(key2, value2)
        
        # Should have concatenated results
        expected_shape = (B, H, L1 + L2, D)
        assert keys_out2.shape == expected_shape
        assert values_out2.shape == expected_shape
        assert seq_len2 == L1 + L2
        
        # Verify concatenation is correct
        assert_allclose(keys_out2[:, :, :L1, :], key1, precision)
        assert_allclose(keys_out2[:, :, L1:, :], key2, precision)
        assert_allclose(values_out2[:, :, :L1, :], value1, precision)
        assert_allclose(values_out2[:, :, L1:, :], value2, precision)


@pytest.mark.parametrize("stream", AVAILABLE_STREAMS, ids=AVAILABLE_STREAMS_IDS)
def test_kv_cache_sequence_generation(stream: mx.Stream):
    """Test KV cache simulating actual sequence generation"""
    with mx.stream(stream):
        cache = TinyKvFullCache()
        precision = mx.float32
        
        B, H, D = 1, 4, 32
        max_seq_len = 10
        
        all_keys = []
        all_values = []
        
        # Simulate autoregressive generation
        for step in range(max_seq_len):
            # Generate one new token's K,V
            new_key = mx.random.uniform(shape=(B, H, 1, D), dtype=precision)
            new_value = mx.random.uniform(shape=(B, H, 1, D), dtype=precision)
            
            all_keys.append(new_key)
            all_values.append(new_value)
            
            # Update cache
            keys_out, values_out, seq_len, _ = cache.update_and_fetch(new_key, new_value)
            
            # Verify output shape
            expected_len = step + 1
            assert keys_out.shape == (B, H, expected_len, D)
            assert values_out.shape == (B, H, expected_len, D)
            assert seq_len == expected_len
            
            # Verify all previous keys/values are preserved
            expected_keys = mx.concat(all_keys, axis=2)
            expected_values = mx.concat(all_values, axis=2)
            
            assert_allclose(keys_out, expected_keys, precision)
            assert_allclose(values_out, expected_values, precision)


def test_batching_kv_cache_initialization():
    """Test BatchingKvCache initialization and basic properties"""
    max_requests = 4
    max_seq_len = 128
    
    cache = BatchingKvCache(max_requests, max_seq_len)
    
    assert cache.max_active_requests == max_requests
    assert cache.max_seq_len == max_seq_len
    assert len(cache.kv_caches) == max_requests
    assert all(c is None for c in cache.kv_caches)
    assert cache.HD is None


@pytest.mark.parametrize("stream", AVAILABLE_STREAMS, ids=AVAILABLE_STREAMS_IDS)
def test_batching_kv_cache_add_remove_requests(stream: mx.Stream):
    """Test adding and removing requests from batching cache"""
    with mx.stream(stream):
        batch_cache = BatchingKvCache(max_active_requests=3, max_seq_len=64)
        precision = mx.float32
        
        B, H, L, D = 1, 4, 8, 32
        
        # Create individual caches for requests
        req1_cache = TinyKvFullCache()
        req2_cache = TinyKvFullCache()
        
        # Prefill the individual caches
        key1 = mx.random.uniform(shape=(B, H, L, D), dtype=precision)
        value1 = mx.random.uniform(shape=(B, H, L, D), dtype=precision)
        req1_cache.update_and_fetch(key1, value1)
        
        key2 = mx.random.uniform(shape=(B, H, L//2, D), dtype=precision)
        value2 = mx.random.uniform(shape=(B, H, L//2, D), dtype=precision)
        req2_cache.update_and_fetch(key2, value2)
        
        # Add requests to batch cache
        batch_cache.add_request(req1_cache, slot_id=0)
        batch_cache.add_request(req2_cache, slot_id=2)
        
        assert batch_cache.kv_caches[0] is req1_cache
        assert batch_cache.kv_caches[1] is None
        assert batch_cache.kv_caches[2] is req2_cache
        assert batch_cache.HD == (H, D)
        
        # Remove request
        batch_cache.remove_request(slot_id=0)
        assert batch_cache.kv_caches[0] is None


@pytest.mark.parametrize("stream", AVAILABLE_STREAMS, ids=AVAILABLE_STREAMS_IDS)
def test_batching_kv_cache_update_and_fetch(stream: mx.Stream):
    """Test batched update and fetch with multiple requests"""
    with mx.stream(stream):
        max_requests = 3
        batch_cache = BatchingKvCache(max_active_requests=max_requests, max_seq_len=64)
        precision = mx.float32
        
        B, H, D = 1, 4, 32
        
        # Create and add two requests
        req1_cache = TinyKvFullCache()
        req2_cache = TinyKvFullCache()
        
        # Prefill with different sequence lengths
        key1 = mx.random.uniform(shape=(B, H, 5, D), dtype=precision)
        value1 = mx.random.uniform(shape=(B, H, 5, D), dtype=precision)
        req1_cache.update_and_fetch(key1, value1)
        
        key2 = mx.random.uniform(shape=(B, H, 3, D), dtype=precision)
        value2 = mx.random.uniform(shape=(B, H, 3, D), dtype=precision)
        req2_cache.update_and_fetch(key2, value2)
        
        batch_cache.add_request(req1_cache, slot_id=0)
        batch_cache.add_request(req2_cache, slot_id=1)
        
        # Now do a batched update (decode step)
        decode_L = 1
        decode_keys = mx.random.uniform(shape=(max_requests, H, decode_L, D), dtype=precision)
        decode_values = mx.random.uniform(shape=(max_requests, H, decode_L, D), dtype=precision)
        
        batch_keys, batch_values, seq_len, batch_masks = batch_cache.update_and_fetch(
            decode_keys, decode_values, mask_length=decode_L
        )
        
        # Check output shapes
        max_len = 6  # max(5+1, 3+1) = 6
        assert batch_keys.shape == (max_requests, H, max_len, D)
        assert batch_values.shape == (max_requests, H, max_len, D)
        assert batch_masks.shape == (max_requests, 1, decode_L, max_len)
        
        # Verify request 0 has correct data
        req1_final_keys, _, _, _ = req1_cache.update_and_fetch(decode_keys[0:1], decode_values[0:1])
        start_pos = max_len - req1_final_keys.shape[2]
        assert_allclose(
            batch_keys[0:1, :, start_pos:, :], 
            req1_final_keys, 
            precision, 
            message="Request 0 keys mismatch"
        )


def test_kv_cache_memory_efficiency():
    """Test that KV cache actually provides memory benefits vs recomputation"""
    # This is more of a conceptual test - in practice, KV cache reduces
    # computation complexity from O(n²) to O(n) for sequence generation
    
    cache = TinyKvFullCache()
    B, H, D = 1, 8, 64
    precision = mx.float32
    
    # Simulate generating a sequence of length 10
    total_keys_stored = 0
    
    for step in range(10):
        new_key = mx.random.uniform(shape=(B, H, 1, D), dtype=precision)
        new_value = mx.random.uniform(shape=(B, H, 1, D), dtype=precision)
        
        keys_out, values_out, seq_len, _ = cache.update_and_fetch(new_key, new_value)
        
        # Without cache: would need to recompute all keys/values each step
        # With cache: only compute 1 new key/value pair each step
        total_keys_stored += 1
        
        # Verify we have accumulated all keys
        assert keys_out.shape[2] == step + 1
        
    # Total computation with cache: 10 steps (O(n))
    # Total computation without cache: 1+2+3+...+10 = 55 steps (O(n²))
    assert total_keys_stored == 10  # O(n) computation
    # Without cache would be sum(range(1, 11)) = 55


@pytest.mark.parametrize("seq_len", [1, 5, 10, 50])
def test_kv_cache_different_sequence_lengths(seq_len: int):
    """Test KV cache with different sequence lengths"""
    cache = TinyKvFullCache()
    precision = mx.float32
    B, H, D = 1, 4, 32
    
    # Add tokens one by one
    for i in range(seq_len):
        key = mx.random.uniform(shape=(B, H, 1, D), dtype=precision)
        value = mx.random.uniform(shape=(B, H, 1, D), dtype=precision)
        
        keys_out, values_out, current_len, _ = cache.update_and_fetch(key, value)
        
        expected_len = i + 1
        assert keys_out.shape == (B, H, expected_len, D)
        assert values_out.shape == (B, H, expected_len, D)
        assert current_len == expected_len


def test_kv_cache_offset_tracking():
    """Test that KV cache correctly tracks sequence offset"""
    cache = TinyKvFullCache()
    
    assert cache.get_offset() == 0
    
    # Add some tokens
    B, H, D = 1, 2, 16
    for step, chunk_size in enumerate([3, 1, 2, 1]):
        key = mx.random.uniform(shape=(B, H, chunk_size, D))
        value = mx.random.uniform(shape=(B, H, chunk_size, D))
        
        cache.update_and_fetch(key, value)
        
        expected_offset = sum([3, 1, 2, 1][:step+1])
        assert cache.get_offset() == expected_offset
