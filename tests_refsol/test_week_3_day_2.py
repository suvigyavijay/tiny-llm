import pytest
from tiny_llm_ref.paged_attention import CacheManager
from extensions_ref import tiny_llm_ext_ref
import mlx.core as mx
import numpy as np

def softmax_np(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)

def paged_attention_ref_py(q, k_cache, v_cache, page_table):
    q_np = np.array(q, copy=False)
    k_cache_np = np.array(k_cache, copy=False)
    v_cache_np = np.array(v_cache, copy=False)
    page_table_np = np.array(page_table, copy=False)

    B, n_heads, seq_len, head_dim = q_np.shape
    page_size = k_cache_np.shape[2]
    
    output = np.zeros_like(q_np)

    for b in range(B):
        for h in range(n_heads):
            # Reconstruct the key and value sequences for the head
            keys = []
            values = []
            for page_idx in page_table_np:
                keys.append(k_cache_np[page_idx, h, :, :])
                values.append(v_cache_np[page_idx, h, :, :])
            
            k_seq = np.concatenate(keys, axis=0)[:seq_len]
            v_seq = np.concatenate(values, axis=0)[:seq_len]

            for i in range(seq_len):
                query_vector = q_np[b, h, i, :]
                
                # Apply causal mask by slicing keys and values
                causal_k = k_seq[:i+1, :]
                causal_v = v_seq[:i+1, :]
                
                scores = (query_vector @ causal_k.T) / np.sqrt(head_dim)
                attention_weights = softmax_np(scores)
                
                output[b, h, i, :] = attention_weights @ causal_v
                
    return mx.array(output)


class TestPagedAttention:
    @pytest.mark.parametrize("seq_len, head_dim, page_size, num_pages", [
        (32, 64, 16, 2),
        (64, 32, 16, 4),
        (15, 64, 16, 1),
        (33, 32, 16, 3),
    ])
    def test_paged_attention_kernel(self, seq_len, head_dim, page_size, num_pages):
        n_heads = 8
        cache_manager = CacheManager(num_pages=num_pages, page_size=page_size, head_dim=head_dim, num_heads=n_heads)
        
        q = mx.random.uniform(shape=(1, n_heads, seq_len, head_dim))
        k_cache, v_cache = cache_manager.get_cache()
    
        # Allocate pages for a single sequence
        cache_manager.add_sequence(0)
        for _ in range(num_pages):
            cache_manager.extend_sequence(0)
        
        page_table = mx.array(cache_manager.get_sequence_page_ids(0), dtype=mx.int32)
        
        # Populate the cache with some data
        for page_idx in cache_manager.get_sequence_page_ids(0):
            k_cache[page_idx] = mx.random.uniform(shape=(n_heads, page_size, head_dim))
            v_cache[page_idx] = mx.random.uniform(shape=(n_heads, page_size, head_dim))
        
        # Run the kernel
        out_ref = paged_attention_ref_py(q, k_cache, v_cache, page_table)
        out = tiny_llm_ext_ref.paged_attention(q, k_cache, v_cache, page_table)
        
        assert out.shape == out_ref.shape
        assert out.dtype == out_ref.dtype
        assert np.allclose(np.array(out), np.array(out_ref), atol=1e-5)
