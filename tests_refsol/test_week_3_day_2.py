import pytest
from tiny_llm_ref.paged_attention import CacheManager
from extensions_ref import tiny_llm_ext_ref
import mlx.core as mx

class TestPagedAttention:
    def test_paged_attention_kernel(self):
        cache_manager = CacheManager(num_pages=100, page_size=16, head_dim=64, num_heads=8)
        
        # This test is a placeholder and does not verify the correctness of the
        # paged attention algorithm. It only checks that the kernel can be
        # launched without errors.
        
        q = mx.random.uniform(shape=(1, 8, 128, 64))
        k_cache = mx.zeros((100, 8, 16, 64))
        v_cache = mx.zeros((100, 8, 16, 64))
        page_table = mx.array([0, 1, 2, 3], dtype=mx.int32)
        
        out = tiny_llm_ext_ref.paged_attention(q, k_cache, v_cache, page_table)
        
        assert out.shape == q.shape
        assert out.dtype == q.dtype
