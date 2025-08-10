import pytest
import mlx.core as mx
from tiny_llm_ref.long_context import sliding_window_attention

class TestLongContext:
    def test_sliding_window_attention(self):
        q = mx.random.uniform(shape=(1, 10, 64))
        k = mx.random.uniform(shape=(1, 10, 64))
        v = mx.random.uniform(shape=(1, 10, 64))
        
        window_size = 3
        out = sliding_window_attention(q, k, v, window_size)
        
        assert out.shape == q.shape
        assert out.dtype == q.dtype

    def test_sliding_window_attention_edge_case(self):
        q = mx.random.uniform(shape=(1, 5, 32))
        k = mx.random.uniform(shape=(1, 5, 32))
        v = mx.random.uniform(shape=(1, 5, 32))
        
        window_size = 5
        out = sliding_window_attention(q, k, v, window_size)
        
        assert out.shape == q.shape
        assert out.dtype == q.dtype
