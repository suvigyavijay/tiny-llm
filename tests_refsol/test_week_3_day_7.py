import pytest
import mlx.core as mx
from tiny_llm_ref.long_context import sliding_window_attention
import numpy as np

def sliding_window_attention_ref(q, k, v, window_size):
    """
    Reference implementation of sliding window attention.
    """
    B, S, D = q.shape
    output = mx.zeros_like(q)
    
    for i in range(S):
        start = max(0, i - window_size + 1)
        end = i + 1
        
        # Create a causal mask for the window
        mask = mx.zeros((1, S))
        mask[:, start:end] = 1.0
        
        # Apply mask to keys
        attention_scores = (q[:, i:i+1] @ k.transpose(0, 2, 1)) / mx.sqrt(D)
        
        # Apply causal mask and window mask
        attention_scores = mx.where(mask == 0, -1e9, attention_scores)
        
        attention_weights = mx.softmax(attention_scores, axis=-1)
        
        output[:, i:i+1] = attention_weights @ v
        
    return output

class TestLongContext:
    @pytest.mark.parametrize("seq_len, head_dim, window_size", [
        (10, 64, 3),
        (20, 32, 5),
        (5, 32, 5),
        (10, 16, 1),
    ])
    def test_sliding_window_attention_parametrized(self, seq_len, head_dim, window_size):
        q = mx.random.uniform(shape=(1, seq_len, head_dim))
        k = mx.random.uniform(shape=(1, seq_len, head_dim))
        v = mx.random.uniform(shape=(1, seq_len, head_dim))
        
        out = sliding_window_attention(q, k, v, window_size)
        
        assert out.shape == q.shape
        assert out.dtype == q.dtype

    def test_sliding_window_attention_causal_mask(self):
        seq_len = 10
        head_dim = 64
        window_size = 3
        q = mx.random.uniform(shape=(1, seq_len, head_dim))
        k = mx.random.uniform(shape=(1, seq_len, head_dim))
        v = mx.random.uniform(shape=(1, seq_len, head_dim))

        # Can't easily inspect the mask, so we'll compare with a reference implementation
        out_ref = sliding_window_attention_ref(q, k, v, window_size)
        out = sliding_window_attention(q, k, v, window_size)

        assert np.allclose(out_ref, out, atol=1e-5)

    def test_sliding_window_attention_values(self):
        seq_len = 4
        head_dim = 2
        window_size = 2
        
        # Use simple, predictable values
        q = mx.array([[[1, 0], [0, 1], [1, 1], [0, 0]]], dtype=mx.float32)
        k = mx.array([[[1, 0], [0, 1], [1, 0], [0, 1]]], dtype=mx.float32)
        v = mx.array([[[1, 2], [3, 4], [5, 6], [7, 8]]], dtype=mx.float32)
        
        out_ref = sliding_window_attention_ref(q, k, v, window_size)
        out = sliding_window_attention(q, k, v, window_size)
        
        assert np.allclose(out_ref, out, atol=1e-5)
