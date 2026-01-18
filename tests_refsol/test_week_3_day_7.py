import pytest
import mlx.core as mx
from tiny_llm_ref.long_context import apply_linear_scaling_rope


def test_linear_scaling_factor_2():
    """Test linear scaling with factor 2 (doubling context)."""
    freqs = mx.array([1.0, 2.0, 4.0])
    scale = 2.0
    
    scaled_freqs = apply_linear_scaling_rope(freqs, scale)
    expected = mx.array([0.5, 1.0, 2.0])
    
    assert mx.allclose(scaled_freqs, expected)


def test_linear_scaling_factor_1():
    """Test that scale factor 1 leaves frequencies unchanged."""
    freqs = mx.array([1.0, 2.0, 4.0])
    
    scaled_freqs = apply_linear_scaling_rope(freqs, 1.0)
    
    assert mx.allclose(scaled_freqs, freqs)


def test_linear_scaling_preserves_dtype():
    """Test that scaling preserves the dtype of input frequencies."""
    freqs = mx.array([1.0, 2.0], dtype=mx.float16)
    
    scaled_freqs = apply_linear_scaling_rope(freqs, 2.0)
    
    assert scaled_freqs.dtype == mx.float16
