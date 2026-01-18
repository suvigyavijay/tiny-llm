import pytest
import mlx.core as mx
from .tiny_llm_base import *
from .utils import *


@pytest.mark.parametrize("stream", AVAILABLE_STREAMS, ids=AVAILABLE_STREAMS_IDS)
@pytest.mark.parametrize("precision", PRECISIONS, ids=PRECISION_IDS)
def test_linear_scaling_rope(stream: mx.Stream, precision: mx.Dtype):
    """Test linear RoPE scaling produces correct frequency scaling."""
    with mx.stream(stream):
        base_freqs = mx.array([1.0, 0.5, 0.25, 0.125], dtype=precision)
        scale_factor = 2.0
        
        scaled = apply_linear_scaling_rope(base_freqs, scale_factor)
        mx.eval(scaled)
        
        # Linear scaling divides frequencies by scale_factor
        expected = base_freqs / scale_factor
        assert_allclose(scaled, expected, precision=precision)


@pytest.mark.parametrize("scale_factor", [1.0, 2.0, 4.0])
def test_linear_scaling_identity(scale_factor: float):
    """Test that scale_factor=1.0 returns original frequencies."""
    base_freqs = mx.array([1.0, 0.5, 0.25])
    
    if scale_factor == 1.0:
        scaled = apply_linear_scaling_rope(base_freqs, scale_factor)
        mx.eval(scaled)
        
        assert mx.allclose(scaled, base_freqs)


@pytest.mark.parametrize("dim", [32, 64, 128])
def test_linear_scaling_preserves_shape(dim: int):
    """Test that output shape matches input shape."""
    base_freqs = mx.random.normal((dim,))
    scale_factor = 2.0
    
    scaled = apply_linear_scaling_rope(base_freqs, scale_factor)
    mx.eval(scaled)
    
    assert scaled.shape == base_freqs.shape
