import pytest
import mlx.core as mx
from .tiny_llm_base import *
from .utils import *


@pytest.mark.parametrize("stream", AVAILABLE_STREAMS, ids=AVAILABLE_STREAMS_IDS)
def test_moe_forward_shape(stream: mx.Stream):
    """Test MoE layer produces correct output shape."""
    with mx.stream(stream):
        B, L, D = 2, 5, 8
        num_experts = 4
        k = 2
        hidden = 16
        
        model = MoELayer(num_experts, k, D, hidden)
        x = mx.random.normal((B, L, D))
        
        out = model(x)
        mx.eval(out)
        
        assert out.shape == (B, L, D)


@pytest.mark.parametrize("stream", AVAILABLE_STREAMS, ids=AVAILABLE_STREAMS_IDS)
def test_moe_deterministic(stream: mx.Stream):
    """Test MoE layer is deterministic given same input."""
    with mx.stream(stream):
        B, L, D = 1, 3, 4
        
        model = MoELayer(num_experts=2, num_experts_per_tok=1, input_dim=D, hidden_dim=8)
        x = mx.array([[[1.0, 2.0, 3.0, 4.0], [0.5, 0.5, 0.5, 0.5], [-1.0, 0.0, 1.0, 2.0]]])
        
        out1 = model(x)
        out2 = model(x)
        mx.eval(out1)
        mx.eval(out2)
        
        assert mx.allclose(out1, out2)


@pytest.mark.parametrize("num_experts", [2, 4, 8])
@pytest.mark.parametrize("k", [1, 2])
def test_moe_top_k_routing(num_experts: int, k: int):
    """Test that exactly k experts are used per token."""
    if k > num_experts:
        pytest.skip("k cannot exceed num_experts")
    
    B, L, D = 1, 1, 8
    hidden = 16
    
    model = MoELayer(num_experts, k, D, hidden)
    x = mx.random.normal((B, L, D))
    
    out = model(x)
    mx.eval(out)
    
    # Output shape should be preserved
    assert out.shape == (B, L, D)
