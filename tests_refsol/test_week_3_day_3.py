import pytest
import mlx.core as mx
from tiny_llm_ref.moe import MoELayer


def test_moe_forward_shape():
    """Test MoE layer produces correct output shape."""
    B, L, D = 2, 5, 8
    num_experts = 4
    k = 2
    hidden = 16
    
    model = MoELayer(num_experts, k, D, hidden)
    x = mx.random.normal((B, L, D))
    
    out = model(x)
    mx.eval(out)
    
    assert out.shape == (B, L, D)


def test_moe_deterministic():
    """Test MoE layer is deterministic given same input."""
    B, L, D = 1, 3, 4
    
    model = MoELayer(num_experts=2, num_experts_per_tok=1, input_dim=D, hidden_dim=8)
    x = mx.array([[[1.0, 2.0, 3.0, 4.0], [0.5, 0.5, 0.5, 0.5], [-1.0, 0.0, 1.0, 2.0]]])
    
    out1 = model(x)
    out2 = model(x)
    mx.eval(out1)
    mx.eval(out2)
    
    assert mx.allclose(out1, out2)
