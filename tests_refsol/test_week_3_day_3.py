import pytest
import mlx.core as mx
from tiny_llm_ref.moe import MoE

class TestMoE:
    def test_moe_creation(self):
        moe = MoE(hidden_dim=128, intermediate_dim=256, num_experts=8, num_experts_per_tok=2)
        assert len(moe.experts) == 8
        assert moe.num_experts_per_tok == 2

    def test_moe_forward_pass(self):
        moe = MoE(hidden_dim=128, intermediate_dim=256, num_experts=8, num_experts_per_tok=2)
        x = mx.random.uniform(shape=(1, 10, 128))
        out = moe(x)
        assert out.shape == x.shape
        assert out.dtype == x.dtype

    def test_gating_weights(self):
        moe = MoE(hidden_dim=128, intermediate_dim=256, num_experts=8, num_experts_per_tok=2)
        x = mx.random.uniform(shape=(1, 10, 128))
        gating_weights = moe.gating(x)
        assert gating_weights.shape == (1, 10, 8)
        assert mx.all(mx.sum(gating_weights, axis=-1) > 0.99)
        assert mx.all(mx.sum(gating_weights, axis=-1) < 1.01)

    def test_top_k_experts(self):
        moe = MoE(hidden_dim=128, intermediate_dim=256, num_experts=8, num_experts_per_tok=2)
        x = mx.random.uniform(shape=(1, 10, 128))
        gating_weights = moe.gating(x)
        top_k_indices = mx.topk(gating_weights, moe.num_experts_per_tok)
        assert top_k_indices.shape == (1, 10, 2)
