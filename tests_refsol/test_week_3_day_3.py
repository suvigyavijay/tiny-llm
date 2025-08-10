import pytest
import mlx.core as mx
from tiny_llm_ref.moe import MoE


class TestMoE:
    @pytest.mark.parametrize("hidden_dim", [64, 128])
    @pytest.mark.parametrize("num_experts", [4, 8])
    @pytest.mark.parametrize("num_experts_per_tok", [1, 2])
    def test_moe_creation(self, hidden_dim, num_experts, num_experts_per_tok):
        moe = MoE(hidden_dim=hidden_dim, intermediate_dim=hidden_dim * 2, num_experts=num_experts, num_experts_per_tok=num_experts_per_tok)
        assert len(moe.experts) == num_experts
        assert moe.num_experts_per_tok == num_experts_per_tok

    @pytest.mark.parametrize("batch_size", [1, 2])
    @pytest.mark.parametrize("seq_len", [10, 20])
    def test_moe_forward_pass(self, batch_size, seq_len):
        moe = MoE(hidden_dim=128, intermediate_dim=256, num_experts=8, num_experts_per_tok=2)
        x = mx.random.uniform(shape=(batch_size, seq_len, 128))
        out = moe(x)
        assert out.shape == x.shape
        assert out.dtype == x.dtype

    def test_gating_weights_sum_to_one(self):
        moe = MoE(hidden_dim=128, intermediate_dim=256, num_experts=8, num_experts_per_tok=2)
        x = mx.random.uniform(shape=(1, 10, 128))
        gating_weights = moe.gating(x)
        assert mx.allclose(mx.sum(gating_weights, axis=-1), mx.ones_like(mx.sum(gating_weights, axis=-1)), atol=1e-5)

    def test_moe_empty_input(self):
        moe = MoE(hidden_dim=128, intermediate_dim=256, num_experts=8, num_experts_per_tok=2)
        x = mx.zeros((1, 0, 128))
        out = moe(x)
        assert out.shape == x.shape

    def test_moe_input_ranks(self):
        moe = MoE(hidden_dim=128, intermediate_dim=256, num_experts=8, num_experts_per_tok=2)
        
        # 2D input
        x_2d = mx.random.uniform(shape=(10, 128))
        out_2d = moe(x_2d)
        assert out_2d.shape == x_2d.shape
        
        # 3D input is already tested
        
        # 4D input
        x_4d = mx.random.uniform(shape=(1, 2, 5, 128))
        out_4d = moe(x_4d)
        assert out_4d.shape == x_4d.shape
