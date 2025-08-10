"""
Tests for Week 3, Day 3: Mixture of Experts (MoE) implementation.

Tests the MoE architecture including expert routing, load balancing,
and sparse computation efficiency.
"""

import pytest
import mlx.core as mx
import time
from src.tiny_llm_ref.mixture_of_experts import (
    Expert, MoELayer, MoETransformerBlock, compute_expert_routing,
    load_balancing_loss, batched_expert_forward, analyze_moe_efficiency
)
from src.tiny_llm_ref.quantize import QuantizedWeights


class TestExpert:
    """Test individual Expert networks."""
    
    def test_expert_creation(self):
        """Test creating an expert network."""
        # Create dummy quantized weights
        input_dim, hidden_dim, output_dim = 128, 256, 128
        
        w_gate = QuantizedWeights(
            scales=mx.ones((hidden_dim // 32, 1)),
            biases=mx.zeros((hidden_dim // 32, 1)),
            group_size=32,
            bits=4,
            weight=mx.random.uniform(0, 15, (hidden_dim, input_dim // 2), dtype=mx.uint8)
        )
        
        w_up = QuantizedWeights(
            scales=mx.ones((hidden_dim // 32, 1)),
            biases=mx.zeros((hidden_dim // 32, 1)),
            group_size=32,
            bits=4,
            weight=mx.random.uniform(0, 15, (hidden_dim, input_dim // 2), dtype=mx.uint8)
        )
        
        w_down = QuantizedWeights(
            scales=mx.ones((output_dim // 32, 1)),
            biases=mx.zeros((output_dim // 32, 1)),
            group_size=32,
            bits=4,
            weight=mx.random.uniform(0, 15, (output_dim, hidden_dim // 2), dtype=mx.uint8)
        )
        
        expert = Expert(input_dim, hidden_dim, output_dim, w_gate, w_up, w_down)
        
        assert expert.input_dim == input_dim
        assert expert.hidden_dim == hidden_dim
        assert expert.output_dim == output_dim
    
    def test_expert_forward(self):
        """Test expert forward pass."""
        input_dim, hidden_dim, output_dim = 64, 128, 64
        
        # Create simplified expert with identity-like weights for testing
        w_gate = QuantizedWeights(
            scales=mx.ones((hidden_dim // 32, 1)),
            biases=mx.zeros((hidden_dim // 32, 1)),
            group_size=32,
            bits=4,
            weight=mx.random.uniform(0, 15, (hidden_dim, input_dim // 2), dtype=mx.uint8)
        )
        
        w_up = QuantizedWeights(
            scales=mx.ones((hidden_dim // 32, 1)),
            biases=mx.zeros((hidden_dim // 32, 1)),
            group_size=32,
            bits=4,
            weight=mx.random.uniform(0, 15, (hidden_dim, input_dim // 2), dtype=mx.uint8)
        )
        
        w_down = QuantizedWeights(
            scales=mx.ones((output_dim // 32, 1)),
            biases=mx.zeros((output_dim // 32, 1)),
            group_size=32,
            bits=4,
            weight=mx.random.uniform(0, 15, (output_dim, hidden_dim // 2), dtype=mx.uint8)
        )
        
        expert = Expert(input_dim, hidden_dim, output_dim, w_gate, w_up, w_down)
        
        # Test forward pass
        batch_size, seq_len = 4, 16
        x = mx.random.normal((batch_size, seq_len, input_dim))
        
        output = expert(x)
        
        assert output.shape == (batch_size, seq_len, output_dim)
        assert not mx.isnan(output).any()


class TestExpertRouting:
    """Test expert routing mechanisms."""
    
    def test_compute_expert_routing(self):
        """Test expert routing computation."""
        batch_size, seq_len, hidden_dim = 2, 8, 64
        num_experts, top_k = 6, 2
        
        x = mx.random.normal((batch_size, seq_len, hidden_dim))
        gate_weights = mx.random.normal((hidden_dim, num_experts)) * 0.1
        
        expert_indices, expert_weights = compute_expert_routing(x, gate_weights, top_k)
        
        assert expert_indices.shape == (batch_size, seq_len, top_k)
        assert expert_weights.shape == (batch_size, seq_len, top_k)
        
        # Check that indices are valid
        assert mx.all(expert_indices >= 0)
        assert mx.all(expert_indices < num_experts)
        
        # Check that weights are normalized (sum to 1 for each token)
        weight_sums = mx.sum(expert_weights, axis=-1)
        assert mx.allclose(weight_sums, 1.0, atol=1e-5)
    
    def test_expert_routing_deterministic(self):
        """Test that expert routing is deterministic."""
        batch_size, seq_len, hidden_dim = 1, 4, 32
        num_experts, top_k = 4, 2
        
        x = mx.random.normal((batch_size, seq_len, hidden_dim))
        gate_weights = mx.random.normal((hidden_dim, num_experts)) * 0.1
        
        # Compute routing twice
        indices1, weights1 = compute_expert_routing(x, gate_weights, top_k)
        indices2, weights2 = compute_expert_routing(x, gate_weights, top_k)
        
        assert mx.array_equal(indices1, indices2)
        assert mx.allclose(weights1, weights2)
    
    def test_load_balancing_loss(self):
        """Test load balancing loss computation."""
        batch_size, seq_len, num_experts = 4, 8, 6
        top_k = 2
        
        # Create gate logits
        gate_logits = mx.random.normal((batch_size, seq_len, num_experts))
        
        # Create expert indices (simulate routing)
        expert_indices = mx.random.randint(0, num_experts, (batch_size, seq_len, top_k))
        
        loss = load_balancing_loss(gate_logits, expert_indices, num_experts)
        
        assert loss.shape == ()  # Scalar loss
        assert loss >= 0  # Loss should be non-negative
    
    def test_load_balancing_uniform_case(self):
        """Test load balancing loss with perfectly uniform distribution."""
        batch_size, seq_len, num_experts = 4, 12, 3
        top_k = 1
        
        # Create perfectly uniform gate logits
        gate_logits = mx.zeros((batch_size, seq_len, num_experts))
        
        # Create perfectly balanced expert assignment
        expert_indices = mx.zeros((batch_size, seq_len, top_k))
        for i in range(batch_size):
            for j in range(seq_len):
                expert_indices = expert_indices.at[i, j, 0].set((i * seq_len + j) % num_experts)
        
        loss = load_balancing_loss(gate_logits, expert_indices, num_experts)
        
        # Should be close to optimal (num_experts for uniform distribution)
        assert mx.allclose(loss, float(num_experts), atol=0.1)


class TestBatchedExpertForward:
    """Test batched expert computation."""
    
    def test_batched_expert_forward_basic(self):
        """Test basic batched expert forward pass."""
        batch_size, seq_len, hidden_dim = 2, 8, 64
        num_experts, top_k = 4, 2
        
        # Create dummy experts
        experts = []
        for i in range(num_experts):
            w_gate = QuantizedWeights(
                scales=mx.ones((hidden_dim // 32, 1)),
                biases=mx.zeros((hidden_dim // 32, 1)),
                group_size=32,
                bits=4,
                weight=mx.random.uniform(0, 15, (hidden_dim, hidden_dim // 2), dtype=mx.uint8)
            )
            expert = Expert(hidden_dim, hidden_dim, hidden_dim, w_gate, w_gate, w_gate)
            experts.append(expert)
        
        # Create input and routing decisions
        tokens = mx.random.normal((batch_size, seq_len, hidden_dim))
        expert_indices = mx.random.randint(0, num_experts, (batch_size, seq_len, top_k))
        expert_weights = mx.softmax(mx.random.normal((batch_size, seq_len, top_k)), axis=-1)
        
        output = batched_expert_forward(tokens, expert_indices, expert_weights, experts)
        
        assert output.shape == (batch_size, seq_len, hidden_dim)
        assert not mx.isnan(output).any()
    
    def test_single_expert_selection(self):
        """Test case where all tokens select the same expert."""
        batch_size, seq_len, hidden_dim = 2, 4, 32
        num_experts, top_k = 3, 1
        
        # Create experts with different behaviors
        experts = []
        for i in range(num_experts):
            w_gate = QuantizedWeights(
                scales=mx.ones((hidden_dim // 32, 1)) * (i + 1),  # Different scales
                biases=mx.zeros((hidden_dim // 32, 1)),
                group_size=32,
                bits=4,
                weight=mx.random.uniform(0, 15, (hidden_dim, hidden_dim // 2), dtype=mx.uint8)
            )
            expert = Expert(hidden_dim, hidden_dim, hidden_dim, w_gate, w_gate, w_gate)
            experts.append(expert)
        
        tokens = mx.random.normal((batch_size, seq_len, hidden_dim))
        
        # All tokens select expert 1
        expert_indices = mx.ones((batch_size, seq_len, top_k), dtype=mx.int32)
        expert_weights = mx.ones((batch_size, seq_len, top_k))
        
        output = batched_expert_forward(tokens, expert_indices, expert_weights, experts)
        
        assert output.shape == (batch_size, seq_len, hidden_dim)


class TestMoELayer:
    """Test complete MoE layer."""
    
    def test_moe_layer_creation(self):
        """Test MoE layer initialization."""
        input_dim, hidden_dim = 128, 256
        num_experts, top_k = 8, 2
        
        moe = MoELayer(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_experts=num_experts,
            top_k=top_k
        )
        
        assert moe.input_dim == input_dim
        assert moe.hidden_dim == hidden_dim
        assert moe.num_experts == num_experts
        assert moe.top_k == top_k
        assert len(moe.experts) == num_experts
    
    def test_moe_layer_forward(self):
        """Test MoE layer forward pass."""
        input_dim, hidden_dim = 64, 128
        num_experts, top_k = 4, 2
        batch_size, seq_len = 2, 8
        
        moe = MoELayer(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_experts=num_experts,
            top_k=top_k
        )
        
        x = mx.random.normal((batch_size, seq_len, input_dim))
        output = moe(x)
        
        assert output.shape == (batch_size, seq_len, input_dim)
        assert not mx.isnan(output).any()
        
        # Should have computed load balancing loss
        load_loss = moe.get_load_balancing_loss()
        assert load_loss >= 0
    
    def test_expert_utilization_tracking(self):
        """Test expert utilization computation."""
        input_dim, hidden_dim = 32, 64
        num_experts, top_k = 4, 1
        
        moe = MoELayer(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_experts=num_experts,
            top_k=top_k
        )
        
        x = mx.random.normal((2, 8, input_dim))
        utilization = moe.get_expert_utilization(x)
        
        assert utilization.shape == (num_experts,)
        assert mx.sum(utilization) <= 1.0 + 1e-5  # Should sum to <= 1 (with floating point tolerance)
        assert mx.all(utilization >= 0)


class TestMoETransformerBlock:
    """Test MoE transformer block."""
    
    def test_moe_transformer_block_creation(self):
        """Test MoE transformer block initialization."""
        hidden_size = 128
        num_heads, num_kv_heads = 8, 4
        num_experts, expert_top_k = 6, 2
        
        block = MoETransformerBlock(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            num_experts=num_experts,
            expert_top_k=expert_top_k
        )
        
        assert block.hidden_size == hidden_size
        assert block.num_experts == num_experts
        assert block.expert_top_k == expert_top_k
    
    def test_moe_transformer_forward_without_attention(self):
        """Test MoE transformer block forward pass without attention layer."""
        hidden_size = 64
        num_experts, expert_top_k = 4, 2
        
        block = MoETransformerBlock(
            hidden_size=hidden_size,
            num_heads=4,
            num_kv_heads=2,
            num_experts=num_experts,
            expert_top_k=expert_top_k,
            attention_layer=None,  # No attention layer
            norm_layer=None       # No normalization
        )
        
        batch_size, seq_len = 2, 8
        x = mx.random.normal((batch_size, seq_len, hidden_size))
        
        output = block(x)
        
        assert output.shape == (batch_size, seq_len, hidden_size)
        assert not mx.isnan(output).any()
        
        # Should have load balancing loss
        load_loss = block.get_load_balancing_loss()
        assert load_loss >= 0
    
    def test_expert_utilization_history(self):
        """Test expert utilization tracking over time."""
        hidden_size = 32
        num_experts = 4
        
        block = MoETransformerBlock(
            hidden_size=hidden_size,
            num_heads=4,
            num_kv_heads=2,
            num_experts=num_experts,
            expert_top_k=1,
            attention_layer=None,
            norm_layer=None
        )
        
        # Run multiple forward passes
        for _ in range(5):
            x = mx.random.normal((1, 4, hidden_size))
            block(x)
        
        # Check utilization history
        avg_utilization = block.get_average_expert_utilization()
        assert avg_utilization.shape == (num_experts,)
        
        # Reset tracking
        block.reset_utilization_tracking()
        assert len(block.expert_utilization_history) == 0


class TestMoEEfficiency:
    """Test MoE efficiency analysis."""
    
    def test_moe_efficiency_analysis(self):
        """Test MoE efficiency computation."""
        input_dim, hidden_dim = 64, 128
        num_experts, top_k = 8, 2
        
        moe = MoELayer(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_experts=num_experts,
            top_k=top_k
        )
        
        # Create test inputs
        test_inputs = [
            mx.random.normal((2, 8, input_dim)),
            mx.random.normal((1, 16, input_dim)),
            mx.random.normal((3, 4, input_dim))
        ]
        
        efficiency = analyze_moe_efficiency(moe, test_inputs)
        
        assert "computational_efficiency" in efficiency
        assert "expert_utilization" in efficiency
        assert "utilization_variance" in efficiency
        assert "most_used_expert" in efficiency
        assert "least_used_expert" in efficiency
        assert "effective_experts" in efficiency
        assert "load_imbalance" in efficiency
        
        # Computational efficiency should be > 1 (MoE should be more efficient)
        assert efficiency["computational_efficiency"] > 1.0
        
        # Expert utilization should be a valid distribution
        utilization = efficiency["expert_utilization"]
        assert utilization.shape == (num_experts,)
        assert mx.all(utilization >= 0)
    
    def test_moe_vs_dense_comparison(self):
        """Test MoE efficiency compared to dense equivalent."""
        input_dim = 64
        num_experts = 8
        top_k = 2
        
        # MoE layer
        moe = MoELayer(
            input_dim=input_dim,
            hidden_dim=input_dim * 2,
            num_experts=num_experts,
            top_k=top_k
        )
        
        test_input = mx.random.normal((4, 16, input_dim))
        
        # Time MoE forward pass
        start_time = time.time()
        moe_output = moe(test_input)
        moe_time = time.time() - start_time
        
        # Verify MoE output
        assert moe_output.shape == test_input.shape
        assert not mx.isnan(moe_output).any()
        
        # Efficiency metrics
        efficiency = analyze_moe_efficiency(moe, [test_input])
        
        # Should be computationally efficient
        assert efficiency["computational_efficiency"] > 1.0
        
        # Should use fewer than all experts effectively
        assert efficiency["effective_experts"] < num_experts


class TestMoEIntegration:
    """Integration tests for MoE system."""
    
    def test_moe_scaling(self):
        """Test MoE performance with different scales."""
        scales = [
            (32, 4, 2),   # Small
            (64, 8, 2),   # Medium
            (128, 16, 4)  # Large
        ]
        
        for input_dim, num_experts, top_k in scales:
            moe = MoELayer(
                input_dim=input_dim,
                hidden_dim=input_dim * 2,
                num_experts=num_experts,
                top_k=top_k
            )
            
            # Test with different batch sizes
            for batch_size in [1, 4, 8]:
                x = mx.random.normal((batch_size, 16, input_dim))
                output = moe(x)
                
                assert output.shape == x.shape
                assert not mx.isnan(output).any()
    
    def test_load_balancing_effectiveness(self):
        """Test that load balancing actually balances experts."""
        input_dim, num_experts = 64, 4
        
        moe = MoELayer(
            input_dim=input_dim,
            hidden_dim=input_dim,
            num_experts=num_experts,
            top_k=1  # Each token uses only one expert
        )
        
        # Generate diverse inputs to encourage balanced routing
        diverse_inputs = []
        for i in range(10):
            # Create inputs with different patterns
            x = mx.random.normal((2, 8, input_dim)) * (i + 1) * 0.5
            diverse_inputs.append(x)
        
        # Process all inputs and track utilization
        all_utilizations = []
        for x in diverse_inputs:
            moe(x)
            utilization = moe.get_expert_utilization(x)
            all_utilizations.append(utilization)
        
        # Average utilization across all inputs
        avg_utilization = mx.mean(mx.stack(all_utilizations), axis=0)
        
        # Check that utilization is reasonably balanced
        # (not perfect due to randomness, but should not be extremely skewed)
        utilization_variance = mx.var(avg_utilization)
        expected_uniform_variance = (1.0 / num_experts) ** 2 * (num_experts - 1) / num_experts
        
        # Utilization variance should be reasonable (not too much higher than uniform)
        assert utilization_variance < expected_uniform_variance * 10


if __name__ == "__main__":
    pytest.main([__file__])
