"""
Mixture of Experts (MoE) implementation for sparse neural networks.

Implements the MoE architecture that enables scaling model capacity while keeping
computational costs manageable through sparse expert activation.
"""

import mlx.core as mx
from typing import List, Tuple, Optional
import math
from .basics import silu
from .quantize import QuantizedWeights, quantized_linear


class Expert:
    """A single expert network (MLP) in the MoE layer."""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, 
                 w_gate: QuantizedWeights, w_up: QuantizedWeights, w_down: QuantizedWeights):
        """
        Initialize a single expert network.
        
        Args:
            input_dim: Input dimension
            hidden_dim: Hidden dimension 
            output_dim: Output dimension
            w_gate: Quantized gate projection weights
            w_up: Quantized up projection weights  
            w_down: Quantized down projection weights
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.w_gate = w_gate
        self.w_up = w_up
        self.w_down = w_down
    
    def __call__(self, x: mx.array) -> mx.array:
        """
        Forward pass through expert network.
        
        Args:
            x: Input tensor [batch_size, seq_len, input_dim]
            
        Returns:
            Output tensor [batch_size, seq_len, output_dim]
        """
        # Standard MLP with SiLU activation: down(silu(gate(x)) * up(x))
        gate_out = quantized_linear(x, self.w_gate)
        up_out = quantized_linear(x, self.w_up)
        return quantized_linear(silu(gate_out) * up_out, self.w_down)


def compute_expert_routing(x: mx.array, gate_weights: mx.array, top_k: int) -> Tuple[mx.array, mx.array]:
    """
    Compute expert routing decisions for tokens.
    
    Args:
        x: Input tokens [batch_size, seq_len, hidden_dim]
        gate_weights: Gating network weights [hidden_dim, num_experts]
        top_k: Number of experts to select per token
        
    Returns:
        Tuple of (expert_indices, expert_weights)
        expert_indices: [batch_size, seq_len, top_k] - which experts to use
        expert_weights: [batch_size, seq_len, top_k] - mixing weights for experts
    """
    batch_size, seq_len, hidden_dim = x.shape
    num_experts = gate_weights.shape[1]
    
    # Compute gate logits for all experts
    gate_logits = mx.matmul(x, gate_weights)  # [batch_size, seq_len, num_experts]
    
    # Select top-k experts per token
    top_k_logits, expert_indices = mx.topk(gate_logits, k=top_k, axis=-1)
    
    # Apply softmax to get mixing weights
    expert_weights = mx.softmax(top_k_logits, axis=-1)
    
    return expert_indices, expert_weights


def load_balancing_loss(gate_logits: mx.array, expert_indices: mx.array, num_experts: int) -> mx.array:
    """
    Compute load balancing loss to encourage even expert utilization.
    
    Args:
        gate_logits: Raw gate logits [batch_size, seq_len, num_experts]
        expert_indices: Selected expert indices [batch_size, seq_len, top_k]
        num_experts: Total number of experts
        
    Returns:
        Load balancing loss scalar
    """
    batch_size, seq_len, _ = gate_logits.shape
    
    # Compute fraction of tokens assigned to each expert
    expert_counts = mx.zeros(num_experts)
    total_tokens = batch_size * seq_len
    
    for expert_id in range(num_experts):
        expert_mask = (expert_indices == expert_id)
        expert_counts = expert_counts.at[expert_id].set(mx.sum(expert_mask) / total_tokens)
    
    # Compute gate probabilities (average over all tokens)
    gate_probs = mx.mean(mx.softmax(gate_logits, axis=-1), axis=(0, 1))
    
    # Load balancing loss: encourage uniform distribution
    load_loss = num_experts * mx.sum(expert_counts * gate_probs)
    
    return load_loss


def batched_expert_forward(tokens: mx.array, 
                          expert_indices: mx.array,
                          expert_weights: mx.array,
                          experts: List[Expert]) -> mx.array:
    """
    Efficiently compute expert outputs by batching tokens per expert.
    
    Args:
        tokens: Input tokens [batch_size, seq_len, hidden_dim]
        expert_indices: Selected experts [batch_size, seq_len, top_k]
        expert_weights: Expert mixing weights [batch_size, seq_len, top_k]
        experts: List of expert networks
        
    Returns:
        Combined expert outputs [batch_size, seq_len, hidden_dim]
    """
    batch_size, seq_len, hidden_dim = tokens.shape
    top_k = expert_indices.shape[-1]
    num_experts = len(experts)
    
    # Initialize output
    output = mx.zeros_like(tokens)
    
    # Process each expert position (0 to top_k-1)
    for k in range(top_k):
        # Get expert IDs and weights for this position
        expert_ids = expert_indices[:, :, k]  # [batch_size, seq_len]
        weights = expert_weights[:, :, k]     # [batch_size, seq_len]
        
        # Process each expert
        for expert_id in range(num_experts):
            # Find tokens assigned to this expert
            expert_mask = (expert_ids == expert_id)
            
            if mx.sum(expert_mask) == 0:
                continue
            
            # Extract tokens for this expert
            expert_tokens = mx.where(expert_mask[:, :, None], tokens, 0)
            
            # Process through expert
            expert_output = experts[expert_id](expert_tokens)
            
            # Apply weights and accumulate
            weighted_output = expert_output * weights[:, :, None]
            masked_output = mx.where(expert_mask[:, :, None], weighted_output, 0)
            output = output + masked_output
    
    return output


class MoELayer:
    """Mixture of Experts layer that replaces standard MLP."""
    
    def __init__(self, 
                 input_dim: int, 
                 hidden_dim: int, 
                 num_experts: int = 8,
                 top_k: int = 2,
                 gate_weights: Optional[mx.array] = None,
                 expert_weights: Optional[List[Tuple[QuantizedWeights, QuantizedWeights, QuantizedWeights]]] = None):
        """
        Initialize MoE layer.
        
        Args:
            input_dim: Input dimension
            hidden_dim: Hidden dimension for experts
            num_experts: Number of expert networks
            top_k: Number of experts to activate per token
            gate_weights: Gating network weights [input_dim, num_experts]
            expert_weights: List of (w_gate, w_up, w_down) for each expert
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.top_k = top_k
        
        # Gating network
        if gate_weights is None:
            # Initialize random weights if not provided
            self.gate_weights = mx.random.normal((input_dim, num_experts)) * 0.02
        else:
            self.gate_weights = gate_weights
        
        # Expert networks
        self.experts = []
        if expert_weights is None:
            # Initialize random experts if not provided (for testing)
            for i in range(num_experts):
                # Create dummy quantized weights
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
                    scales=mx.ones((input_dim // 32, 1)),
                    biases=mx.zeros((input_dim // 32, 1)),
                    group_size=32,
                    bits=4,
                    weight=mx.random.uniform(0, 15, (input_dim, hidden_dim // 2), dtype=mx.uint8)
                )
                expert = Expert(input_dim, hidden_dim, input_dim, w_gate, w_up, w_down)
                self.experts.append(expert)
        else:
            for w_gate, w_up, w_down in expert_weights:
                expert = Expert(input_dim, hidden_dim, input_dim, w_gate, w_up, w_down)
                self.experts.append(expert)
        
        # Track load balancing loss
        self.last_load_loss = 0.0
    
    def __call__(self, x: mx.array) -> mx.array:
        """
        Forward pass through MoE layer.
        
        Args:
            x: Input tensor [batch_size, seq_len, input_dim]
            
        Returns:
            Output tensor [batch_size, seq_len, input_dim]
        """
        # Compute expert routing
        expert_indices, expert_weights = compute_expert_routing(x, self.gate_weights, self.top_k)
        
        # Compute load balancing loss (stored for training)
        gate_logits = mx.matmul(x, self.gate_weights)
        self.last_load_loss = load_balancing_loss(gate_logits, expert_indices, self.num_experts)
        
        # Compute expert outputs
        output = batched_expert_forward(x, expert_indices, expert_weights, self.experts)
        
        return output
    
    def get_load_balancing_loss(self) -> mx.array:
        """Get the load balancing loss from the last forward pass."""
        return self.last_load_loss
    
    def get_expert_utilization(self, x: mx.array) -> mx.array:
        """
        Compute expert utilization statistics.
        
        Args:
            x: Input tensor to analyze
            
        Returns:
            Expert utilization array [num_experts]
        """
        expert_indices, _ = compute_expert_routing(x, self.gate_weights, self.top_k)
        
        utilization = mx.zeros(self.num_experts)
        total_selections = expert_indices.size
        
        for expert_id in range(self.num_experts):
            expert_count = mx.sum(expert_indices == expert_id)
            utilization = utilization.at[expert_id].set(expert_count / total_selections)
        
        return utilization


class MoETransformerBlock:
    """Transformer block with MoE layer replacing the MLP."""
    
    def __init__(self,
                 hidden_size: int,
                 num_heads: int,
                 num_kv_heads: int,
                 num_experts: int = 8,
                 expert_top_k: int = 2,
                 intermediate_size: int = None,
                 attention_layer=None,
                 norm_layer=None,
                 **kwargs):
        """
        Initialize MoE transformer block.
        
        Args:
            hidden_size: Hidden dimension
            num_heads: Number of attention heads
            num_kv_heads: Number of key-value heads
            num_experts: Number of MoE experts
            expert_top_k: Number of experts to activate per token
            intermediate_size: Hidden size for expert MLPs
            attention_layer: Attention layer to use
            norm_layer: Normalization layer to use
        """
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.expert_top_k = expert_top_k
        
        if intermediate_size is None:
            intermediate_size = hidden_size * 4
        
        # Attention layer (same as standard transformer)
        self.self_attn = attention_layer
        
        # Replace MLP with MoE
        self.moe = MoELayer(
            input_dim=hidden_size,
            hidden_dim=intermediate_size,
            num_experts=num_experts,
            top_k=expert_top_k
        )
        
        # Normalization layers
        self.input_layernorm = norm_layer
        self.post_attention_layernorm = norm_layer
        
        # Track MoE statistics
        self.expert_utilization_history = []
    
    def __call__(self, 
                 x: mx.array, 
                 offset: int = None, 
                 cache=None, 
                 mask: mx.array = None) -> mx.array:
        """
        Forward pass through MoE transformer block.
        
        Args:
            x: Input tensor [batch_size, seq_len, hidden_size]
            offset: Sequence offset for attention
            cache: KV cache for attention
            mask: Attention mask
            
        Returns:
            Output tensor [batch_size, seq_len, hidden_size]
        """
        # Pre-attention normalization and self-attention with residual
        if self.self_attn is not None:
            attn_input = self.input_layernorm(x) if self.input_layernorm else x
            if cache is not None:
                attn_output = self.self_attn(attn_input, offset, cache, mask)
            else:
                attn_output = self.self_attn(attn_input, mask)
            x = x + attn_output
        
        # Pre-MoE normalization and MoE with residual
        moe_input = self.post_attention_layernorm(x) if self.post_attention_layernorm else x
        moe_output = self.moe(moe_input)
        x = x + moe_output
        
        # Track expert utilization for analysis
        utilization = self.moe.get_expert_utilization(moe_input)
        self.expert_utilization_history.append(utilization)
        
        return x
    
    def get_load_balancing_loss(self) -> mx.array:
        """Get load balancing loss from MoE layer."""
        return self.moe.get_load_balancing_loss()
    
    def get_average_expert_utilization(self) -> mx.array:
        """Get average expert utilization over recent forward passes."""
        if not self.expert_utilization_history:
            return mx.zeros(self.num_experts)
        
        # Average over last 100 forward passes
        recent_history = self.expert_utilization_history[-100:]
        return mx.mean(mx.stack(recent_history), axis=0)
    
    def reset_utilization_tracking(self):
        """Reset expert utilization tracking."""
        self.expert_utilization_history = []


def analyze_moe_efficiency(moe_layer: MoELayer, test_inputs: List[mx.array]) -> dict:
    """
    Analyze MoE efficiency and expert utilization.
    
    Args:
        moe_layer: MoE layer to analyze
        test_inputs: List of test input tensors
        
    Returns:
        Dictionary with efficiency statistics
    """
    total_flops_moe = 0
    total_flops_dense = 0
    expert_usage_counts = mx.zeros(moe_layer.num_experts)
    
    for test_input in test_inputs:
        batch_size, seq_len, hidden_dim = test_input.shape
        
        # Compute expert routing
        expert_indices, _ = compute_expert_routing(test_input, moe_layer.gate_weights, moe_layer.top_k)
        
        # Count expert usage
        for expert_id in range(moe_layer.num_experts):
            usage = mx.sum(expert_indices == expert_id)
            expert_usage_counts = expert_usage_counts.at[expert_id].add(usage)
        
        # Estimate FLOPs
        # MoE: Only top_k experts are active per token
        tokens = batch_size * seq_len
        flops_per_expert = 2 * hidden_dim * moe_layer.hidden_dim * 3  # gate, up, down projections
        moe_flops = tokens * moe_layer.top_k * flops_per_expert
        
        # Dense: All parameters active for all tokens
        dense_flops = tokens * moe_layer.num_experts * flops_per_expert
        
        total_flops_moe += moe_flops
        total_flops_dense += dense_flops
    
    # Normalize expert usage
    total_selections = mx.sum(expert_usage_counts)
    expert_utilization = expert_usage_counts / total_selections if total_selections > 0 else expert_usage_counts
    
    return {
        "computational_efficiency": total_flops_dense / total_flops_moe,
        "expert_utilization": expert_utilization,
        "utilization_variance": mx.var(expert_utilization),
        "most_used_expert": int(mx.argmax(expert_utilization)),
        "least_used_expert": int(mx.argmin(expert_utilization)),
        "effective_experts": mx.sum(expert_utilization > 0.01),  # Experts with >1% usage
        "load_imbalance": mx.max(expert_utilization) / mx.mean(expert_utilization)
    }
