import mlx.core as mx
import mlx.nn as nn
from typing import List

class Expert(nn.Module):
    def __init__(self, hidden_dim: int, intermediate_dim: int):
        super().__init__()
        self.w1 = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.w2 = nn.Linear(intermediate_dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, intermediate_dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.w2(nn.silu(self.w1(x)) * self.w3(x))

class Gating(nn.Module):
    def __init__(self, hidden_dim: int, num_experts: int):
        super().__init__()
        self.gate = nn.Linear(hidden_dim, num_experts, bias=False)

    def __call__(self, x: mx.array):
        return mx.softmax(self.gate(x), axis=-1)

class MoE(nn.Module):
    def __init__(self, hidden_dim: int, intermediate_dim: int, num_experts: int, num_experts_per_tok: int):
        super().__init__()
        self.experts = [Expert(hidden_dim, intermediate_dim) for _ in range(num_experts)]
        self.gating = Gating(hidden_dim, num_experts)
        self.num_experts_per_tok = num_experts_per_tok

    def __call__(self, x: mx.array) -> mx.array:
        gating_weights = self.gating(x)
        top_k_indices = mx.topk(gating_weights, self.num_experts_per_tok)
        
        # This is a simplified implementation. A real implementation would use
        # more efficient routing and dispatching mechanisms.
        
        final_output = mx.zeros_like(x)
        gating_weights_list = gating_weights.tolist()
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                token_output = mx.zeros_like(x[i, j])
                for k in range(self.num_experts_per_tok):
                    expert_idx = int(top_k_indices[i, j, k].item())
                    weight = gating_weights_list[i][j][expert_idx]
                    token_output += weight * self.experts[expert_idx](x[i, j])
                final_output[i, j] = token_output
                
        return final_output
