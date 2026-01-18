import mlx.core as mx
import mlx.nn as nn


class MoELayer(nn.Module):
    def __init__(self, num_experts: int, num_experts_per_tok: int, input_dim: int, hidden_dim: int):
        super().__init__()
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.gate = nn.Linear(input_dim, num_experts, bias=False)
        self.experts = [
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, input_dim)
            ) for _ in range(num_experts)
        ]

    def __call__(self, x: mx.array) -> mx.array:
        B, L, D = x.shape
        k = self.num_experts_per_tok
        
        gate_logits = self.gate(x)
        
        # Top-K selection
        # When k == num_experts, use argsort instead (argpartition requires k < size)
        if k >= self.num_experts:
            indices = mx.argsort(-gate_logits, axis=-1)[..., :k]
        else:
            indices = mx.argpartition(-gate_logits, k, axis=-1)[..., :k]
        scores = mx.take_along_axis(gate_logits, indices, axis=-1)
        weights = mx.softmax(scores, axis=-1)
        
        # Simple implementation: run all experts and gather results
        # (Not sparse, but correct for small-scale testing)
        expert_outputs = mx.stack([e(x) for e in self.experts], axis=-1)  # [B, L, D, E]
        
        # Weighted sum of selected experts
        output = mx.zeros((B, L, D))
        for i in range(k):
            expert_idx = indices[..., i]  # [B, L]
            weight = weights[..., i:i+1]  # [B, L, 1]
            
            # Gather expert outputs for this choice
            for b in range(B):
                for l in range(L):
                    e = expert_idx[b, l].item()
                    output[b, l] += weight[b, l, 0] * expert_outputs[b, l, :, e]
        
        return output
