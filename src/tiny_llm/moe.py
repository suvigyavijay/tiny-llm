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
        pass
