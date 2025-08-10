from typing import List
import mlx.core as mx
import mlx.nn as nn

class Expert(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        # TODO: implement
        pass

    def __call__(self, x: mx.array) -> mx.array:
        # TODO: implement
        pass

class Gating(nn.Module):
    def __init__(self, d_model: int, n_experts: int):
        super().__init__()
        # TODO: implement
        pass

    def __call__(self, x: mx.array) -> mx.array:
        # TODO: implement
        pass

class MoE(nn.Module):
    def __init__(self, d_model: int, n_experts: int, d_ff: int, top_k: int):
        super().__init__()
        # TODO: implement
        pass

    def __call__(self, x: mx.array) -> mx.array:
        # TODO: implement
        pass
