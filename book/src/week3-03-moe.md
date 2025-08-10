# Week 3 Day 3: Mixture of Experts (MoE)

Mixture of Experts (MoE) is a neural architecture that dramatically scales model capacity while keeping computation costs manageable. Instead of using all parameters for every token, MoE models activate only a subset of "expert" networks, enabling massive models with efficient inference.

## MoE Architecture Overview

**Core Concept**: Replace dense MLP layers with multiple expert networks and a gating mechanism:

```
Input Token
     ↓
Gate Network (routing)
     ↓
[Expert 1] [Expert 2] [Expert 3] [Expert 4] [Expert 5] [Expert 6] [Expert 7] [Expert 8]
     ↓           ↓
  Selected    Selected
   Experts     Experts
     ↓           ↓
   Weight    Combine
    ↓
  Output
```

**Key Properties**:
- **Sparse Activation**: Only 2-4 experts active per token
- **Specialized Experts**: Each expert learns different patterns  
- **Scalable**: Add experts without increasing per-token compute
- **Load Balancing**: Ensure experts are used evenly

**Readings**

- [Switch Transformer Paper](https://arxiv.org/abs/2101.03961)
- [GLaM: Generalist Language Model](https://arxiv.org/abs/2112.06905)
- [Mixtral 8x7B](https://arxiv.org/abs/2401.04088)

## Task 1: Implement Expert Networks

Create the basic expert architecture:

```python
class Expert:
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        """
        TODO: Initialize a single expert network
        - Create MLP with gate, up, and down projections
        - Use same architecture as standard transformer MLP
        """
        pass
    
    def __call__(self, x: mx.array) -> mx.array:
        """TODO: Forward pass through expert"""
        pass

class MoELayer:
    def __init__(self, 
                 input_dim: int, 
                 hidden_dim: int, 
                 num_experts: int = 8,
                 top_k: int = 2):
        """
        TODO: Initialize MoE layer
        - Create num_experts Expert networks
        - Create gating network
        - Set up load balancing
        """
        pass
```

## Task 2: Gating and Expert Selection

Implement the gating mechanism:

```python
def compute_expert_routing(x: mx.array, gate_weights: mx.array, 
                          top_k: int) -> tuple[mx.array, mx.array]:
    """
    TODO: Implement expert routing
    - Compute gate logits for each expert
    - Select top-k experts per token
    - Apply softmax to get mixing weights
    - Return (expert_indices, expert_weights)
    """
    pass

def load_balancing_loss(gate_logits: mx.array, expert_indices: mx.array) -> mx.array:
    """
    TODO: Implement load balancing loss
    - Encourage even distribution across experts
    - Penalize experts that are over/under-utilized
    """
    pass
```

## Task 3: Efficient Expert Computation

Implement batched expert computation:

```python
def batched_expert_forward(tokens: mx.array, 
                          expert_indices: mx.array,
                          expert_weights: mx.array,
                          experts: list[Expert]) -> mx.array:
    """
    TODO: Efficient batched expert computation
    - Group tokens by selected experts
    - Batch computation within each expert
    - Combine results using expert weights
    - Handle tokens that select the same experts
    """
    pass
```

## Task 4: MoE Integration with Qwen

Replace MLP layers with MoE:

```python
class Qwen3MoETransformerBlock:
    def __init__(self, 
                 hidden_size: int,
                 num_experts: int = 8,
                 top_k: int = 2,
                 **kwargs):
        """
        TODO: Create transformer block with MoE
        - Keep attention layer unchanged
        - Replace MLP with MoE layer
        - Add load balancing loss tracking
        """
        pass
    
    def __call__(self, x: mx.array, offset: int, cache, mask=None):
        """TODO: Forward pass with MoE computation"""
        pass
```

{{#include copyright.md}}
