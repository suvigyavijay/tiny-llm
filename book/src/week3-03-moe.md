# Week 3 Day 3: Mixture of Experts (MoE)

In this chapter, we will implement a **Mixture of Experts (MoE)** layer. We will build a router that dynamically selects the best experts for each token, allowing us to scale up model parameters without exploding inference cost.

Mixture of Experts models scale parameter count massively while keeping active parameters per token low.

**📚 Readings**

- [Mixture of Experts Explained - Hugging Face](https://huggingface.co/blog/moe)
- [Mixtral of Experts - Mistral AI Blog](https://mistral.ai/news/mixtral-of-experts/)

## Motivation

In a dense Transformer (like Llama 2), every token uses 100% of the model parameters. To make the model "smarter" (larger), we usually increase depth or width, which increases inference cost quadratically or linearly.
MoE offers a way to scale parameters **without** scaling compute proportionally.

## The MoE Layer

An MoE layer replaces the standard MLP (FeedForward) layer in a Transformer.
Instead of one big MLP, we have `N` experts (smaller MLPs).
For each token, a **Router** (or Gating Network) decides which experts process it.

### Top-K Gating

We typically select the top-K experts (e.g., K=2) with the highest router scores.

$$
h = \sum_{i \in TopK} w_i \cdot E_i(x)
$$

where $w_i$ are the softmax-normalized routing weights.

### Shapes

- Input `x`: `[Batch, Seq_Len, Hidden_Dim]`
- Router weights `Wg`: `[Hidden_Dim, Num_Experts]`
- Expert weights: `[Num_Experts, Hidden_Dim, Intermediate_Dim]` etc.

## Task 1: MoE Router

```
src/tiny_llm/moe.py
```

Implement `MoELayer`.
1. Compute routing logits: `x @ Wg`.
2. Select top-K indices and weights.
3. Normalize weights (softmax).
4. Route input to experts.
5. Combine outputs.

*Note*: In optimized frameworks, we use sparse kernels. Here, we might use a masked approach or loop for simplicity.

### Code Walkthrough

```python
class MoELayer(nn.Module):
    def __call__(self, x):
        # 1. Router
        gate_logits = self.gate(x) # [B, L, Num_Experts]
        
        # 2. Select Experts
        # Use argpartition or topk to find indices of top K experts
        
        # 3. Dispatch
        # For a simple implementation, you can iterate over experts and apply masks
        pass
```

## Testing

```bash
pdm run test-refsol tests_refsol/test_week_3_day_3.py
```

{{#include copyright.md}}
