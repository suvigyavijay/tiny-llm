# Week 3 Day 7: Long Context

In this chapter, we will implement **Long Context** handling capabilities. We will use RoPE scaling techniques like Linear Interpolation to allow our model to process sequences longer than its training limit.

Scaling RoPE to handle longer sequences than trained on.

**📚 Readings**

- [RoPE Scaling Explained - EleutherAI Blog](https://blog.eleuther.ai/rope-scaling/)
- [Dynamically Scaled RoPE - Reddit (LocalLLaMA)](https://www.reddit.com/r/LocalLLaMA/comments/14mrgpr/dynamically_scaled_rope_further_increases/)

## Motivation

Models are trained with a fixed context window (e.g., 2048 tokens). RoPE rotates embeddings based on position `m` and frequency `theta`.
If `m > 2048`, the model sees "out of distribution" rotation angles.
Scaling techniques map the new range `[0, 4096]` back to the familiar `[0, 2048]`.

## Linear Scaling

If the model was trained with max position `2048`, and we want `4096`, we can scale positions by `0.5` (or scale frequencies by `1/0.5 = 2`).
`RoPE(x, pos) -> RoPE(x, pos * scale)` where `scale = L_train / L_target`.

## Task 1: RoPE Scaling

```
src/tiny_llm/long_context.py
```

Implement a helper to adjust RoPE frequencies.

### Code Walkthrough

```python
def apply_linear_scaling_rope(freqs, scale_factor):
    # freqs is the set of theta values (1/10000^(2i/d))
    # We want to slow down the rotation so that position 4096 acts like 2048.
    # Effectively: new_freqs = freqs / scale_factor
    pass
```

## Testing

```bash
pdm run test-refsol tests_refsol/test_week_3_day_7.py
```

{{#include copyright.md}}
