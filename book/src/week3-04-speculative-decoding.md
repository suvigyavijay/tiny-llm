# Week 3 Day 4: Speculative Decoding

In this chapter, we will implement **Speculative Decoding**, a technique to accelerate inference. We will use a small draft model to generate candidate tokens and a large target model to verify them in parallel.

Speculative Decoding exploits the fact that small models are faster than large models, and many tokens are "easy" to predict.

**📚 Readings**

- [Unlocking the Power of Speculative Decoding - Hugging Face](https://huggingface.co/blog/speculative-decoding)
- [Speculative Sampling - Jay Mody](https://jaykmody.com/blog/speculative-sampling/)

## Intuition

Generating tokens is expensive because we have to load the huge model weights for *every single token*.
However, checking 5 tokens at once costs almost the same as checking 1 token, because GPUs are parallel.
If we can guess the next 5 tokens cheaply (using a tiny model), we can validate them in one go with the big model.

## The Algorithm

1.  **Draft**: Run the small model for `K` steps to produce `[t1, t2, ... tK]`.
2.  **Verify**: Run the large model on the sequence `[prefix, t1, ... tK]` in *one forward pass*.
3.  **Accept/Reject**:
    - For each token `ti`, compare the probability $P_{large}(t_i)$ with $P_{small}(t_i)$.
    - If accepted, keep it.
    - If rejected, stop and sample a new token from the large model's distribution.

## Task 1: Speculate & Verify

```
src/tiny_llm/speculative.py
```

Implement `speculative_decode(target_model, draft_model, prompt, k=4)`.

### Implementation Tips

- **Drafting**: Use a simple loop with `draft_model`. Store tokens.
- **Verification**: Concat `prompt` + `draft_tokens`. Run `target_model`.
- **Validation**: Compare `argmax` of target logits at position `i` with `draft_token[i]`. (Greedy version).

## Testing

```bash
pdm run test-refsol tests_refsol/test_week_3_day_4.py
```

{{#include copyright.md}}
