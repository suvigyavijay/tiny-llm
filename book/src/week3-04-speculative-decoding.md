# Week 3 Day 4: Speculative Decoding

Speculative decoding accelerates LLM inference by using a smaller, faster "draft" model to propose multiple tokens, which are then verified in parallel by the target model. This achieves significant speedups without any loss in output quality.

## Speculative Decoding Algorithm

**Core Idea**: Use a fast model to "guess ahead" and verify guesses in parallel:

```
Draft Model (Fast):     Target Model (Accurate):
Token 1 → Token 2 →     Verify [Token 1, Token 2, Token 3]
Token 3 → Token 4       in parallel ✓ ✓ ✗
                        Accept: Token 1, Token 2
                        Reject: Token 3, resample
```

**Key Benefits**:
- 2-3x speedup with no quality loss
- Maintains exact sampling distribution
- Works with any draft/target model pair
- No training required

**Readings**

- [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192)
- [SpecInfer: Accelerating Generative LLM Serving](https://arxiv.org/abs/2305.09781)

## Task 1: Draft Model Integration

Set up the draft model system:

```python
class SpeculativeDecoder:
    def __init__(self, 
                 draft_model,      # Fast, smaller model
                 target_model,     # Accurate, larger model  
                 lookahead: int = 4):
        """
        TODO: Initialize speculative decoder
        - Store both models
        - Set up shared tokenizer
        - Configure lookahead window
        """
        pass
    
    def draft_generation(self, prompt_tokens: mx.array, 
                        num_tokens: int) -> list[int]:
        """
        TODO: Generate draft tokens
        - Use draft model to generate num_tokens
        - Return proposed token sequence
        """
        pass
```

## Task 2: Parallel Verification

Implement efficient verification:

```python
def verify_speculation(draft_tokens: list[int],
                      target_logits: mx.array,
                      draft_logits: mx.array) -> tuple[int, list[int]]:
    """
    TODO: Verify speculative tokens
    - Compare target vs draft probabilities
    - Accept tokens that meet acceptance criteria
    - Return (num_accepted, corrected_tokens)
    
    Algorithm:
    1. For each draft token, compute acceptance probability
    2. Sample to decide accept/reject
    3. If rejected, resample from corrected distribution
    4. Return accepted prefix + correction
    """
    pass
```

## Task 3: Adaptive Speculation Length

Implement dynamic lookahead adjustment:

```python
class AdaptiveSpeculator:
    def __init__(self, initial_lookahead: int = 4):
        """TODO: Initialize adaptive speculation"""
        pass
    
    def update_lookahead(self, acceptance_rate: float):
        """
        TODO: Adjust lookahead based on success rate
        - Increase lookahead if acceptance rate high
        - Decrease if acceptance rate low
        - Track rolling average of acceptance
        """
        pass
```

## Task 4: Batched Speculative Decoding

Extend to batch processing:

```python
def batched_speculative_decode(batch_prompts: list[list[int]],
                              draft_model, target_model,
                              max_tokens: int = 100) -> list[list[int]]:
    """
    TODO: Implement batched speculative decoding
    - Handle different sequence lengths in batch
    - Optimize for different acceptance rates per sequence
    - Balance draft computation vs verification
    """
    pass
```

{{#include copyright.md}}
