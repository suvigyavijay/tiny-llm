# Week 3 Day 4: Speculative Decoding

Speculative decoding is an innovative technique that can significantly accelerate LLM inference. It works by using a small, fast "draft" model to generate a sequence of candidate tokens, and then using the larger, more powerful "target" model to validate these tokens in parallel. This approach reduces the number of sequential steps required for generation, leading to substantial speedups.

[📚 Reading: Speculative Decoding](https://huggingface.co/docs/transformers/main/en/llm_tutorial/faster_generation_speculative_decoding)

## Task: Implement Speculative Decoding

In this task, you will implement a simplified version of the speculative decoding algorithm. We have included tests for various acceptance and rejection scenarios, as well as for empty drafts.

```
src/tiny_llm/speculative.py
```

The implementation will involve the following steps:
1. Use the draft model to generate a sequence of candidate tokens.
2. Use the target model to get the probability distributions for the candidate tokens.
3. Compare the draft tokens with the target model's predictions to decide which tokens to accept.
4. For the first rejected token, sample a new token from the target model's distribution.

This simplified implementation will give you a solid understanding of the core concepts behind speculative decoding. In a real-world scenario, you would use more sophisticated sampling and acceptance strategies to maximize the performance gains.

You can run the following tests to verify your implementation:

```
pdm run test-refsol tests_refsol/test_week_3_day_4.py
```

{{#include copyright.md}}
