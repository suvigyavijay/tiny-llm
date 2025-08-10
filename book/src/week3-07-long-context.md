# Week 3 Day 7: Long Context

Handling long contexts is a major challenge in LLM research and engineering. As input sequences grow longer, the computational and memory requirements of the standard attention mechanism become prohibitive. In this final chapter, we will explore one of the techniques for enabling LLMs to handle long contexts: sliding window attention.

Sliding window attention is a simple but effective technique that limits the attention computation to a fixed-size window around each token. This significantly reduces the computational complexity and memory usage, allowing the model to handle much longer sequences.

[📚 Reading: Long-Context Language Modeling with Parallel Context Encoding](https://arxiv.org/abs/2207.03170)

## Task 1: Implement Sliding Window Attention

Your task is to implement the `sliding_window_attention` function.

```
src/tiny_llm/long_context.py
```

The implementation will involve a loop that iterates over each token in the sequence. For each token, you will:
1. Define a window of a fixed size around the current token.
2. Perform the attention computation only within that window.

This simplified implementation will give you a basic understanding of how sliding window attention works. In a real-world application, you would use more efficient implementations that avoid the explicit loop and leverage parallel computation.

You can run the following tests to verify your implementation:

```
pdm run test --week 3 --day 7
```

Congratulations on completing the tiny-llm course! You have built a sophisticated LLM inference engine from scratch and have learned about some of the most advanced techniques used in modern LLM serving. We hope this course has provided you with a solid foundation for your future work in this exciting field.

{{#include copyright.md}}
