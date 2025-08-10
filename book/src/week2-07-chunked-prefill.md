# Week 2 Day 7: Chunked Prefill

When dealing with very long prompts, processing the entire prompt at once can be inefficient and memory-intensive. Chunked prefill is a technique that addresses this by breaking down the prompt into smaller, more manageable chunks. This approach reduces the peak memory usage during the prefill stage and allows the system to start generating tokens even before the entire prompt has been processed.

[📚 Reading: Optimizing LLM Serving with Continuous Batching](https://www.databricks.com/blog/llm-serving-simplified-nvidias-tensorrt-llm)

## Task: Implement Chunked Prefill

In this task, you will implement the chunked prefill functionality within the `Request` class. We have included tests for various prompt lengths, single-step prefill, and empty prompts.

```
src/tiny_llm/batch.py
```

You will need to implement the `try_prefill` method. This method will process a chunk of the prompt in each call. The size of the chunk is determined by the `prefill_max_step` parameter. The method should keep track of the current position in the prompt and process the next chunk of tokens until the entire prompt has been consumed.

After implementing the `try_prefill` method, you can run the following tests to verify your implementation:

```
pdm run test-refsol tests_refsol/test_week_2_day_7.py
```

This concludes Week 2 of the course. You have now implemented a sophisticated LLM inference engine with several key optimizations, including KV caching, quantization, Flash Attention, and continuous batching. In Week 3, we will explore even more advanced topics, such as paged attention and mixture of experts.

{{#include copyright.md}}
