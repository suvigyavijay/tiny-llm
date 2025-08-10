# Week 2 Day 6: Continuous Batching

In a production LLM serving system, it's crucial to handle multiple requests concurrently to maximize throughput. A naive approach would be to batch requests together and process them all at once. However, this can lead to low utilization, as the entire batch must wait for the slowest request to finish.

Continuous batching is a more advanced technique that addresses this issue. Instead of waiting for the entire batch to complete, continuous batching processes requests as they arrive and removes them from the batch as they finish. This allows the system to maintain a high level of concurrency and significantly improves throughput.

[📚 Reading: Optimizing LLM Serving with Continuous Batching](https://www.databricks.com/blog/llm-serving-simplified-nvidias-tensorrt-llm)

## Task 1: Implement Continuous Batching

In this task, you will implement the `batch_generate` function, which orchestrates the continuous batching process.

```
src/tiny_llm/batch.py
```

The `batch_generate` function will manage a queue of incoming requests and a batch of active requests. The main loop of the function will perform the following steps:
1. Prefill new requests from the queue.
2. Add the prefilled requests to the active batch.
3. Perform a single decoding step for all active requests.
4. Check for completed requests and remove them from the batch.

You will also need to implement the `Request` class, which encapsulates the state of a single request, and the `BatchingKvCache`, which manages the KV caches for the batched requests.

You can run the following tests to verify your implementation:

```
pdm run test --week 2 --day 6
```

{{#include copyright.md}}
