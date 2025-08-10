# Week 3 Day 1: Paged Attention - Part 1

Welcome to Week 3! In this week, we will explore some of the most advanced techniques used in modern LLM serving systems. We'll start with paged attention, a memory management technique that is at the heart of systems like vLLM.

Paged attention addresses the problem of memory fragmentation and inefficiency in traditional KV cache implementations. By dividing the cache into fixed-size pages, paged attention allows for more flexible and efficient memory management, enabling higher throughput and the ability to serve more concurrent requests.

[📚 Reading: PagedAttention for Large Language Models](https://vllm.ai/posts/2023-06-20-pagedattention.html)

## Task 1: Implement Paged Attention Data Structures

In this first part of the paged attention implementation, we will focus on the core data structures that will form the foundation of our memory manager.

```
src/tiny_llm/paged_attention.py
```

You will need to implement the following classes:
- **`Page`**: A simple class that represents a single page in the KV cache.
- **`PageTable`**: A class that maps a sequence to a list of pages.
- **`CacheManager`**: A class that manages the allocation and deallocation of pages.

The `CacheManager` will be responsible for keeping track of free pages, allocating new pages to sequences, and freeing pages when they are no longer needed.

You can run the following tests to verify your implementation:

```
pdm run test --week 3 --day 1
```

In the next chapter, we will build upon these data structures to implement the full paged attention algorithm.

{{#include copyright.md}}
