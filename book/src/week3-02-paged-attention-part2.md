# Week 3 Day 2: Paged Attention - Function

In the previous chapter, we laid the groundwork for our paged attention implementation by creating the necessary data structures. Now, it's time to build the core of the paged attention algorithm: the custom attention kernel.

## Task: Implement the Paged Attention Kernel

Your main task in this chapter is to implement the paged attention kernel in Metal and the C++ function that launches it. We have included tests for various sequence lengths, head dimensions, page sizes, and number of pages.

```
src/extensions/src/paged_attention.metal
src/extensions/src/paged_attention.cpp
```

The paged attention kernel is more complex than a standard attention kernel. It needs to:
1. Receive the query, the paged KV cache, and the page table as input.
2. Use the page table to look up the physical memory addresses of the KV cache pages for each sequence.
3. Perform the attention computation, taking into account the non-contiguous memory layout of the paged cache.

For a numerically stable softmax, the kernel uses a two-pass approach. The first pass finds the maximum score for each query-key dot product. The second pass computes the exponential of the scores, subtracts the max score, and then sums them up to get the denominator for the softmax. Finally, the attention weights are computed and multiplied by the value vectors.

The C++ function will be responsible for:
1. Setting up the Metal kernel and command encoder.
2. Creating and populating the page table.
3. Passing the query, KV cache, page table, and other necessary data to the kernel.
4. Dispatching the kernel for execution.

After implementing the C++ and Metal code, you need to rebuild the extension:

```
pdm run build-ext
```

You can run the following tests to verify your implementation:

```
pdm run test-refsol tests_refsol/test_week_3_day_2.py
```

{{#include copyright.md}}
