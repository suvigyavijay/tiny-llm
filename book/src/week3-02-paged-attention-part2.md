# Week 3 Day 2: Paged Attention - Part 2

In the previous chapter, we laid the groundwork for our paged attention implementation by creating the necessary data structures. Now, it's time to build the core of the paged attention algorithm: the custom attention kernel.

## Task 1: Implement the Paged Attention Kernel

Your main task in this chapter is to implement the paged attention kernel in Metal. This kernel will be responsible for performing the attention computation on the paged KV cache.

```
src/extensions/src/paged_attention.metal
```

The paged attention kernel will be more complex than a standard attention kernel. It will need to:
1. Receive the query, the paged KV cache, and the page table as input.
2. Use the page table to look up the physical memory addresses of the KV cache pages for each sequence.
3. Perform the attention computation, taking into account the non-contiguous memory layout of the paged cache.

## Task 2: Implement the C++ Launch Function

After implementing the Metal kernel, you will need to write the C++ function that launches it.

```
src/extensions/src/paged_attention.cpp
```

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
pdm run test --week 3 --day 2
```

{{#include copyright.md}}
