# Week 2 Day 4: Flash Attention 2 - CPU

Flash Attention is a groundbreaking algorithm that reorders the attention computation to be more memory-efficient. Instead of materializing the large attention matrix, Flash Attention processes the input in smaller blocks, or tiles, and uses an online softmax method to compute the final result. This significantly reduces the memory usage and can lead to substantial speedups, especially for long sequences.

In this chapter, we will implement a simplified version of Flash Attention 2 on the CPU. This will give you a deep understanding of the algorithm's mechanics before moving on to the GPU implementation.

[📚 Reading: FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691)

## Task 1: Implement Flash Attention on CPU

Your task is to implement the `FlashAttention::eval_cpu` function in C++. This function will perform the Flash Attention computation.

```
src/extensions/src/flash_attention.cpp
```

The core of the implementation will involve a nested loop that iterates over the blocks of the input matrices. For each block, you will:
1. Load blocks of Q, K, and V from memory.
2. Compute the attention scores for the block.
3. Apply the online softmax trick to update the output and the normalization statistics.
4. Store the final block output.

After implementing the C++ function, you need to rebuild the extension:

```
pdm run build-ext
```

You can run the following tests to verify your implementation:

```
pdm run test --week 2 --day 4
```

{{#include copyright.md}}
