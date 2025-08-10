# Week 2 Day 5: Flash Attention 2 - GPU

Now that you have a solid understanding of the Flash Attention algorithm from the CPU implementation, it's time to unleash its full potential by running it on the GPU. In this chapter, you will implement the Flash Attention kernel in Metal and the corresponding C++ code to launch it.

## Task 1: Implement the Flash Attention Metal Kernel

Your first task is to implement the `flash_attention` kernel in Metal.

```
src/extensions/src/flash_attention.metal
```

The Metal kernel for Flash Attention is significantly more complex than the previous kernels we've written. It involves careful management of threadgroups, SIMD groups, and threadgroup memory to efficiently compute the attention in parallel.

The kernel will be structured as follows:
- Each threadgroup will be responsible for a block of the output matrix.
- Within a threadgroup, SIMD groups will be used to parallelize the computation of dot products and other operations.
- Threadgroup memory will be used to store blocks of Q, K, and V, reducing the number of reads from global memory.

## Task 2: Implement the GPU Evaluation Function

After writing the Metal kernel, you need to implement the C++ function that will launch it.

```
src/extensions/src/flash_attention.cpp
```

You will need to implement the `FlashAttention::eval_gpu` function. This function will be responsible for:
1. Setting up the Metal device, command encoder, and kernel.
2. Allocating and setting the input and output buffers.
3. Defining the threadgroup and grid dimensions based on the input shapes.
4. Launching the kernel.

After implementing the C++ and Metal code, you need to rebuild the extension:

```
pdm run build-ext
```

You can run the following tests to verify your implementation:

```
pdm run test --week 2 --day 5
```

{{#include copyright.md}}
