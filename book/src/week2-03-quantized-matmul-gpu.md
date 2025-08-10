# Week 2 Day 3: Quantized Matmul and Linear - GPU

In the previous chapter, we implemented quantized matrix multiplication on the CPU. Now, we will accelerate this operation by leveraging the power of the GPU. We will be using Metal, Apple's graphics and compute API, to write a custom kernel for this task.

[📚 Reading: Introduction to Metal for Developers](https://developer.apple.com/metal/)

## Task 1: Implement the Metal Kernel

The first step is to write the Metal kernel that will perform the quantized matrix multiplication.

```
src/extensions/src/quantized_matmul.metal
```

The Metal kernel will be similar in logic to the C++ CPU implementation, but it will be executed in parallel by many threads on the GPU. Each thread will be responsible for calculating a single element in the output matrix. The kernel will receive the quantized weights, scales, biases, and the input matrix as buffers.

Your task is to implement the `quantized_matmul_w4a16_g64` kernel. This kernel will dequantize the weights and perform the multiplication and accumulation in a loop.

## Task 2: Implement the GPU Evaluation Function

After writing the Metal kernel, you need to implement the C++ function that will launch this kernel.

```
src/extensions/src/quantized_matmul.cpp
```

You will need to implement the `QuantizedMatmul::eval_gpu` function. This function will:
1. Get the Metal device and command encoder.
2. Get the compiled Metal kernel from the library.
3. Set the input and output buffers for the kernel.
4. Define the grid and threadgroup sizes.
5. Dispatch the kernel for execution.

After implementing the C++ and Metal code, you need to rebuild the extension:

```
pdm run build-ext
```

You can run the following tests to verify your implementation:

```
pdm run test --week 2 --day 3
```

{{#include copyright.md}}
