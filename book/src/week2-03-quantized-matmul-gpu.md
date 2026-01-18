# Week 2 Day 3: Quantized Matrix Multiplication (GPU)

Now that we have a CPU implementation, let's accelerate it using Metal on macOS GPUs. Writing high-performance GPU kernels requires understanding the hardware's parallel architecture.

**📚 Readings**

- [Performing Calculations on a GPU - Apple Developer](https://developer.apple.com/documentation/metal/performing_calculations_on_a_gpu)
- [Optimizing Matrix Multiplication on GPU - Siboehm](https://siboehm.com/articles/22/CUDA-MMM)
- [MLX: An Array Framework for Apple Silicon - Awni Hannun](https://awni.github.io/mlx/)

## Mental Model: The GPU Grid

Imagine the output matrix $Y$ of size $M \times N$. We launch a **grid** of threads, where each thread (or small group of threads) is responsible for computing one element (or one tile) of $Y$.

In Metal:
- **Grid**: The full problem space.
- **Threadgroup**: A block of threads that can share fast memory (Threadgroup Memory / Shared Memory) and synchronize.
- **Thread (SIMD Lane)**: The smallest unit of execution.

![GPU Grid Diagram](https://developer.apple.com/documentation/metal/performing_calculations_on_a_gpu/grid_of_threads_2x.png)
*(Conceptual diagram of grid decomposition)*

## Metal Kernel

We will implement the kernel in `src/extensions/src/quantized_matmul.metal`.

The kernel signature matches the data layout we used on CPU:

```cpp
[[kernel]] void quantized_matmul_w4a16_g64(
    device const T* scales [[buffer(0)]],
    device const T* biases [[buffer(1)]],
    device const T* a [[buffer(2)]],
    device const uint32_t* b [[buffer(3)]],
    device T* out [[buffer(4)]],
    // ... dims ...
)
```

## Task 1: Implement the Kernel

```
src/extensions/src/quantized_matmul.metal
```

### Threading Model

We will use a straightforward mapping for this kernel:
- Global X maps to `M` (rows of A).
- Global Y maps to `K` (rows of B / cols of Output).
- Each thread computes **one output element**.

### Algorithm

For each thread `(i, k)`:
1. Initialize `sum = 0`.
2. Iterate over the shared dimension `N` in chunks of `group_size`.
3. For each group:
    - Load `scale` and `bias`.
    - Iterate through the group.
    - Load packed `uint32` from `b`.
    - Unpack 8 weights.
    - Dequantize and multiply with corresponding elements from `a`.
    - Accumulate into `sum`.
4. Store `sum` to `out[i * K + k]`.

**Optimization Note**:
In a naive GPU implementation, every thread accessing `scale` and `bias` independently can cause memory contention.
The *Siboehm* article discusses "Memory Coalescing": threads in a group should access adjacent memory addresses.
Since our `b` matrix is packed, adjacent threads (handling different output columns) might access different parts of `b`.
However, because we use a "row-major" dispatch (or col-major depending on how you view it), ensure that threads in a SIMD group (warp) are reading contiguous chunks of memory where possible.

## Task 2: Host Code Dispatch

```
src/extensions/src/quantized_matmul.cpp
```

Update the `eval_gpu` function to dispatch this kernel.
1. Get the kernel function `quantized_matmul_w4a16_g64_f16` (or `bf16`).
2. Set input buffers.
3. Calculate grid dimensions:
   - `threadsPerThreadgroup`: e.g., `(32, 32, 1)`.
   - `threadgroups`: `(M/32, K/32, 1)`.
4. Dispatch.

## Testing

Run the tests again. This time, the GPU tests should pass.

```bash
pdm run test-refsol tests_refsol/test_week_2_day_2.py
```

{{#include copyright.md}}
