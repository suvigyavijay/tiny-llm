# Week 2 Day 3: Quantized Matrix Multiplication - GPU

Building on the CPU implementation from Day 2, we now optimize quantized matrix multiplication for GPU using Metal Shading Language. GPU optimization is crucial for practical LLM serving as it provides much higher memory bandwidth and parallel computation capability.

## GPU vs CPU: Architecture Differences

**CPU Characteristics**:
- Few cores (4-16), each very fast
- Large caches (MB per core)
- Complex instruction sets and branch prediction
- Optimized for sequential and complex tasks

**GPU Characteristics**: 
- Many cores (1000s), each simpler
- Smaller caches (KB per core)  
- SIMD (Single Instruction, Multiple Data) execution
- Optimized for parallel, data-intensive tasks

For matrix multiplication, GPU's massive parallelism typically wins despite slower individual cores.

## Metal Shading Language

Apple's Metal provides GPU compute capabilities through Metal Shading Language (MSL), similar to CUDA for NVIDIA GPUs.

**Key concepts**:
- **Kernels**: Functions that run on GPU
- **Thread groups**: Collections of threads that can share memory
- **SIMD groups**: Sets of threads that execute in lockstep (usually 32 threads)
- **Memory hierarchy**: Device memory → Threadgroup memory → Thread memory

**Readings**

- [Metal Programming Guide](https://developer.apple.com/metal/)
- [Metal Shading Language Specification](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf)
- [GPU Matrix Multiplication Optimization](https://siboehm.com/articles/22/CUDA-MMM)

## Task 1: Understanding the GPU Kernel

Examine the Metal kernel in `src/extensions_ref/src/quantized_matmul.metal`:

```metal
kernel void quantized_matmul_kernel(
    device const float* x [[buffer(0)]],
    device const uint8_t* w [[buffer(1)]],  // Packed 4-bit weights
    device const float* scales [[buffer(2)]],
    device const float* biases [[buffer(3)]],
    device float* out [[buffer(4)]],
    // ... more parameters
    uint2 grid_pos [[threadgroup_position_in_grid]],
    uint local_id [[thread_index_in_threadgroup]])
```

The kernel structure:
1. **Thread mapping**: Each thread computes part of the output matrix
2. **Memory coalescing**: Threads access memory in patterns that maximize bandwidth
3. **Shared memory**: Use threadgroup memory to cache frequently accessed data
4. **Quantization**: Dequantize weights on-the-fly during computation

## Task 2: Optimization Strategies

### Memory Coalescing
```metal
// Good: Consecutive threads access consecutive memory
for (uint i = local_id; i < N; i += threads_per_group) {
    float value = x[i];  // Coalesced access
}

// Bad: Random memory access pattern  
float value = x[random_index[local_id]];  // Non-coalesced
```

### Shared Memory Usage
```metal
threadgroup float shared_x[TILE_SIZE];  // Shared among threadgroup

// Load data collaboratively
shared_x[local_id] = x[base_offset + local_id];
threadgroup_barrier(mem_flags::mem_threadgroup);

// All threads can now access shared_x efficiently
```

### Loop Unrolling and Tiling
```metal
// Process multiple elements per thread to reduce overhead
for (uint i = 0; i < TILE_SIZE; i += 4) {
    // Unrolled loop processes 4 elements at once
    result += x[i+0] * w[i+0];
    result += x[i+1] * w[i+1]; 
    result += x[i+2] * w[i+2];
    result += x[i+3] * w[i+3];
}
```

## Task 3: 4-bit Weight Unpacking on GPU

The GPU kernel must efficiently unpack 4-bit weights:

```metal
// Pack/unpack utilities
uint8_t packed_byte = w[byte_index];
float w1 = float((packed_byte >> 4) & 0xF);  // Upper 4 bits
float w2 = float(packed_byte & 0xF);         // Lower 4 bits

// Dequantize  
float dequant_w1 = w1 * scale + bias;
float dequant_w2 = w2 * scale + bias;
```

**Optimization**: Process multiple packed bytes simultaneously using SIMD operations.

## Task 4: Performance Analysis

Compare CPU vs GPU performance characteristics:

**Memory Bandwidth**:
- M1 Max CPU: ~250 GB/s
- M1 Max GPU: ~400 GB/s  
- **GPU advantage**: ~1.6x

**Compute Throughput**:
- CPU: ~1 TFLOPS (high precision)
- GPU: ~10 TFLOPS (parallel operations)
- **GPU advantage**: ~10x

**Expected speedup**: 3-8x for quantized matrix multiplication, depending on matrix size.

## Task 5: Kernel Launch and Threading

Understanding how the GPU kernel is launched:

```cpp
// In C++ host code
auto kernel = device.get_kernel("quantized_matmul_kernel", library);

// Set up grid dimensions
uint32_t threads_per_group = 256;
uint32_t num_groups_x = (output_width + threads_per_group - 1) / threads_per_group;
uint32_t num_groups_y = output_height;

// Launch kernel
compute_encoder.dispatch_threadgroups(
    {num_groups_x, num_groups_y, 1},     // Number of threadgroups
    {threads_per_group, 1, 1});          // Threads per threadgroup
```

Each thread computes one or more elements of the output matrix.

## Task 6: Debugging GPU Kernels

Common GPU kernel issues:

1. **Out-of-bounds access**: Use proper bounds checking
```metal
if (global_id >= array_size) return;  // Prevent buffer overrun
```

2. **Memory alignment**: Ensure data is properly aligned
```metal
// Align to 16-byte boundaries for optimal performance
device float4* aligned_ptr = (device float4*)ptr;
```

3. **Threadgroup synchronization**: Use barriers correctly
```metal
threadgroup_barrier(mem_flags::mem_threadgroup);  // Sync before shared memory use
```

## Task 7: Benchmarking GPU vs CPU

Run performance comparisons:

```bash
# Benchmark different matrix sizes
pdm run python benches/test_quantized_matmul.py --device gpu
pdm run python benches/test_quantized_matmul.py --device cpu

# Profile memory usage
pdm run python -c "
import mlx.core as mx
# ... run quantized operations and check mx.metal.get_peak_memory()
"
```

Expected results:
- **Small matrices** (< 1024): CPU competitive due to launch overhead
- **Large matrices** (> 4096): GPU significantly faster  
- **Memory usage**: GPU uses device memory, reducing system RAM pressure

## Integration with MLX

The GPU kernel integrates with MLX's compilation and execution system:

```python
# MLX automatically chooses GPU when available
result = quantized_linear(x, quantized_weights)  # Runs on GPU if available

# Force specific device
with mx.stream(mx.gpu):
    result = quantized_linear(x, quantized_weights)
```

MLX handles:
- **Memory management**: Automatic transfers between CPU/GPU memory
- **Kernel compilation**: JIT compilation of Metal kernels
- **Stream scheduling**: Asynchronous execution and dependencies

At the end of this day, you should understand GPU architecture differences and how to optimize quantized operations for Metal/GPU execution.

```bash
pdm run test --week 2 --day 3
```

{{#include copyright.md}}
