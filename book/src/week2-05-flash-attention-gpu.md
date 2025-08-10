# Week 2 Day 5: Flash Attention - GPU Implementation

The GPU implementation of Flash Attention achieves significant speedups over CPU by leveraging massive parallelism and optimized memory hierarchies. This implementation uses Metal Shading Language to efficiently utilize Apple Silicon GPU cores.

## GPU Memory Hierarchy for Flash Attention

**Memory Types in Metal**:
1. **Device memory**: Main GPU memory (shared with CPU on Apple Silicon)
2. **Threadgroup memory**: Fast shared memory within a threadgroup (~32KB)
3. **Thread memory**: Private registers per thread

**Flash Attention GPU Strategy**:
- Store Q, K, V in device memory
- Load tiles into threadgroup memory for fast access
- Compute partial results in thread registers
- Minimize device memory roundtrips

## Task 1: Understanding GPU Threading Model

The GPU kernel maps computation to Metal's threading hierarchy:

```metal
kernel void flash_attention_f32_e128(
    device const float* q [[buffer(0)]],
    device const float* k [[buffer(1)]],  
    device const float* v [[buffer(2)]],
    device const float* mask [[buffer(3)]],
    device float* out [[buffer(4)]],
    // ... parameters ...
    uint2 group_id [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]])
```

**Thread mapping**:
- `group_id.x`: Batch/head index (which sequence)
- `group_id.y`: Query block index (which Q tile)  
- `simd_gid`: Row within Q block (which query vector)
- `simd_lid`: Column within K/V block (parallel processing)

**Readings**

- [Metal Best Practices Guide](https://developer.apple.com/documentation/metal/metal_best_practices_guide)
- [GPU Attention Optimization](https://arxiv.org/abs/2307.08691)
- [Metal Threading and Memory Model](https://developer.apple.com/documentation/metal/threads_and_threadgroups)

## Task 2: Threadgroup Memory Optimization

The kernel uses threadgroup memory to cache frequently accessed data:

```metal
threadgroup float q_shared[Br][E];   // Shared Q tile
threadgroup float k_shared[Bc][E];   // Shared K tile  
threadgroup float v_shared[Bc][E];   // Shared V tile

// Collaborative loading - each thread loads part of the tile
uint threads_per_group = get_simdgroups_per_threadgroup() * get_threads_per_simdgroup();
uint thread_id = simd_gid * get_threads_per_simdgroup() + simd_lid;

// Load Q tile collaboratively
for (uint idx = thread_id; idx < Br * E; idx += threads_per_group) {
    uint row = idx / E;
    uint col = idx % E;
    if (q_row_base + row < L) {
        q_shared[row][col] = q[(n * L + q_row_base + row) * E + col];
    }
}

threadgroup_barrier(mem_flags::mem_threadgroup);
```

## Task 3: SIMD-Level Parallelism

Within each SIMD group (32 threads), we parallelize the inner loops:

```metal
// Each thread in SIMD processes different K/V positions
for (uint kv_block = 0; kv_block < (S + Bc - 1) / Bc; kv_block++) {
    // Load K, V tile into threadgroup memory
    load_kv_tile(k, v, k_shared, v_shared, kv_block, ...);
    
    // Each thread computes scores for its assigned positions
    float scores[Br];  // Per-thread scores array
    
    for (uint q_row = 0; q_row < Br; q_row++) {
        if (simd_lid < Bc) {  // Guard against out-of-bounds
            scores[q_row] = 0.0f;
            
            // Dot product: Q[q_row] · K[simd_lid]
            for (uint e = 0; e < E; e++) {
                scores[q_row] += q_shared[q_row][e] * k_shared[simd_lid][e];
            }
            
            scores[q_row] *= scale;
            scores[q_row] += mask_value;  // Apply mask
        }
    }
    
    // Update running statistics using SIMD reductions
    update_running_stats(scores, v_shared, ...);
}
```

## Task 4: Online Softmax with SIMD Reductions

The GPU implementation uses SIMD reductions for efficient max and sum operations:

```metal
float compute_block_max(float local_scores[Br]) {
    float block_max = -INFINITY;
    
    for (uint q_row = 0; q_row < Br; q_row++) {
        if (simd_lid < Bc) {
            // SIMD reduction to find max across K positions
            float row_max = simd_max(local_scores[q_row]);
            if (simd_lid == 0) {  // One thread per SIMD updates shared memory
                block_max = max(block_max, row_max);
            }
        }
    }
    
    return block_max;
}

float compute_block_sum(float local_scores[Br], float block_max) {
    float block_sum = 0.0f;
    
    for (uint q_row = 0; q_row < Br; q_row++) {
        if (simd_lid < Bc) {
            float exp_score = exp(local_scores[q_row] - block_max);
            // SIMD reduction to sum across K positions  
            float row_sum = simd_sum(exp_score);
            if (simd_lid == 0) {
                block_sum += row_sum;
            }
        }
    }
    
    return block_sum;
}
```

## Task 5: Memory Coalescing and Access Patterns

Optimize memory access for maximum bandwidth:

```metal
// Good: Coalesced access - consecutive threads access consecutive memory
for (uint e = simd_lid; e < E; e += get_threads_per_simdgroup()) {
    float value = q_global[base_offset + e];  // Coalesced
}

// Bad: Strided access - threads access memory with large strides  
float value = q_global[base_offset + simd_lid * large_stride];  // Non-coalesced
```

**Bank conflict avoidance**:
```metal
// Avoid threadgroup memory bank conflicts by padding
threadgroup float shared_data[ROWS][COLS + 1];  // +1 padding prevents conflicts
```

## Task 6: Kernel Launch Configuration

Optimize the kernel launch parameters:

```cpp
// In C++ host code - optimal launch configuration
const uint32_t Br = 32;  // Query block size
const uint32_t Bc = 32;  // Key/Value block size  
const uint32_t threads_per_group = 256;  // Total threads per threadgroup
const uint32_t simdgroups_per_group = threads_per_group / 32;

// Grid dimensions
uint32_t num_groups_x = batch_size * num_heads;
uint32_t num_groups_y = (seq_len_q + Br - 1) / Br;

// Set kernel parameters
compute_encoder.set_bytes(Br, buffer_index++);
compute_encoder.set_bytes(Bc, buffer_index++);
compute_encoder.set_bytes(int(ceil(float(seq_len_k) / Bc)), buffer_index++);  // Tc

// Launch with optimal configuration
compute_encoder.dispatch_threadgroups(
    {num_groups_x, num_groups_y, 1},
    {threads_per_group, 1, 1});
```

## Task 7: Numerical Precision and Stability

GPU kernels require extra care for numerical stability:

```metal
// Use higher precision for accumulation
float high_precision_sum = 0.0f;  // vs half precision

// Kahan summation for better precision
float compensation = 0.0f;
for (...) {
    float y = value - compensation;
    float t = high_precision_sum + y;
    compensation = (t - high_precision_sum) - y;
    high_precision_sum = t;
}

// Safe exponential with range checking
float safe_exp(float x, float max_val) {
    float diff = x - max_val;
    return (diff < -20.0f) ? 0.0f : exp(diff);  // Avoid underflow
}
```

## Task 8: Performance Optimization Checklist

**Memory optimization**:
- [ ] Minimize device memory reads/writes
- [ ] Use threadgroup memory for frequently accessed data
- [ ] Ensure coalesced memory access patterns
- [ ] Avoid threadgroup memory bank conflicts

**Compute optimization**:
- [ ] Maximize SIMD utilization (32 threads working together)
- [ ] Use SIMD intrinsics (simd_max, simd_sum) for reductions
- [ ] Minimize thread divergence (avoid if/else that differ across SIMD)
- [ ] Balance work across threadgroups

**Kernel optimization**:
- [ ] Choose optimal block sizes (Br, Bc) for target hardware
- [ ] Minimize register usage to increase occupancy
- [ ] Use appropriate data types (float vs half precision)

## Task 9: Benchmarking GPU Performance

```bash
# Compare GPU vs CPU Flash Attention
pdm run python -c "
import mlx.core as mx
import time
from tiny_llm_ref import flash_attention

# Benchmark different sequence lengths
for seq_len in [512, 1024, 2048, 4096]:
    q = mx.random.uniform(shape=(1, 32, seq_len, 128), dtype=mx.float32)
    k = mx.random.uniform(shape=(1, 32, seq_len, 128), dtype=mx.float32)  
    v = mx.random.uniform(shape=(1, 32, seq_len, 128), dtype=mx.float32)
    
    # GPU timing
    with mx.stream(mx.gpu):
        start = time.time()
        result_gpu = flash_attention(q, k, v, scale=0.125)
        mx.eval(result_gpu)
        gpu_time = time.time() - start
    
    print(f'Seq {seq_len}: GPU {gpu_time:.4f}s')
"
```

Expected performance improvements:
- **Memory usage**: Constant regardless of sequence length (O(n) scaling)
- **Speed**: 3-10x faster than CPU for large sequences
- **Throughput**: Enables processing of much longer sequences

## Integration and Testing

```bash
# Test GPU Flash Attention
pdm run test --week 2 --day 5 -- -k flash_attention_gpu

# Verify correctness against reference implementation
pdm run test --week 2 --day 5 -- -k test_flash_attention_correctness
```

At the end of this day, you should understand how to implement and optimize Flash Attention for GPU execution, achieving significant speedups for attention computation.

```bash
pdm run test --week 2 --day 5
```

{{#include copyright.md}}
