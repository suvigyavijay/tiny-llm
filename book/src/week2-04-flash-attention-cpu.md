# Week 2 Day 4: Flash Attention - CPU Implementation

Flash Attention is a memory-efficient algorithm that reduces the memory complexity of attention from O(n²) to O(n) while maintaining mathematical equivalence to standard attention. This is crucial for handling long sequences that would otherwise exceed GPU memory.

## The Memory Problem with Standard Attention

Standard attention computation:
```python
# Standard attention - high memory usage
scores = Q @ K.T                    # Shape: [batch, heads, seq_len, seq_len]
scores = scores / sqrt(head_dim)    # Apply scaling
scores = apply_mask(scores, mask)   # Apply causal mask
probs = softmax(scores)             # Softmax across last dimension
output = probs @ V                  # Final output
```

**Memory complexity**: O(n²) for the `scores` and `probs` matrices
- For sequence length 4096: 4096² = 16M elements per head
- With 32 heads, bfloat16: 16M × 32 × 2 bytes = 1GB just for attention matrices!

## Flash Attention Algorithm

Flash Attention computes the same result but never materializes the full attention matrix. Instead, it:

1. **Tiles the computation**: Process small blocks at a time
2. **Fuses operations**: Combine scaling, masking, and softmax 
3. **Recomputes when needed**: Trade computation for memory

**Key insight**: Softmax can be computed incrementally using the mathematical identity:
```
softmax([x1, x2, ..., xn]) = exp(xi - max(x)) / Σ exp(xj - max(x))
```

**Readings**

- [Flash Attention Paper](https://arxiv.org/abs/2205.14135) - Original algorithm
- [Flash Attention 2](https://arxiv.org/abs/2307.08691) - Optimized version
- [Stanford CS 329S Lecture on Flash Attention](https://stanford-cs329s.github.io/syllabus.html)

## Task 1: Understanding the Tiling Strategy

Flash Attention divides the attention computation into tiles:

```
Q: [B, H, L, E] -> Process in blocks of size [Br, E]
K: [B, H, S, E] -> Process in blocks of size [Bc, E]  
V: [B, H, S, E] -> Process in blocks of size [Bc, E]

For each Q block (size Br):
    For each K,V block (size Bc):
        Compute partial attention for this Q,K,V tile
        Update running statistics (max, sum, output)
```

The algorithm maintains:
- **Running maximum**: `m_i = max(m_{i-1}, max(scores_tile))`
- **Running sum**: `l_i = exp(m_{i-1} - m_i) * l_{i-1} + sum(exp(scores_tile - m_i))`
- **Running output**: `O_i = updated based on new partial results`

## Task 2: Implement Flash Attention Tiling

Examine the CPU implementation in `src/extensions_ref/src/flash_attention.cpp`:

```cpp
void flash_attention_cpu_kernel(
    const float* q, const float* k, const float* v, const float* mask,
    float* output, int B, int H, int L, int S, int E, float scale) {
    
    const int Br = 32;  // Query block size
    const int Bc = 32;  // Key/Value block size
    
    // For each batch and head
    for (int b = 0; b < B; b++) {
        for (int h = 0; h < H; h++) {
            // Process Q in blocks of size Br
            for (int i = 0; i < L; i += Br) {
                float m[Br];      // Running max for this Q block
                float l[Br];      // Running sum for this Q block  
                float o[Br][E];   // Running output for this Q block
                
                // Initialize running statistics
                for (int r = 0; r < Br; r++) {
                    m[r] = -INFINITY;
                    l[r] = 0.0f;
                    memset(o[r], 0, E * sizeof(float));
                }
                
                // Process K,V in blocks of size Bc
                for (int j = 0; j < S; j += Bc) {
                    // Compute Q @ K^T for this tile
                    float scores[Br][Bc];
                    compute_scores_tile(q, k, scores, ...);
                    
                    // Apply scaling and masking
                    apply_scale_and_mask(scores, mask, scale, ...);
                    
                    // Update running statistics with online softmax
                    update_statistics(scores, v, m, l, o, ...);
                }
                
                // Finalize output for this Q block
                finalize_output(o, l, output, ...);
            }
        }
    }
}
```

## Task 3: Online Softmax Algorithm

The core of Flash Attention is the online softmax update:

```cpp
void update_statistics(float scores[Br][Bc], const float* v_tile,
                      float* m, float* l, float o[Br][E], ...) {
    for (int r = 0; r < Br; r++) {
        // Find max of current tile
        float m_new = m[r];
        for (int c = 0; c < Bc; c++) {
            m_new = max(m_new, scores[r][c]);
        }
        
        // Compute sum of exponentials
        float l_new = l[r] * exp(m[r] - m_new);
        for (int c = 0; c < Bc; c++) {
            l_new += exp(scores[r][c] - m_new);
        }
        
        // Update output using corrected weights  
        float correction = l[r] * exp(m[r] - m_new) / l_new;
        for (int e = 0; e < E; e++) {
            o[r][e] = o[r][e] * correction;  // Correct previous contributions
            
            // Add new contributions
            for (int c = 0; c < Bc; c++) {
                float attn_weight = exp(scores[r][c] - m_new) / l_new;
                o[r][e] += attn_weight * v_tile[c * E + e];
            }
        }
        
        // Update running statistics
        m[r] = m_new;
        l[r] = l_new;
    }
}
```

## Task 4: Handling Masks and Causal Attention

Flash Attention must handle attention masks efficiently:

```cpp
void apply_scale_and_mask(float scores[Br][Bc], const float* mask,
                         float scale, int i, int j, ...) {
    for (int r = 0; r < Br; r++) {
        for (int c = 0; c < Bc; c++) {
            // Apply scaling
            scores[r][c] *= scale;
            
            // Apply mask (causal or custom)
            int query_pos = i + r;
            int key_pos = j + c;
            
            if (mask != nullptr) {
                float mask_val = mask[query_pos * S + key_pos];
                scores[r][c] += mask_val;  // Add mask (usually -inf for masked)
            } else if (is_causal && key_pos > query_pos) {
                scores[r][c] = -INFINITY;  // Causal mask
            }
        }
    }
}
```

## Task 5: CPU Optimization Techniques

**Cache optimization**:
```cpp
// Transpose V for better cache access patterns
float v_transposed[E][Bc];
for (int c = 0; c < Bc; c++) {
    for (int e = 0; e < E; e++) {
        v_transposed[e][c] = v_tile[c * E + e];
    }
}
```

**SIMD vectorization**:
```cpp
// Use vector instructions for dot products
#pragma omp simd
for (int e = 0; e < E; e++) {
    o[r][e] += attn_weight * v_tile[c * E + e];
}
```

**Blocking for cache locality**:
- Choose `Br` and `Bc` to fit in L1/L2 cache
- Typical values: `Br = 32`, `Bc = 32` for good balance

## Task 6: Numerical Stability

Flash Attention maintains numerical stability through:

1. **Safe softmax**: Always subtract the maximum before exponential
2. **Incremental updates**: Avoid catastrophic cancellation in floating point
3. **Careful order of operations**: Process tiles in a stable order

```cpp
// Numerically stable exponential
float safe_exp(float x, float max_val) {
    return exp(x - max_val);  // Prevents overflow
}
```

## Task 7: Test and Benchmark

```bash
# Test Flash Attention correctness
pdm run test --week 2 --day 4 -- -k flash_attention_cpu

# Benchmark memory usage and speed
pdm run python -c "
import mlx.core as mx
from tiny_llm_ref import flash_attention, scaled_dot_product_attention_grouped

# Compare memory usage for large sequences
seq_len = 4096
# ... benchmark both implementations
"
```

Expected results:
- **Memory**: O(n) vs O(n²) - dramatic savings for long sequences
- **Speed**: Competitive or faster than standard attention due to better cache usage
- **Accuracy**: Numerically equivalent (within floating point precision)

## Integration with Attention Layer

Flash Attention integrates seamlessly:

```python
def __call__(self, x, offset, cache, mask=None):
    # ... compute Q, K, V ...
    
    if self.use_flash_attention:
        output = flash_attention(q, k, v, scale=self.scale, mask=mask)
    else:
        output = scaled_dot_product_attention_grouped(q, k, v, scale=self.scale, mask=mask)
    
    # ... rest of attention computation ...
```

At the end of this day, you should understand how Flash Attention achieves O(n) memory complexity while maintaining mathematical equivalence to standard attention.

```bash
pdm run test --week 2 --day 4
```

{{#include copyright.md}}
