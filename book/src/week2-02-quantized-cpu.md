# Week 2 Day 2: Quantized Matrix Multiplication - CPU

Quantization is a crucial optimization that reduces model memory usage by representing weights with lower precision. Instead of 16-bit floating point (bfloat16), we use 4-bit integers (INT4), achieving a ~75% reduction in memory usage with minimal quality loss.

## Understanding Quantization

**Standard Weights**: Each weight is a 16-bit float (~2 bytes)
```
W = [1.23, -0.87, 2.14, -1.56, ...]  # bfloat16
```

**Quantized Weights**: Weights are grouped and represented as:
- **4-bit integers** (0-15): The quantized values
- **Scales**: Floating-point scale factors per group  
- **Biases**: Floating-point bias terms per group

```
# Group of 32 weights -> 16 bytes (32 * 4 bits) + scale + bias
quantized = [12, 3, 15, 1, ...]     # 4-bit values
scale = 0.1847                       # Float scale for this group
bias = -0.0234                       # Float bias for this group

# Dequantization: W[i] = quantized[i] * scale + bias
```

## Why INT4 Quantization Works

1. **Weight Distribution**: Neural network weights typically follow normal distributions
2. **Group-wise Quantization**: Small groups (32-128 weights) have similar ranges
3. **Affine Quantization**: Scale + bias captures the weight range accurately

**Memory Savings**:
- Original: 16 bits/weight = 16 bits
- Quantized: 4 bits/weight + scales/biases ≈ 4.5 bits  
- **Compression Ratio**: ~3.5x smaller

**Readings**

- [QLoRA: Efficient Fine-tuning with 4-bit Quantization](https://arxiv.org/abs/2305.14314)
- [GPTQ: Accurate Post-Training Quantization](https://arxiv.org/abs/2210.17323)
- [MLX Quantization Documentation](https://ml-explore.github.io/mlx/build/html/python/nn.html#quantization)

## Task 1: Understand Quantized Matrix Multiplication

You will work with these files:
```
src/tiny_llm/quantize.py
src/extensions/src/quantized_matmul.cpp
```

The quantized matrix multiplication performs:
```
Y = X @ dequantize(W_quantized)
```

Where dequantization happens on-the-fly during computation:
```python
# Conceptual implementation
def quantized_matmul(scales, biases, group_size, bits, x, w_quantized):
    # Dequantize weights group by group
    w_full = dequantize(w_quantized, scales, biases, group_size, bits)
    # Standard matrix multiplication  
    return x @ w_full
```

## Task 2: Implement Quantized Linear Layer

Implement the `quantized_linear` function in `src/tiny_llm/quantize.py`:

```python
def quantized_linear(
    x: mx.array,
    w: QuantizedWeights,
    bias: mx.array | None = None,
) -> mx.array:
    # Your implementation:
    # 1. Call quantized_matmul with weight parameters
    # 2. Add bias if provided
    # 3. Handle reshaping for batched inputs
```

The `QuantizedWeights` class contains:
- `scales`: Per-group scale factors
- `biases`: Per-group bias terms  
- `group_size`: Number of weights per group (usually 32 or 64)
- `bits`: Quantization bits (4 for INT4)
- `weight`: The quantized weight tensor

## Task 3: CPU Kernel Implementation

Examine the CPU kernel in `src/extensions/src/quantized_matmul.cpp`. The kernel:

1. **Processes groups sequentially**: Dequantize one group at a time
2. **Minimizes memory allocation**: Reuses temporary buffers
3. **Optimizes cache usage**: Tiles the computation for better cache locality

**Key optimizations**:
```cpp
// Process in tiles to improve cache locality
for (int row_tile = 0; row_tile < M; row_tile += TILE_SIZE) {
    for (int col_tile = 0; col_tile < N; col_tile += TILE_SIZE) {
        // Dequantize and multiply tile by tile
        quantized_matmul_tile(x, w, scales, biases, ...);
    }
}
```

## Task 4: Understanding the Quantization Format

MLX uses a specific quantization format:

**Weight Packing**: 4-bit values are packed into bytes
```
# Two 4-bit values per byte
byte = (w1 << 4) | w2  # Pack two 4-bit weights
w1 = (byte >> 4) & 0xF  # Extract first weight
w2 = byte & 0xF         # Extract second weight
```

**Group Layout**: Weights are organized in groups
```
# For group_size=32, each group has:
# - 16 bytes of quantized weights (32 * 4 bits = 128 bits = 16 bytes)
# - 1 float32 scale (4 bytes)  
# - 1 float32 bias (4 bytes)
# Total: 24 bytes per group (vs 64 bytes unquantized)
```

## Task 5: Performance Analysis

Understand the performance trade-offs:

**Memory Bandwidth**: 
- Quantized: Load 4 bits + scale/bias per weight
- Full precision: Load 16 bits per weight
- **Bandwidth reduction**: ~3.5x

**Computation**: 
- Quantized: Extra dequantization overhead
- Full precision: Direct multiplication
- **Overhead**: ~10-20% slower per operation

**Net effect**: Memory bandwidth is usually the bottleneck, so quantization provides overall speedup despite computation overhead.

## Task 6: Test Your Implementation

```bash
# Test quantized operations
pdm run test --week 2 --day 2 -- -k quantized

# Benchmark performance difference
pdm run python benches/test_quantized_matmul.py
```

## Integration with Qwen2

The quantized operations integrate seamlessly with the model:

```python
class Qwen2MultiHeadAttention:
    def __init__(self, ..., wq: QuantizedWeights, ...):
        self.wq = wq  # Quantized query projection
        
    def __call__(self, x, ...):
        # Use quantized linear instead of regular linear
        q = quantized_linear(x, self.wq)  # 4-bit computation
        # ... rest of attention computation
```

This allows the model to maintain the same API while using dramatically less memory.

## Common Issues and Debugging

1. **Shape mismatches**: Ensure quantized weights have correct dimensions
2. **Precision errors**: Quantization introduces small numerical differences
3. **Group alignment**: Weight tensors must be divisible by group_size

At the end of this day, you should understand INT4 quantization and have implemented efficient CPU kernels for quantized matrix multiplication.

```bash
pdm run test --week 2 --day 2
```

{{#include copyright.md}}
