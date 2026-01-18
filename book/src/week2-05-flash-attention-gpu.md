# Week 2 Day 5: Flash Attention (GPU)

Today we implement Flash Attention on the GPU using Metal. This is significantly more complex than the CPU version because we need to manage threadgroups and SIMD groups (warps) explicitly to achieve high performance.

**📚 Readings**

- [GPU Kernels for Fast Attention - Huyen Chip](https://huyenchip.com/2023/01/24/gpu-kernels.html)
- [Using Metal for Machine Learning - Apple Developer](https://developer.apple.com/documentation/metal/metal_sample_code_library/compute_and_machine_learning/using_metal_for_machine_learning)
- [Metal Shading Language: SIMD Functions](https://developer.apple.com/documentation/metal/metal_shading_language_specification)

## The Kernel

```
src/extensions/src/flash_attention.metal
```

We will implement `flash_attention_f32_e128`.
- **Threadgroup**: Each threadgroup handles one block of $Q$ (size $B_r \times E$).
- **SIMD Group**: We use SIMD primitives (`simd_max`, `simd_sum`) to perform reductions across the threadgroup efficiently without explicit shared memory barriers for every operation.

### Thread Layout

- `simd_gid` (lane ID / 32): Maps to the row index $a$ in the $Q$ block ($0 \dots B_r-1$).
- `simd_lid` (lane ID % 32): Maps to the dimension $E$ or column index $b$ in $K$ block.
- $B_r = 32$, $B_c = 32$.
- $E$ is up to 128.

### SIMD Reduction Explained

In CPU code, we might loop:
```cpp
float max_val = -inf;
for (int i = 0; i < N; ++i) max_val = max(max_val, data[i]);
```
On GPU, within a SIMD group (warp), all 32 threads run in lockstep.
`simd_max(val)` computes the maximum of `val` across all threads in the SIMD group and broadcasts the result to all of them.
This allows us to compute row-max or row-sum instantly across the warp.

### Algorithm

1.  **Load Q**: Each thread loads its part of $Q_i$ into registers (or threadgroup memory).
2.  **Loop over j** (blocks of K, V):
    - Load $K_j$.
    - Compute $S_{ij} = Q_i \times K_j^T$.
    - Note: This is a matrix multiplication of shape $[1, E] \times [E, B_c]$.
    - Result is a row vector of length $B_c$.
    - Since `simd_lid` maps to columns of $K_j$ ($0..31$), each thread computes **one element** of the score row.
    - `s_a_b` is the score for $Q[a] \cdot K[b]$.
3.  **Softmax**:
    - Compute `rowmax` using `simd_max(s_a_b)`.
    - Update `m_i`, `l_i`.
4.  **Update Output**:
    - Compute `P * V`.
    - $P$ is a scalar per thread (since each thread handles one $b$).
    - $V$ is $[B_c, E]$.
    - We need to sum over $B_c$.
    - For a fixed $c$, each thread $b$ computes $P[a,b] \times V[b,c]$.
    - Then `simd_sum` reduces across $b$ (0..31).
    - The result is $(P \times V)[a, c]$.
    - Only lane 0 writes it.

## Task 1: Implement `flash_attention.metal`

Follow the logic above.
- Use `threadgroup` memory for `o_i`.
- Use `simd` primitives for reductions.
- Be careful with synchronizations (`threadgroup_barrier`).

## Testing

```bash
pdm run test-refsol tests_refsol/test_week_2_day_4.py
```

Run `test_flash_attention_gpu`.

{{#include copyright.md}}
