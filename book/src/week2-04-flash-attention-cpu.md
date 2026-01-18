# Week 2 Day 4: Flash Attention (CPU)

Standard attention has $O(N^2)$ memory complexity because it materializes the $S = Q K^T$ matrix (size $L \times L$). For long sequences, this is prohibitively expensive.
**Flash Attention** computes the attention output without ever materializing the full $S$ matrix in slow memory (HBM/DRAM), using tiling and online softmax.

**📚 Readings**

- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness - Tri Dao](https://tridao.me/publications/flash2/)
- [ELI5: Flash Attention - Aleksa Gordić](https://gordicaleksa.medium.com/eli5-flash-attention-5c479723dcab)
- [From Online Softmax to FlashAttention](https://courses.cs.washington.edu/courses/cse599m/23sp/notes/flash_attention.pdf)

## The Algorithm

The key idea is to compute the softmax normalization factor incrementally.
Recall:
$$ \text{Softmax}(x)_i = \frac{e^{x_i}}{\sum e^{x_j}} $$

Standard softmax requires seeing *all* $x_j$ to compute the denominator.
**Online Softmax** allows us to update the max $m$ and sum-exp $l$ as we see new blocks of $K, V$.

### Tiling

We divide $Q, K, V$ into blocks of size $B_r$ (block row) and $B_c$ (block col).
Instead of computing the full $N \times N$ attention matrix, we iterate through these blocks. We load them into fast memory (SRAM/L1 cache), compute the partial attention for that small block, update our running statistics ($m$ and $l$), and write back to output.

On CPU, we simulate this "SRAM" behavior by explicitly looping over blocks and computing locally.

## Task 1: CPU Implementation

```
src/extensions/src/flash_attention.cpp
```

Implement `flash_attention` using the tiled approach.

### Pseudocode

The following pseudocode outlines the tiled attention algorithm with online softmax updates.

```python
for n in range(Batch):
  for i in range(Tr):  # Loop over Q blocks (rows of attention matrix)
    Load Q_i
    Initialize O_i = 0, l_i = 0, m_i = -inf
    
    for j in range(Tc):  # Loop over K, V blocks (cols of attention matrix)
      Load K_j, V_j
      
      # Compute S_ij = Q_i * K_j^T
      S_ij = matmul(Q_i, K_j.T)
      
      # Masking
      Apply causal mask to S_ij
      
      # Online Softmax updates
      m_ij = rowmax(S_ij)
      P_ij = exp(S_ij - m_ij)
      l_ij = rowsum(P_ij)
      
      # Update global stats
      # We need to rescale the old running sum and output based on the new max
      m_new = max(m_i, m_ij)
      l_new = exp(m_i - m_new) * l_i + exp(m_ij - m_new) * l_ij
      
      # Update Output
      # O_i = diag(l_new)^-1 * (diag(l_i)*exp(m_i - m_new)*O_i + exp(m_ij - m_new)*P_ij*V_j)
      # Simpler form: keep O_i un-normalized until the end
      
      # Update accumulators
      m_i = m_new
      l_i = l_new
      
    # Final normalization
    O_i = O_i / l_i
    Store O_i
```

### Implementation Details

- **Block Sizes**: Use $B_r = 32, B_c = 32$.
- **Dimensions**:
    - $Q: [N, L, E]$
    - $K, V: [N, S, E]$
- **Masking**: You need to handle the causal mask correctly within the tiled loop.

## Testing

```bash
pdm run test-refsol tests_refsol/test_week_2_day_4.py
```

Run `test_flash_attention_cpu`.

{{#include copyright.md}}
