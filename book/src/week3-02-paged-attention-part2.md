# Week 3 Day 2: PagedAttention - The Kernel

In this chapter, we will implement the **PagedAttention kernel** and integrate it with our model. This kernel allows the attention mechanism to read from non-contiguous memory blocks managed by the Block Table.

Now that we have the logical structure (Block Tables), we need to implement the Attention operation that uses it.

**📚 Readings**

- [Kernels for PagedAttention - vLLM Blog](https://blog.vllm.ai/2023/06/20/vllm.html)
- [FlashAttention-2: Faster Attention with Better Parallelism - Tri Dao](https://tridao.me/publications/flash2/)

## Motivation

Standard attention `Attention(Q, K, V)` assumes `K` and `V` are contiguous tensors of shape `[B, L, H, D]`.
With Paging, `K` and `V` are scattered in memory: `cache[block_idx]`.
We need a custom kernel that can "gather" these blocks on-the-fly and compute attention scores.

## The Paged Attention Operation

We want to compute Attention(Q, K, V).
- **Q**: `[Batch, 1, Head_Dim]` (Decoding phase - one token at a time).
- **K, V**: Stored in the Paged Cache.

The kernel needs to:
1.  Read the `block_table` for the current request.
2.  Fetch the relevant K/V blocks from non-contiguous memory.
3.  Compute attention scores and output.

### Memory Layout

The Paged KV Cache is usually a large pre-allocated tensor:
`[Num_Blocks, Block_Size, Head_Dim]` (per head/layer).

For a request `i`:
1.  Get `block_table[i]`: e.g., `[7, 42]`.
2.  The K/V data corresponds to `cache[7]` and `cache[42]`.

### Kernel Logic (Simplified)

```python
# For each sequence in batch
for i in range(batch_size):
    query = Q[i]
    blocks = block_table[i]
    
    # Iterate over blocks to compute scores
    for block_idx in blocks:
        k_block = cache_k[block_idx] # [Block_Size, Head_Dim]
        v_block = cache_v[block_idx]
        
        # Compute Q * K^T for this block
        scores += matmul(query, k_block.T)
        
    softmax(scores)
    output += scores * values
```

In the GPU kernel, this is parallelized. Each block (or group of blocks) is handled by threads.

## Task 1: PagedAttention Kernel

```
src/extensions/src/paged_attention.metal
```

Implement the `paged_attention` kernel.
Inputs:
- `query`: `[Batch, Num_Heads, Head_Dim]`
- `key_cache`, `value_cache`: `[Num_Blocks, Block_Size, Num_Heads, Head_Dim]`
- `block_tables`: `[Batch, Max_Num_Blocks_Per_Seq]`
- `context_lens`: `[Batch]` (Actual length of each sequence)

**Note**: For simplicity in `tiny-llm`, we stick to a simpler layout `[Num_Blocks, Block_Size, Num_Heads, Head_Dim]`.

## Task 2: Python Integration

```
src/tiny_llm/paged_attention.py
```

Implement the `PagedAttention` module that wraps the kernel.

## Testing

```bash
pdm run test-refsol tests_refsol/test_week_3_day_2.py
```

{{#include copyright.md}}
