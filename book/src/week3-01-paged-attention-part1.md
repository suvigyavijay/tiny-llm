# Week 3 Day 1: PagedAttention - The Memory Problem

In this chapter, we will tackle the memory fragmentation problem in LLM serving by implementing the **Block Table** for PagedAttention.

As we saw in Week 2, the KV cache grows linearly with sequence length. In standard attention, we store the KV cache as contiguous tensors: `[Batch, Num_Heads, Max_Length, Head_Dim]`.
This requires us to pre-allocate memory for the *maximum* possible sequence length, even if the actual generation is short. This leads to **memory fragmentation** and **waste**.

**📚 Readings**

- [vLLM: Easy, Fast, and Cheap LLM Serving with PagedAttention](https://blog.vllm.ai/2023/06/20/vllm.html)
- [PagedAttention Explained Visually](https://medium.com/@plg1017/pagedattention-explained-visually-7b7001431252)

## Motivation

As we saw in Week 2, the KV cache grows linearly with sequence length. In standard attention, we store the KV cache as contiguous tensors: `[Batch, Num_Heads, Max_Length, Head_Dim]`.
This requires us to pre-allocate memory for the *maximum* possible sequence length, even if the actual generation is short. This leads to **memory fragmentation** and **waste**.

Imagine a batch of 4 requests.
- Request A: 10 tokens generated so far.
- Request B: 100 tokens.
- Request C: 5 tokens.
- Request D: 50 tokens.

If we reserved `Max_Length=2048` for each, we have allocated `4 * 2048` slots. Most are empty. We can't put other data there because the tensor must be contiguous. This is the **"Swiss Cheese"** problem: memory is full of holes that can't be used.

## The Solution: Paging

PagedAttention takes inspiration from Operating Systems virtual memory.
- **Logical Memory**: The view from the model. It sees a continuous sequence of tokens `[0, 1, 2, ... L]`.
- **Physical Memory**: The actual storage in GPU RAM. It is divided into non-contiguous **Blocks**.
- **Block Table**: A mapping from Logical Blocks to Physical Blocks.

### Key Concepts

1.  **Block Size**: The number of tokens per block (e.g., 16 or 32).
2.  **Physical Block Number**: The index of a block in the global memory pool.
3.  **Block Table**: For each sequence, a list of physical block numbers.

Example (Block Size = 4):
Sequence: "The cat sat on the mat" (6 tokens).

Logical view:
- Block 0: ["The", "cat", "sat", "on"]
- Block 1: ["the", "mat", empty, empty]

Physical view (Global Memory):
- Block 0: [Data for some other req...]
- Block 7: ["The", "cat", "sat", "on"]  (Logical Block 0)
- Block 8: [Data for another req...]
- Block 42: ["the", "mat", empty, empty] (Logical Block 1)

Block Table for this sequence: `[7, 42]`.

## Task 1: The Block Table

```
src/tiny_llm/paged_attention.py
```

Implement a `BlockTable` class (or structure) that manages the mapping.
It needs to support:
- Adding a new block when the current one is full.
- Retrieving the physical block indices for a given sequence.

You should implement the `append_tokens` method in `BlockTable` and the `allocate` method in `BlockAllocator`.

### Code Walkthrough

```python
class BlockTable:
    def append_tokens(self, num_new_tokens: int):
        # 1. Calculate how much space is left in the last block
        # 2. If space > 0, fill it
        # 3. If tokens remain, allocate new blocks from allocator
        pass
```

## Testing

```bash
pdm run test-refsol tests_refsol/test_week_3_day_1.py
```

{{#include copyright.md}}
