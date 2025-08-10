# Week 2 Day 1: Key-Value Cache

The Key-Value (KV) cache is one of the most important optimizations for autoregressive text generation. Without it, generating a sequence of length `n` requires `O(n²)` computation because each new token requires recomputing attention over all previous tokens. With KV caching, this becomes `O(n)`.

## The Problem: Redundant Computation

In autoregressive generation, each step generates one new token based on all previous tokens. The attention mechanism needs to compute:

```
Attention(Q, K, V) = softmax(QK^T / √d) V
```

For a sequence of length `n`, step `i` computes attention over positions `[0, 1, ..., i]`. This means:
- Step 1: Compute attention for 1 token
- Step 2: Compute attention for 2 tokens (including redundant computation from step 1)
- Step 3: Compute attention for 3 tokens (including redundant computation from steps 1-2)
- ...
- Step n: Compute attention for n tokens

Total computation: `1 + 2 + 3 + ... + n = O(n²)`

## The Solution: Cache Key-Value Pairs

The key insight is that the Key (K) and Value (V) matrices depend only on the input tokens, not the current query position. Once computed, they can be reused:

```
# Step 1: Generate token 1
K₁, V₁ = compute_kv(token₀)  # Compute for prompt
Q₁ = compute_q(token₀)       # Query for next token
out₁ = attention(Q₁, K₁, V₁)

# Step 2: Generate token 2 
K₂ = concat(K₁, compute_kv(token₁))  # Reuse K₁, only compute new K
V₂ = concat(V₁, compute_kv(token₁))  # Reuse V₁, only compute new V
Q₂ = compute_q(token₁)               # New query
out₂ = attention(Q₂, K₂, V₂)
```

With caching, each step only computes K and V for the single new token, reducing total computation to `O(n)`.

**Readings**

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Original transformer paper
- [KV Cache in Transformers](https://kipp.ly/transformer-inference-arithmetic/) - Detailed analysis
- [vLLM KV Cache Management](https://github.com/vllm-project/vllm/blob/main/vllm/attention/backends/abstract.py)

## Task 1: Understand KV Cache Interface

You will work with the following file:
```
src/tiny_llm/kv_cache.py
```

The `TinyKvCache` abstract class defines the interface:

```python
class TinyKvCache:
    def update_and_fetch(
        self,
        key: mx.array,      # New key tensor [B, H, L, D]
        value: mx.array,    # New value tensor [B, H, L, D]
        mask_length: int | None = None,
        mask: mx.array | str | None = None,
    ) -> tuple[mx.array, mx.array, int, mx.array]:
        # Returns: (all_keys, all_values, seq_len, updated_mask)
```

The method:
1. **Updates** the cache with new key/value pairs
2. **Returns** all cached keys/values (including new ones)
3. **Manages** attention masks for proper causal attention

## Task 2: Implement `TinyKvFullCache`

Implement the `TinyKvFullCache` class in `src/tiny_llm/kv_cache.py`. This stores the complete key-value history:

```python
class TinyKvFullCache(TinyKvCache):
    def __init__(self):
        self.key_values = None  # (keys, values) tuple
        self.offset = 0         # Current sequence length
    
    def update_and_fetch(self, key, value, mask_length=None, mask=None):
        # Your implementation here:
        # 1. If first call, store key/value directly
        # 2. Otherwise, concatenate with existing keys/values
        # 3. Update offset
        # 4. Return combined keys/values with updated mask
```

**Key points:**
- Handle the first call (when `self.key_values` is None)
- Concatenate along the sequence dimension (axis=2)
- Maintain the offset for sequence length tracking
- Return the mask unchanged (it will be processed elsewhere)

## Task 3: Test Your Implementation

Test your KV cache implementation:

```bash
pdm run test --week 2 --day 1 -- -k task_2
```

## Task 4: Understanding Batched KV Cache

Examine the `BatchingKvCache` implementation in the reference solution. This more complex cache:

1. **Manages multiple requests** simultaneously
2. **Handles variable sequence lengths** across requests
3. **Provides unified batched output** for efficient computation

Key challenges in batched caching:
- **Memory layout**: How to store different sequence lengths
- **Masking**: Proper attention masks for variable-length sequences  
- **Request lifecycle**: Adding/removing requests dynamically

## Memory Analysis

Consider memory usage for different caching strategies:

**No Cache (Recompute):**
- Memory: O(1) per step
- Computation: O(n²) total

**Full Cache:**
- Memory: O(n) total  
- Computation: O(n) total

**Trade-offs:**
- Full cache uses more memory but dramatically reduces computation
- For typical inference (generating 100s of tokens), memory cost is justified
- Production systems may use more sophisticated caching (e.g., paged attention)

## Integration with Attention

The KV cache integrates with the attention mechanism in `Qwen2MultiHeadAttention`:

```python
# In attention forward pass:
def __call__(self, x, offset, cache, mask=None):
    # Compute Q, K, V for new tokens
    q = self.wq(x)  # Query for new tokens
    k = self.wk(x)  # Key for new tokens  
    v = self.wv(x)  # Value for new tokens
    
    # Update cache and get all K, V
    all_k, all_v, seq_len, mask = cache.update_and_fetch(k, v, mask=mask)
    
    # Compute attention with full context
    output = attention(q, all_k, all_v, mask=mask)
    return output
```

This pattern allows each attention layer to maintain its own cache while providing a simple, consistent interface.

At the end of this day, you should understand how KV caching transforms autoregressive generation from `O(n²)` to `O(n)` complexity and have implemented a working cache system.

```bash
pdm run test --week 2 --day 1
```

{{#include copyright.md}}
