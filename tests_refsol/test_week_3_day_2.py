import pytest
import mlx.core as mx
import math
from tiny_llm_ref.paged_attention import PagedAttention


@pytest.mark.skip(reason="Requires Metal extension to be built")
def test_paged_attention_kernel():
    """Test PagedAttention kernel correctness."""
    bs = 1
    num_heads = 4
    num_kv_heads = 4
    head_dim = 16
    block_size = 4
    num_blocks = 10
    scale = 1.0 / math.sqrt(head_dim)
    
    query = mx.random.uniform(shape=(bs, num_heads, head_dim))
    key_cache = mx.random.uniform(shape=(num_blocks, block_size, num_kv_heads, head_dim))
    value_cache = mx.random.uniform(shape=(num_blocks, block_size, num_kv_heads, head_dim))
    
    # Sequence length 6 (needs 2 blocks)
    block_tables = mx.array([[0, 1] + [0]*8], dtype=mx.int32)
    context_lens = mx.array([6], dtype=mx.int32)
    
    pa = PagedAttention(num_heads, head_dim, num_kv_heads, scale)
    output = pa(query, key_cache, value_cache, block_tables, context_lens, block_size)
    mx.eval(output)
    
    assert output.shape == (bs, num_heads, head_dim)
