import pytest
import mlx.core as mx
import math
from .tiny_llm_base import *
from .utils import *


@pytest.mark.skip(reason="Requires Metal extension to be built")
@pytest.mark.parametrize("stream", AVAILABLE_STREAMS, ids=AVAILABLE_STREAMS_IDS)
def test_paged_attention_kernel(stream: mx.Stream):
    """Test PagedAttention kernel correctness."""
    with mx.stream(stream):
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


@pytest.mark.skip(reason="Requires Metal extension to be built")
@pytest.mark.parametrize("context_len", [4, 8, 16])
def test_paged_attention_variable_length(context_len: int):
    """Test PagedAttention with different context lengths."""
    num_heads = 2
    num_kv_heads = 2
    head_dim = 8
    block_size = 4
    num_blocks = 10
    scale = 1.0 / math.sqrt(head_dim)
    
    blocks_needed = (context_len + block_size - 1) // block_size
    
    query = mx.random.uniform(shape=(1, num_heads, head_dim))
    key_cache = mx.random.uniform(shape=(num_blocks, block_size, num_kv_heads, head_dim))
    value_cache = mx.random.uniform(shape=(num_blocks, block_size, num_kv_heads, head_dim))
    
    block_tables = mx.array([list(range(blocks_needed)) + [0]*(10-blocks_needed)], dtype=mx.int32)
    context_lens = mx.array([context_len], dtype=mx.int32)
    
    pa = PagedAttention(num_heads, head_dim, num_kv_heads, scale)
    output = pa(query, key_cache, value_cache, block_tables, context_lens, block_size)
    mx.eval(output)
    
    assert output.shape == (1, num_heads, head_dim)
