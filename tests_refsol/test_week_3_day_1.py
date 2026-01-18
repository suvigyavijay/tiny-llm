import pytest
import mlx.core as mx
from .tiny_llm_base import *
from .utils import *


def test_block_allocator_basic():
    """Test basic allocation and free operations."""
    allocator = BlockAllocator(num_blocks=10)
    
    block1 = allocator.allocate()
    block2 = allocator.allocate()
    
    assert block1 != block2
    assert len(allocator.free_blocks) == 8
    
    allocator.free(block1)
    assert len(allocator.free_blocks) == 9


def test_block_allocator_oom():
    """Test out-of-memory error when no blocks are available."""
    allocator = BlockAllocator(num_blocks=1)
    allocator.allocate()
    
    with pytest.raises(ValueError, match="Out of memory"):
        allocator.allocate()


@pytest.mark.parametrize("block_size", [4, 8, 16])
def test_block_table_append_tokens(block_size: int):
    """Test block table correctly allocates blocks as tokens are added."""
    num_blocks = 10
    allocator = BlockAllocator(num_blocks)
    table = BlockTable(block_size, allocator)
    
    # Append tokens less than block_size -> 1 block needed
    table.append_tokens(block_size // 2)
    assert len(table.physical_blocks) == 1
    assert table.num_tokens == block_size // 2
    
    # Append more to cross block boundary
    table.append_tokens(block_size)
    assert len(table.physical_blocks) == 2
    assert table.num_tokens == block_size // 2 + block_size


@pytest.mark.parametrize("block_size", [4, 8])
def test_block_table_fork_cow(block_size: int):
    """Test forking and Copy-on-Write logic."""
    allocator = BlockAllocator(20)
    table = BlockTable(block_size, allocator)
    
    # Fill 1.5 blocks worth of tokens
    tokens = block_size + block_size // 2
    table.append_tokens(tokens)
    parent_blocks = list(table.physical_blocks)
    
    # Fork
    child = table.fork()
    assert child.physical_blocks == parent_blocks
    assert allocator.ref_counts[parent_blocks[0]] == 2
    assert allocator.ref_counts[parent_blocks[1]] == 2
    
    # Append to child. Should trigger CoW for the partial block
    child.append_tokens(1) 
    
    # First block (full) should still be shared
    assert child.physical_blocks[0] == parent_blocks[0]
    # Second block (partial) should be copied
    assert child.physical_blocks[1] != parent_blocks[1]
    
    # Check ref counts
    assert allocator.ref_counts[parent_blocks[0]] == 2
    assert allocator.ref_counts[parent_blocks[1]] == 1
