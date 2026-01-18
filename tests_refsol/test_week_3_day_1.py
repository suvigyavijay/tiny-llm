import pytest
import mlx.core as mx
from tiny_llm_ref.paged_attention import BlockTable, BlockAllocator


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


def test_block_table_append_tokens():
    """Test block table correctly allocates blocks as tokens are added."""
    block_size = 4
    num_blocks = 10
    allocator = BlockAllocator(num_blocks)
    table = BlockTable(block_size, allocator)
    
    # Append 2 tokens -> 1 block needed
    table.append_tokens(2)
    assert len(table.physical_blocks) == 1
    assert table.num_tokens == 2
    
    # Append 3 more tokens -> total 5 -> 2 blocks needed
    table.append_tokens(3)
    assert len(table.physical_blocks) == 2
    assert table.num_tokens == 5
    
    # Verify allocator state
    assert len(allocator.free_blocks) == num_blocks - 2


def test_block_table_exact_fit():
    """Test when tokens exactly fill blocks."""
    block_size = 4
    allocator = BlockAllocator(10)
    table = BlockTable(block_size, allocator)
    
    # Exactly 4 tokens = 1 block
    table.append_tokens(4)
    assert len(table.physical_blocks) == 1
    
    # Add 1 more = 5 tokens = 2 blocks
    table.append_tokens(1)
    assert len(table.physical_blocks) == 2
