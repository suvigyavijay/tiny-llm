import mlx.core as mx
from typing import List, Optional


class BlockAllocator:
    """
    Manages the pool of physical blocks.
    Think of this as `malloc` for KV cache blocks.
    """
    def __init__(self, num_blocks: int):
        self.num_blocks = num_blocks
        self.free_blocks = list(range(num_blocks))

    def allocate(self) -> int:
        """
        Pop a block index from the free list. Raise Error if full.
        """
        pass

    def free(self, block_idx: int):
        """
        Push a block index back to the free list.
        """
        pass


class BlockTable:
    """
    Manages the mapping between logical blocks and physical blocks for a single sequence.
    """
    def __init__(self, block_size: int, allocator: BlockAllocator):
        self.block_size = block_size
        self.allocator = allocator
        self.physical_blocks: List[int] = []
        self.num_tokens = 0

    def append_tokens(self, num_new_tokens: int):
        """
        Allocate new blocks if necessary to accommodate new tokens.
        """
        pass

    def get_physical_blocks(self) -> List[int]:
        return self.physical_blocks


class PagedAttention:
    """
    The PagedAttention module.
    """
    def __init__(self, num_heads: int, head_dim: int, num_kv_heads: int, scale: float):
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_kv_heads = num_kv_heads
        self.scale = scale

    def __call__(
        self,
        query: mx.array,
        key_cache: mx.array,
        value_cache: mx.array,
        block_tables: mx.array,
        context_lens: mx.array,
        block_size: int
    ) -> mx.array:
        """
        Args:
            query: [batch, num_heads, head_dim]
            key_cache: [num_blocks, block_size, num_kv_heads, head_dim]
            value_cache: [num_blocks, block_size, num_kv_heads, head_dim]
            block_tables: [batch, max_blocks_per_seq]
            context_lens: [batch]
        """
        pass
