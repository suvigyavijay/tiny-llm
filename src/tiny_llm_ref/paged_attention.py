import mlx.core as mx
from typing import List

try:
    import tiny_llm_ext_ref._ext as _ext
except ImportError:
    _ext = None


class BlockAllocator:
    """
    Manages the pool of physical blocks.
    """
    def __init__(self, num_blocks: int):
        self.num_blocks = num_blocks
        self.free_blocks = list(range(num_blocks))
        self.ref_counts = [0] * num_blocks

    def allocate(self) -> int:
        if not self.free_blocks:
            raise ValueError("Out of memory: No free blocks available")
        block_idx = self.free_blocks.pop()
        self.ref_counts[block_idx] = 1
        return block_idx

    def free(self, block_idx: int):
        if block_idx < 0 or block_idx >= self.num_blocks:
            raise ValueError(f"Invalid block index {block_idx}")
        
        self.ref_counts[block_idx] -= 1
        if self.ref_counts[block_idx] == 0:
            self.free_blocks.append(block_idx)

    def add_ref(self, block_idx: int):
        if self.ref_counts[block_idx] == 0:
             raise ValueError(f"Block {block_idx} is not allocated")
        self.ref_counts[block_idx] += 1


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
        total_tokens_after = self.num_tokens + num_new_tokens
        blocks_needed = (total_tokens_after + self.block_size - 1) // self.block_size
        current_blocks = len(self.physical_blocks)
        
        # Copy-on-Write check for partial last block
        if current_blocks > 0 and self.num_tokens % self.block_size != 0:
            last_block = self.physical_blocks[-1]
            if self.allocator.ref_counts[last_block] > 1:
                # Need to copy the partial block
                new_block = self.allocator.allocate()
                self.allocator.free(last_block)
                self.physical_blocks[-1] = new_block
        
        for _ in range(blocks_needed - current_blocks):
            new_block = self.allocator.allocate()
            self.physical_blocks.append(new_block)
            
        self.num_tokens = total_tokens_after

    def fork(self) -> "BlockTable":
        """Create a new BlockTable sharing physical blocks (for beam search)."""
        new_table = BlockTable(self.block_size, self.allocator)
        new_table.physical_blocks = self.physical_blocks.copy()
        new_table.num_tokens = self.num_tokens
        
        # Increment ref counts for all shared blocks
        for block_idx in self.physical_blocks:
            self.allocator.add_ref(block_idx)
            
        return new_table

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
        if _ext is not None:
            return _ext.paged_attention(
                query,
                key_cache,
                value_cache,
                block_tables,
                context_lens,
                block_size,
                self.scale,
                self.num_kv_heads,
                self.num_heads
            )
        else:
            raise RuntimeError("PagedAttention extension not available")
