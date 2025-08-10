"""
PagedAttention implementation for efficient memory management in LLM serving.

Based on the vLLM PagedAttention algorithm that enables dynamic memory allocation
and sharing for KV caches, achieving significant memory efficiency improvements.
"""

import mlx.core as mx
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
import math


@dataclass
class PageBlock:
    """A single page block storing KV cache data."""
    page_id: int                    # Unique page identifier
    tokens_stored: int              # Number of tokens currently in page
    max_tokens: int                 # Page capacity (e.g., 16)
    ref_count: int                  # Number of requests sharing this page
    key_data: Optional[mx.array]    # Key tensor data [H, max_tokens, D]
    value_data: Optional[mx.array]  # Value tensor data [H, max_tokens, D]
    is_dirty: bool = False          # Whether page has been modified


@dataclass 
class PageTableEntry:
    """Entry in a logical-to-physical page mapping table."""
    logical_start: int             # Starting token position in logical sequence
    logical_end: int               # Ending token position in logical sequence
    physical_page_id: int          # Which physical page stores this data
    is_cow: bool = False           # Copy-on-write flag for shared pages


class PageAllocator:
    """Manages allocation and deallocation of memory pages."""
    
    def __init__(self, total_pages: int, page_size: int, num_heads: int, head_dim: int):
        """
        Initialize page allocator.
        
        Args:
            total_pages: Total number of pages in memory pool
            page_size: Number of tokens per page
            num_heads: Number of attention heads
            head_dim: Dimension per head
        """
        self.total_pages = total_pages
        self.page_size = page_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        
        # Free page pool - initially all pages are free
        self.free_pages: set[int] = set(range(total_pages))
        
        # Physical page storage
        self.page_blocks: Dict[int, PageBlock] = {}
        
        # Initialize all page blocks
        for page_id in range(total_pages):
            self.page_blocks[page_id] = PageBlock(
                page_id=page_id,
                tokens_stored=0,
                max_tokens=page_size,
                ref_count=0,
                key_data=mx.zeros((num_heads, page_size, head_dim), dtype=mx.float16),
                value_data=mx.zeros((num_heads, page_size, head_dim), dtype=mx.float16)
            )
    
    def allocate_page(self) -> Optional[int]:
        """
        Allocate a free page.
        
        Returns:
            Page ID if successful, None if out of memory
        """
        if not self.free_pages:
            return None
            
        page_id = self.free_pages.pop()
        page_block = self.page_blocks[page_id]
        page_block.ref_count = 1
        page_block.tokens_stored = 0
        page_block.is_dirty = False
        
        return page_id
    
    def free_page(self, page_id: int):
        """
        Free an allocated page by decrementing reference count.
        
        Args:
            page_id: ID of page to free
        """
        if page_id not in self.page_blocks:
            return
            
        page_block = self.page_blocks[page_id]
        page_block.ref_count -= 1
        
        if page_block.ref_count <= 0:
            # Return to free pool
            page_block.ref_count = 0
            page_block.tokens_stored = 0
            page_block.is_dirty = False
            # Clear data
            page_block.key_data.fill(0)
            page_block.value_data.fill(0)
            self.free_pages.add(page_id)
    
    def share_page(self, page_id: int):
        """
        Increment reference count for page sharing.
        
        Args:
            page_id: ID of page to share
        """
        if page_id in self.page_blocks:
            self.page_blocks[page_id].ref_count += 1
    
    def get_memory_usage(self) -> dict:
        """Get current memory usage statistics."""
        allocated_pages = len(self.page_blocks) - len(self.free_pages)
        total_memory = self.total_pages * self.page_size * self.num_heads * self.head_dim * 2 * 2  # K+V, float16
        used_memory = allocated_pages * self.page_size * self.num_heads * self.head_dim * 2 * 2
        
        return {
            "total_pages": self.total_pages,
            "allocated_pages": allocated_pages,
            "free_pages": len(self.free_pages),
            "utilization": allocated_pages / self.total_pages,
            "total_memory_mb": total_memory / (1024 * 1024),
            "used_memory_mb": used_memory / (1024 * 1024)
        }


class PageTable:
    """Manages logical-to-physical address translation for a sequence."""
    
    def __init__(self, page_size: int):
        """
        Initialize page table.
        
        Args:
            page_size: Number of tokens per page
        """
        self.page_size = page_size
        self.entries: List[PageTableEntry] = []
        self.sequence_length = 0
    
    def logical_to_physical(self, logical_pos: int) -> Tuple[int, int]:
        """
        Convert logical token position to (page_id, offset_in_page).
        
        Args:
            logical_pos: Logical position in sequence
            
        Returns:
            Tuple of (physical_page_id, page_offset)
        """
        if logical_pos >= self.sequence_length:
            raise ValueError(f"Logical position {logical_pos} exceeds sequence length {self.sequence_length}")
        
        for entry in self.entries:
            if entry.logical_start <= logical_pos < entry.logical_end:
                page_offset = logical_pos - entry.logical_start
                return entry.physical_page_id, page_offset
        
        raise ValueError(f"No page mapping found for logical position {logical_pos}")
    
    def append_tokens(self, num_tokens: int, page_allocator: PageAllocator) -> bool:
        """
        Extend the logical sequence by allocating new pages as needed.
        
        Args:
            num_tokens: Number of tokens to add
            page_allocator: Page allocator to use
            
        Returns:
            True if successful, False if out of memory
        """
        tokens_remaining = num_tokens
        
        while tokens_remaining > 0:
            if self.entries and self.entries[-1].logical_end - self.entries[-1].logical_start < page_allocator.page_size:
                # Current page has space
                last_entry = self.entries[-1]
                available_space = page_allocator.page_size - (last_entry.logical_end - last_entry.logical_start)
                tokens_to_add = min(tokens_remaining, available_space)
                
                last_entry.logical_end += tokens_to_add
                self.sequence_length += tokens_to_add
                tokens_remaining -= tokens_to_add
                
                # Update page block
                page_block = page_allocator.page_blocks[last_entry.physical_page_id]
                page_block.tokens_stored += tokens_to_add
            else:
                # Need new page
                page_id = page_allocator.allocate_page()
                if page_id is None:
                    return False  # Out of memory
                
                tokens_to_add = min(tokens_remaining, page_allocator.page_size)
                
                entry = PageTableEntry(
                    logical_start=self.sequence_length,
                    logical_end=self.sequence_length + tokens_to_add,
                    physical_page_id=page_id
                )
                
                self.entries.append(entry)
                self.sequence_length += tokens_to_add
                tokens_remaining -= tokens_to_add
                
                # Update page block
                page_block = page_allocator.page_blocks[page_id]
                page_block.tokens_stored = tokens_to_add
        
        return True
    
    def copy_prefix(self, source_table: 'PageTable', prefix_length: int, page_allocator: PageAllocator):
        """
        Copy prefix pages from another page table (for copy-on-write sharing).
        
        Args:
            source_table: Source page table to copy from
            prefix_length: Number of tokens to copy
            page_allocator: Page allocator for reference counting
        """
        if prefix_length > source_table.sequence_length:
            prefix_length = source_table.sequence_length
        
        tokens_copied = 0
        for entry in source_table.entries:
            if tokens_copied >= prefix_length:
                break
                
            entry_length = entry.logical_end - entry.logical_start
            tokens_to_copy = min(entry_length, prefix_length - tokens_copied)
            
            if tokens_to_copy == entry_length:
                # Copy entire entry and share the page
                new_entry = PageTableEntry(
                    logical_start=tokens_copied,
                    logical_end=tokens_copied + tokens_to_copy,
                    physical_page_id=entry.physical_page_id,
                    is_cow=True
                )
                self.entries.append(new_entry)
                page_allocator.share_page(entry.physical_page_id)
            else:
                # Partial copy - need new page
                page_id = page_allocator.allocate_page()
                if page_id is None:
                    break
                
                new_entry = PageTableEntry(
                    logical_start=tokens_copied,
                    logical_end=tokens_copied + tokens_to_copy,
                    physical_page_id=page_id
                )
                self.entries.append(new_entry)
                
                # Copy data to new page
                source_page = page_allocator.page_blocks[entry.physical_page_id]
                target_page = page_allocator.page_blocks[page_id]
                
                start_offset = 0
                target_page.key_data[:, :tokens_to_copy, :] = source_page.key_data[:, start_offset:start_offset + tokens_to_copy, :]
                target_page.value_data[:, :tokens_to_copy, :] = source_page.value_data[:, start_offset:start_offset + tokens_to_copy, :]
                target_page.tokens_stored = tokens_to_copy
            
            tokens_copied += tokens_to_copy
        
        self.sequence_length = tokens_copied
    
    def copy_on_write(self, logical_pos: int, page_allocator: PageAllocator) -> bool:
        """
        Perform copy-on-write when modifying a shared page.
        
        Args:
            logical_pos: Position being modified
            page_allocator: Page allocator
            
        Returns:
            True if COW was performed or not needed, False if out of memory
        """
        page_id, page_offset = self.logical_to_physical(logical_pos)
        page_block = page_allocator.page_blocks[page_id]
        
        # Find the entry for this page
        entry = None
        for e in self.entries:
            if e.physical_page_id == page_id:
                entry = e
                break
        
        if entry is None or not entry.is_cow or page_block.ref_count <= 1:
            return True  # No COW needed
        
        # Need to create a private copy
        new_page_id = page_allocator.allocate_page()
        if new_page_id is None:
            return False  # Out of memory
        
        # Copy data to new page
        new_page = page_allocator.page_blocks[new_page_id]
        new_page.key_data[:, :page_block.tokens_stored, :] = page_block.key_data[:, :page_block.tokens_stored, :]
        new_page.value_data[:, :page_block.tokens_stored, :] = page_block.value_data[:, :page_block.tokens_stored, :]
        new_page.tokens_stored = page_block.tokens_stored
        
        # Update entry
        entry.physical_page_id = new_page_id
        entry.is_cow = False
        
        # Decrement ref count of original page
        page_allocator.free_page(page_id)
        
        return True


def paged_attention(
    query: mx.array,                    # [batch, heads, seq_len, head_dim]
    page_tables: List[PageTable],       # Page mapping for each request in batch
    page_allocator: PageAllocator,      # Physical page storage
    scale: Optional[float] = None,
) -> mx.array:
    """
    Compute attention using paged KV cache.
    
    Args:
        query: Query tensor [batch, heads, seq_len, head_dim]
        page_tables: Page table for each request
        page_allocator: Page allocator with physical storage
        scale: Attention scale factor
        
    Returns:
        Attention output [batch, heads, seq_len, head_dim]
    """
    batch_size, num_heads, seq_len, head_dim = query.shape
    
    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)
    
    # Gather keys and values from pages for each request
    batch_keys = []
    batch_values = []
    max_context_len = 0
    
    for batch_idx, page_table in enumerate(page_tables):
        # Determine sequence length for this request
        context_len = page_table.sequence_length
        max_context_len = max(max_context_len, context_len)
        
        # Gather K,V from pages
        request_keys = mx.zeros((num_heads, context_len, head_dim), dtype=query.dtype)
        request_values = mx.zeros((num_heads, context_len, head_dim), dtype=query.dtype)
        
        for entry in page_table.entries:
            page_block = page_allocator.page_blocks[entry.physical_page_id]
            entry_len = entry.logical_end - entry.logical_start
            
            # Copy from page to request tensors
            request_keys[:, entry.logical_start:entry.logical_end, :] = \
                page_block.key_data[:, :entry_len, :].astype(query.dtype)
            request_values[:, entry.logical_start:entry.logical_end, :] = \
                page_block.value_data[:, :entry_len, :].astype(query.dtype)
        
        batch_keys.append(request_keys)
        batch_values.append(request_values)
    
    # Pad sequences to same length for batch processing
    padded_keys = mx.zeros((batch_size, num_heads, max_context_len, head_dim), dtype=query.dtype)
    padded_values = mx.zeros((batch_size, num_heads, max_context_len, head_dim), dtype=query.dtype)
    
    for batch_idx, (keys, values) in enumerate(zip(batch_keys, batch_values)):
        seq_len = keys.shape[1]
        padded_keys[batch_idx, :, :seq_len, :] = keys
        padded_values[batch_idx, :, :seq_len, :] = values
    
    # Compute attention scores
    scores = mx.matmul(query, padded_keys.transpose(0, 1, 3, 2)) * scale
    
    # Apply causal mask
    causal_mask = mx.triu(mx.full((seq_len, max_context_len), -mx.inf), k=max_context_len - seq_len + 1)
    scores = scores + causal_mask[None, None, :, :]
    
    # Apply softmax
    attn_weights = mx.softmax(scores, axis=-1)
    
    # Apply attention to values
    output = mx.matmul(attn_weights, padded_values)
    
    return output


def analyze_memory_efficiency(
    traditional_batch_size: int,
    traditional_max_seq_len: int,
    paged_sequence_lengths: List[int],
    page_size: int,
    num_heads: int,
    head_dim: int
) -> dict:
    """
    Analyze memory efficiency comparison between traditional and paged attention.
    
    Returns:
        Dictionary with memory usage statistics
    """
    # Traditional memory usage (static allocation)
    traditional_memory = traditional_batch_size * traditional_max_seq_len * num_heads * head_dim * 2 * 2  # K+V, float16
    
    # Paged memory usage (only allocated pages)
    total_pages_needed = sum((length + page_size - 1) // page_size for length in paged_sequence_lengths)
    paged_memory = total_pages_needed * page_size * num_heads * head_dim * 2 * 2
    
    # Calculate efficiency metrics
    actual_tokens = sum(paged_sequence_lengths)
    traditional_tokens = traditional_batch_size * traditional_max_seq_len
    
    return {
        "traditional_memory_mb": traditional_memory / (1024 * 1024),
        "paged_memory_mb": paged_memory / (1024 * 1024),
        "memory_savings_ratio": traditional_memory / paged_memory,
        "memory_savings_percent": (traditional_memory - paged_memory) / traditional_memory * 100,
        "traditional_utilization": actual_tokens / traditional_tokens,
        "paged_utilization": actual_tokens / (total_pages_needed * page_size),
        "pages_needed": total_pages_needed,
        "avg_sequence_length": sum(paged_sequence_lengths) / len(paged_sequence_lengths)
    }
