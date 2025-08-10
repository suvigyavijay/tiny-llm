# Week 3 Day 1: Paged Attention - Part 1

PagedAttention is one of the most significant innovations in LLM serving, revolutionizing how we manage memory for KV caches. Inspired by virtual memory systems in operating systems, it enables efficient memory utilization and sharing, leading to dramatic improvements in serving throughput.

## The Memory Wall Problem

Traditional KV caching has severe memory limitations:

**Static Allocation Problem**:
```python
# Traditional approach: pre-allocate for max sequence length
max_seq_len = 4096
batch_size = 8
hidden_size = 4096
num_layers = 32

# Each request reserves max memory upfront
kv_cache_memory = batch_size * max_seq_len * hidden_size * num_layers * 2 * 2  # K+V, bfloat16
print(f"Memory needed: {kv_cache_memory / 1e9:.2f} GB")  # ~67 GB!
```

**Problems with static allocation**:
1. **Memory waste**: Most requests don't use full capacity
2. **Fragmentation**: Variable-length requests create memory holes  
3. **Low utilization**: Can't pack requests efficiently
4. **No sharing**: Identical prefixes stored separately

## PagedAttention Solution

PagedAttention treats KV cache like virtual memory:

**Key Concepts**:
- **Pages**: Fixed-size blocks (e.g., 16 tokens worth of KV data)
- **Page Tables**: Map logical positions to physical memory pages
- **Copy-on-Write**: Share pages for identical prefixes
- **Dynamic Allocation**: Allocate pages only as needed

**Memory Layout**:
```
Logical Sequence: [tok1][tok2][tok3][tok4][tok5][tok6]...
                     ↓
Physical Pages:   [Page 0: tok1-4][Page 1: tok5-8][Page 2: tok9-12]...
                     ↓
Page Table:       Request A: [Pg0, Pg1, Pg2, ...]
                  Request B: [Pg0, Pg4, Pg5, ...]  # Shares Pg0!
```

**Benefits**:
- **Memory efficiency**: Only allocate what's needed
- **Sharing**: Common prefixes use same physical pages
- **Flexibility**: No fixed sequence length limits
- **Defragmentation**: Pages can be allocated anywhere

**Readings**

- [vLLM PagedAttention Paper](https://arxiv.org/abs/2309.06180)
- [Virtual Memory Concepts](https://en.wikipedia.org/wiki/Virtual_memory) 
- [Copy-on-Write](https://en.wikipedia.org/wiki/Copy-on-write)

## Task 1: Understanding Page-Based Memory Layout

You will work with these files:
```
src/tiny_llm/paged_attention.py      # Page management
src/tiny_llm/page_allocator.py       # Page allocation algorithms
```

**Page Structure**:
```python
@dataclass
class PageBlock:
    page_id: int                    # Unique page identifier
    tokens_stored: int              # Number of tokens currently in page
    max_tokens: int                 # Page capacity (e.g., 16)
    ref_count: int                  # Number of requests sharing this page
    key_data: mx.array             # Key tensor data [H, max_tokens, D]
    value_data: mx.array           # Value tensor data [H, max_tokens, D]
```

**Page Table Entry**:
```python
@dataclass 
class PageTableEntry:
    logical_start: int             # Starting token position
    logical_end: int               # Ending token position  
    physical_page_id: int          # Which physical page
    is_cow: bool                   # Copy-on-write flag
```

## Task 2: Implement Basic Page Allocator

Create the page allocation system:

```python
class PageAllocator:
    def __init__(self, total_pages: int, page_size: int):
        """
        Initialize page allocator.
        
        TODO: Implement initialization
        - Create pool of free pages
        - Track allocated pages
        - Set up page metadata
        """
        pass
    
    def allocate_page(self) -> int | None:
        """
        Allocate a free page.
        
        TODO: Implement page allocation
        - Find free page from pool
        - Mark as allocated
        - Return page ID or None if out of memory
        """
        pass
    
    def free_page(self, page_id: int):
        """
        Free an allocated page.
        
        TODO: Implement page deallocation  
        - Decrement reference count
        - Return to free pool when ref_count = 0
        - Clear page data
        """
        pass
```

## Task 3: Page Table Management

Implement logical-to-physical address translation:

```python
class PageTable:
    def __init__(self, page_size: int):
        """Initialize page table for a request."""
        self.page_size = page_size
        self.entries: list[PageTableEntry] = []
    
    def logical_to_physical(self, logical_pos: int) -> tuple[int, int]:
        """
        Convert logical token position to (page_id, offset_in_page).
        
        TODO: Implement address translation
        - Find which page contains logical_pos
        - Calculate offset within that page
        - Return (physical_page_id, page_offset)
        """
        pass
    
    def append_tokens(self, num_tokens: int, page_allocator: PageAllocator):
        """
        Extend the logical sequence by allocating new pages as needed.
        
        TODO: Implement sequence extension
        - Check if current page has space
        - Allocate new page if needed
        - Update page table entries
        """
        pass
```

## Task 4: Copy-on-Write Implementation

Implement shared prefix optimization:

```python
def share_prefix(source_table: PageTable, target_table: PageTable, 
                prefix_length: int):
    """
    Share prefix pages between two requests.
    
    TODO: Implement prefix sharing
    - Copy page table entries for prefix
    - Increment reference counts
    - Mark pages as copy-on-write
    
    Example:
    - Request A: "The quick brown fox"
    - Request B: "The quick brown cat" 
    - Share first 3 pages for "The quick brown"
    """
    pass

def copy_on_write(page_table: PageTable, logical_pos: int, 
                 page_allocator: PageAllocator) -> bool:
    """
    Perform copy-on-write when modifying shared page.
    
    TODO: Implement COW
    - Check if page is shared (ref_count > 1)
    - Allocate new page if shared
    - Copy data from original page
    - Update page table to point to new page
    - Decrement ref_count of original page
    """
    pass
```

## Task 5: Memory Efficiency Analysis

Compare memory usage between approaches:

```python
def analyze_memory_efficiency():
    """Analyze memory usage for different scenarios."""
    
    # Scenario 1: Traditional static allocation
    batch_size = 16
    max_seq_len = 2048
    traditional_memory = batch_size * max_seq_len * 8  # Simplified calculation
    
    # Scenario 2: PagedAttention with variable lengths
    actual_lengths = [100, 500, 50, 1500, 200, 800, 150, 1000, 
                     300, 600, 75, 1200, 400, 900, 250, 700]
    page_size = 16
    
    # Calculate pages needed
    pages_needed = sum((length + page_size - 1) // page_size 
                      for length in actual_lengths)
    paged_memory = pages_needed * page_size * 8
    
    print(f"Traditional memory: {traditional_memory}")
    print(f"Paged memory: {paged_memory}")
    print(f"Memory savings: {(traditional_memory - paged_memory) / traditional_memory * 100:.1f}%")
    
    # Scenario 3: With prefix sharing
    # TODO: Calculate memory with shared prefixes
```

## Task 6: Integration with Attention Computation

Modify attention to work with paged memory:

```python
def paged_attention(query: mx.array, page_tables: list[PageTable], 
                   page_blocks: dict[int, PageBlock]) -> mx.array:
    """
    Compute attention with paged KV cache.
    
    TODO: Implement paged attention computation
    - Gather keys/values from multiple pages per request
    - Handle variable sequence lengths
    - Maintain causality across page boundaries
    
    Steps:
    1. For each request, gather K,V from its pages
    2. Concatenate page data to form full K,V sequences  
    3. Apply standard attention computation
    4. Handle masking for variable lengths
    """
    pass
```

## Task 7: Page Allocation Policies

Implement different allocation strategies:

```python
class FIFOPageAllocator(PageAllocator):
    """First-In-First-Out page allocation."""
    def allocate_page(self) -> int | None:
        # TODO: Implement FIFO allocation
        pass

class LRUPageAllocator(PageAllocator):
    """Least-Recently-Used page allocation."""
    def allocate_page(self) -> int | None:
        # TODO: Implement LRU allocation with eviction
        pass
        
class BuddyPageAllocator(PageAllocator):
    """Buddy system for reducing fragmentation."""
    def allocate_page(self) -> int | None:
        # TODO: Implement buddy allocation
        pass
```

## Task 8: Performance Benchmarking

Compare performance across different memory systems:

```python
def benchmark_memory_systems():
    """Compare traditional vs paged attention performance."""
    
    # Setup test scenarios
    scenarios = [
        {"name": "Short sequences", "lengths": [50, 100, 150] * 10},
        {"name": "Mixed lengths", "lengths": [100, 500, 1000, 2000] * 8},
        {"name": "Long sequences", "lengths": [1500, 2000, 2500] * 6},
    ]
    
    for scenario in scenarios:
        # Traditional allocation
        start = time.time()
        traditional_result = traditional_batched_attention(scenario["lengths"])
        traditional_time = time.time() - start
        
        # Paged allocation  
        start = time.time()
        paged_result = paged_batched_attention(scenario["lengths"])
        paged_time = time.time() - start
        
        print(f"{scenario['name']}:")
        print(f"  Traditional: {traditional_time:.3f}s")
        print(f"  Paged: {paged_time:.3f}s")
        print(f"  Speedup: {traditional_time / paged_time:.2f}x")
```

## Memory Layout Example

Here's how a batch of requests might be laid out:

**Request A**: "The quick brown fox jumps over"
**Request B**: "The quick brown cat runs fast"  
**Request C**: "Hello world"

```
Physical Memory Pages:
┌─────────────────┐
│ Page 0          │ ← "The quick brown " (shared by A & B)
│ [The][quick]    │
│ [brown][ ]      │
└─────────────────┘
┌─────────────────┐  
│ Page 1          │ ← "fox jumps over" (A only)
│ [fox][jumps]    │
│ [over][ ]       │
└─────────────────┘
┌─────────────────┐
│ Page 2          │ ← "cat runs fast" (B only)  
│ [cat][runs]     │
│ [fast][ ]       │
└─────────────────┘
┌─────────────────┐
│ Page 3          │ ← "Hello world" (C only)
│ [Hello][world]  │
│ [ ][ ]          │
└─────────────────┘

Page Tables:
Request A: [Page 0, Page 1]
Request B: [Page 0, Page 2]  
Request C: [Page 3]
```

## Testing Your Implementation

```bash
# Test page allocation
pdm run test --week 3 --day 1 -- -k page_allocation

# Test copy-on-write  
pdm run test --week 3 --day 1 -- -k copy_on_write

# Test memory efficiency
pdm run test --week 3 --day 1 -- -k memory_efficiency

# Full test suite
pdm run test --week 3 --day 1
```

At the end of this day, you should understand how PagedAttention revolutionizes memory management for LLM serving and have implemented the core page allocation and management system.

{{#include copyright.md}}
