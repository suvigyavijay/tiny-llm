"""
Tests for Week 3, Day 2: Paged Attention - Part 2 (Advanced Features).

Tests advanced paged attention features including dynamic page management,
memory pool optimization, and multi-tenant scenarios.
"""

import pytest
import mlx.core as mx
import time
from src.tiny_llm_ref.paged_attention import (
    PageBlock, PageTableEntry, PageAllocator, PageTable, 
    paged_attention, analyze_memory_efficiency
)


class AdvancedPageAllocator(PageAllocator):
    """Extended page allocator with advanced features for Part 2."""
    
    def __init__(self, total_pages: int, page_size: int, num_heads: int, head_dim: int):
        super().__init__(total_pages, page_size, num_heads, head_dim)
        self.allocation_strategy = "first_fit"  # first_fit, best_fit, worst_fit
        self.fragmentation_threshold = 0.1
        self.gc_enabled = True
        
    def set_allocation_strategy(self, strategy: str):
        """Set page allocation strategy."""
        assert strategy in ["first_fit", "best_fit", "worst_fit"]
        self.allocation_strategy = strategy
        
    def get_fragmentation_ratio(self) -> float:
        """Calculate memory fragmentation ratio."""
        if not self.free_pages:
            return 0.0
            
        # Simple fragmentation metric: ratio of free pages to total pages
        return len(self.free_pages) / self.total_pages
        
    def defragment(self) -> int:
        """Defragment memory by compacting allocated pages."""
        # In a real implementation, this would move data to reduce fragmentation
        # For testing, we'll simulate by reorganizing free page tracking
        if not self.gc_enabled:
            return 0
            
        initial_fragmentation = self.get_fragmentation_ratio()
        
        # Simulate defragmentation by reordering free pages
        self.free_pages = set(sorted(self.free_pages))
        
        final_fragmentation = self.get_fragmentation_ratio()
        return int((initial_fragmentation - final_fragmentation) * 100)
        
    def allocate_contiguous_pages(self, num_pages: int) -> list[int]:
        """Allocate multiple contiguous pages."""
        if num_pages <= 0:
            return []
            
        if len(self.free_pages) < num_pages:
            return []
            
        # Find contiguous sequence
        sorted_free = sorted(self.free_pages)
        
        for start_idx in range(len(sorted_free) - num_pages + 1):
            # Check if we have contiguous pages
            candidate_pages = sorted_free[start_idx:start_idx + num_pages]
            
            is_contiguous = all(
                candidate_pages[i] + 1 == candidate_pages[i + 1] 
                for i in range(len(candidate_pages) - 1)
            )
            
            if is_contiguous:
                # Allocate these pages
                allocated_pages = []
                for page_id in candidate_pages:
                    self.free_pages.remove(page_id)
                    self.page_blocks[page_id].ref_count = 1
                    allocated_pages.append(page_id)
                    
                return allocated_pages
                
        return []  # No contiguous block found


class TestAdvancedPagedAttention:
    """Test advanced paged attention features."""
    
    def test_advanced_page_allocator_strategies(self):
        """Test different allocation strategies."""
        allocator = AdvancedPageAllocator(
            total_pages=20, page_size=16, num_heads=4, head_dim=32
        )
        
        # Test different strategies
        strategies = ["first_fit", "best_fit", "worst_fit"]
        
        for strategy in strategies:
            allocator.set_allocation_strategy(strategy)
            
            # Allocate some pages
            allocated = []
            for _ in range(5):
                page_id = allocator.allocate_page()
                if page_id is not None:
                    allocated.append(page_id)
            
            assert len(allocated) == 5
            
            # Free them for next strategy test
            for page_id in allocated:
                allocator.free_page(page_id)
    
    def test_memory_fragmentation_tracking(self):
        """Test memory fragmentation tracking and defragmentation."""
        allocator = AdvancedPageAllocator(
            total_pages=20, page_size=16, num_heads=4, head_dim=32
        )
        
        # Initial fragmentation should be 1.0 (all pages free)
        initial_frag = allocator.get_fragmentation_ratio()
        assert initial_frag == 1.0
        
        # Allocate some pages to create fragmentation
        allocated = []
        for i in range(0, 10, 2):  # Allocate every other page
            page_id = allocator.allocate_page()
            allocated.append(page_id)
        
        # Fragmentation should be reduced
        mid_frag = allocator.get_fragmentation_ratio()
        assert mid_frag < initial_frag
        
        # Test defragmentation
        defrag_improvement = allocator.defragment()
        assert defrag_improvement >= 0
    
    def test_contiguous_page_allocation(self):
        """Test contiguous page allocation for large sequences."""
        allocator = AdvancedPageAllocator(
            total_pages=20, page_size=16, num_heads=4, head_dim=32
        )
        
        # Allocate contiguous blocks
        block1 = allocator.allocate_contiguous_pages(3)
        assert len(block1) == 3
        assert block1 == sorted(block1)  # Should be sorted (contiguous)
        
        # Verify contiguity
        for i in range(len(block1) - 1):
            assert block1[i + 1] == block1[i] + 1
        
        # Try to allocate another contiguous block
        block2 = allocator.allocate_contiguous_pages(5)
        assert len(block2) == 5
        
        # Should not overlap with first block
        assert not set(block1).intersection(set(block2))
        
        # Test allocation failure when not enough contiguous space
        large_block = allocator.allocate_contiguous_pages(15)  # Too large
        assert len(large_block) == 0
    
    def test_large_sequence_paged_attention(self):
        """Test paged attention with very large sequences."""
        allocator = AdvancedPageAllocator(
            total_pages=100, page_size=32, num_heads=8, head_dim=64
        )
        
        # Create large sequences
        large_sequences = [512, 768, 1024]
        page_tables = []
        
        for seq_len in large_sequences:
            page_table = PageTable(page_size=32)
            success = page_table.append_tokens(seq_len, allocator)
            assert success, f"Failed to allocate {seq_len} tokens"
            
            # Fill with realistic data patterns
            for entry in page_table.entries:
                page_block = allocator.page_blocks[entry.physical_page_id]
                # Use different patterns for different sequences
                page_block.key_data = mx.random.normal(page_block.key_data.shape) * (seq_len / 1000)
                page_block.value_data = mx.random.normal(page_block.value_data.shape) * (seq_len / 1000)
            
            page_tables.append(page_table)
        
        # Test attention computation
        batch_size = len(page_tables)
        query = mx.random.normal((batch_size, 8, 64, 64))  # Large query
        
        start_time = time.time()
        output = paged_attention(query, page_tables, allocator)
        computation_time = time.time() - start_time
        
        assert output.shape == (batch_size, 8, 64, 64)
        assert not mx.isnan(output).any()
        assert computation_time < 30.0  # Should complete in reasonable time
    
    def test_memory_pressure_scenarios(self):
        """Test paged attention under memory pressure."""
        # Small allocator to simulate memory pressure
        allocator = AdvancedPageAllocator(
            total_pages=10, page_size=16, num_heads=4, head_dim=32
        )
        
        # Try to create many sequences that together exceed memory
        page_tables = []
        successful_allocations = 0
        
        for i in range(8):  # Try to create 8 sequences of 20 tokens each
            page_table = PageTable(page_size=16)
            success = page_table.append_tokens(20, allocator)
            
            if success:
                page_tables.append(page_table)
                successful_allocations += 1
            else:
                break
        
        # Should have allocated some but not all due to memory pressure
        assert 0 < successful_allocations < 8
        
        # Test attention with what we managed to allocate
        if page_tables:
            query = mx.random.normal((len(page_tables), 4, 8, 32))
            output = paged_attention(query, page_tables, allocator)
            assert output.shape == (len(page_tables), 4, 8, 32)
    
    def test_dynamic_sequence_extension(self):
        """Test dynamic extension of sequences during generation."""
        allocator = AdvancedPageAllocator(
            total_pages=50, page_size=16, num_heads=4, head_dim=32
        )
        
        # Start with small sequences
        page_tables = []
        for i in range(3):
            page_table = PageTable(page_size=16)
            page_table.append_tokens(16, allocator)  # One page initially
            page_tables.append(page_table)
        
        # Simulate generation process with dynamic extension
        generation_steps = 5
        
        for step in range(generation_steps):
            # Extend each sequence by a few tokens
            for page_table in page_tables:
                # Simulate token generation - extend by 2-4 tokens
                new_tokens = 2 + step % 3
                success = page_table.append_tokens(new_tokens, allocator)
                
                if success:
                    # Fill new data for the extended part
                    for entry in page_table.entries:
                        if entry.logical_end > 16 + step * 3:  # New data
                            page_block = allocator.page_blocks[entry.physical_page_id]
                            page_block.key_data = mx.random.normal(page_block.key_data.shape)
                            page_block.value_data = mx.random.normal(page_block.value_data.shape)
            
            # Test attention at this step
            query = mx.random.normal((len(page_tables), 4, 4, 32))
            output = paged_attention(query, page_tables, allocator)
            assert output.shape == (len(page_tables), 4, 4, 32)
        
        # Check final sequence lengths
        for page_table in page_tables:
            assert page_table.sequence_length > 16  # Should have grown


class TestMultiTenantPagedAttention:
    """Test paged attention in multi-tenant scenarios."""
    
    def test_tenant_isolation(self):
        """Test that different tenants don't interfere with each other."""
        allocator = AdvancedPageAllocator(
            total_pages=60, page_size=16, num_heads=4, head_dim=32
        )
        
        # Create sequences for different "tenants"
        tenant_a_tables = []
        tenant_b_tables = []
        
        # Tenant A: Large sequences
        for i in range(2):
            page_table = PageTable(page_size=16)
            page_table.append_tokens(64, allocator)
            tenant_a_tables.append(page_table)
        
        # Tenant B: Many small sequences
        for i in range(4):
            page_table = PageTable(page_size=16)
            page_table.append_tokens(16, allocator)
            tenant_b_tables.append(page_table)
        
        # Fill with tenant-specific data patterns
        for page_table in tenant_a_tables:
            for entry in page_table.entries:
                page_block = allocator.page_blocks[entry.physical_page_id]
                page_block.key_data = mx.ones(page_block.key_data.shape) * 1.0  # Tenant A pattern
                page_block.value_data = mx.ones(page_block.value_data.shape) * 1.0
        
        for page_table in tenant_b_tables:
            for entry in page_table.entries:
                page_block = allocator.page_blocks[entry.physical_page_id]
                page_block.key_data = mx.ones(page_block.key_data.shape) * 2.0  # Tenant B pattern
                page_block.value_data = mx.ones(page_block.value_data.shape) * 2.0
        
        # Test attention for each tenant separately
        query_a = mx.random.normal((len(tenant_a_tables), 4, 8, 32))
        output_a = paged_attention(query_a, tenant_a_tables, allocator)
        
        query_b = mx.random.normal((len(tenant_b_tables), 4, 8, 32))
        output_b = paged_attention(query_b, tenant_b_tables, allocator)
        
        # Outputs should be different (due to different data patterns)
        assert not mx.allclose(mx.mean(output_a), mx.mean(output_b), atol=0.1)
    
    def test_prefix_sharing_across_tenants(self):
        """Test sharing common prefixes across different tenant requests."""
        allocator = AdvancedPageAllocator(
            total_pages=40, page_size=16, num_heads=4, head_dim=32
        )
        
        # Create a common prefix
        base_prefix = PageTable(page_size=16)
        base_prefix.append_tokens(32, allocator)  # Common 32-token prefix
        
        # Fill prefix with common data
        for entry in base_prefix.entries:
            page_block = allocator.page_blocks[entry.physical_page_id]
            page_block.key_data = mx.ones(page_block.key_data.shape) * 0.5
            page_block.value_data = mx.ones(page_block.value_data.shape) * 0.5
        
        # Create tenant-specific extensions
        tenant_tables = []
        for tenant_id in range(3):
            tenant_table = PageTable(page_size=16)
            
            # Copy shared prefix
            tenant_table.copy_prefix(base_prefix, 32, allocator)
            
            # Add tenant-specific extension
            tenant_table.append_tokens(16, allocator)
            
            # Fill extension with tenant-specific data
            for entry in tenant_table.entries:
                if entry.logical_start >= 32:  # Extension part
                    page_block = allocator.page_blocks[entry.physical_page_id]
                    page_block.key_data = mx.ones(page_block.key_data.shape) * (tenant_id + 1)
                    page_block.value_data = mx.ones(page_block.value_data.shape) * (tenant_id + 1)
            
            tenant_tables.append(tenant_table)
        
        # Verify memory sharing (should use fewer pages than without sharing)
        stats = allocator.get_memory_usage()
        expected_pages_without_sharing = 3 * 3  # 3 tenants * 3 pages each
        assert stats["allocated_pages"] < expected_pages_without_sharing
        
        # Test attention for all tenants
        query = mx.random.normal((len(tenant_tables), 4, 12, 32))
        output = paged_attention(query, tenant_tables, allocator)
        assert output.shape == (len(tenant_tables), 4, 12, 32)
    
    def test_priority_based_allocation(self):
        """Test priority-based page allocation for different tenant tiers."""
        allocator = AdvancedPageAllocator(
            total_pages=20, page_size=16, num_heads=4, head_dim=32
        )
        
        # Simulate high-priority tenant allocation
        high_priority_tables = []
        for i in range(2):
            page_table = PageTable(page_size=16)
            success = page_table.append_tokens(32, allocator)
            assert success, "High priority allocation should succeed"
            high_priority_tables.append(page_table)
        
        # Simulate medium-priority tenant allocation
        medium_priority_tables = []
        for i in range(2):
            page_table = PageTable(page_size=16)
            success = page_table.append_tokens(24, allocator)
            if success:
                medium_priority_tables.append(page_table)
        
        # Simulate low-priority tenant allocation (might fail due to memory pressure)
        low_priority_tables = []
        for i in range(3):
            page_table = PageTable(page_size=16)
            success = page_table.append_tokens(16, allocator)
            if success:
                low_priority_tables.append(page_table)
        
        # High priority should have succeeded
        assert len(high_priority_tables) == 2
        
        # Others may have succeeded partially
        assert len(medium_priority_tables) + len(low_priority_tables) > 0


class TestPagedAttentionOptimizations:
    """Test advanced optimizations for paged attention."""
    
    def test_cache_aware_page_layout(self):
        """Test cache-aware page layout optimization."""
        allocator = AdvancedPageAllocator(
            total_pages=32, page_size=16, num_heads=8, head_dim=64
        )
        
        # Allocate contiguous pages for better cache locality
        page_table = PageTable(page_size=16)
        
        # Use contiguous allocation for better cache performance
        contiguous_pages = allocator.allocate_contiguous_pages(4)
        assert len(contiguous_pages) == 4
        
        # Manually set up page table with contiguous pages
        tokens_per_page = 16
        for i, page_id in enumerate(contiguous_pages):
            entry = PageTableEntry(
                logical_start=i * tokens_per_page,
                logical_end=(i + 1) * tokens_per_page,
                physical_page_id=page_id
            )
            page_table.entries.append(entry)
        
        page_table.sequence_length = len(contiguous_pages) * tokens_per_page
        
        # Fill with test data
        for page_id in contiguous_pages:
            page_block = allocator.page_blocks[page_id]
            page_block.key_data = mx.random.normal(page_block.key_data.shape)
            page_block.value_data = mx.random.normal(page_block.value_data.shape)
            page_block.tokens_stored = tokens_per_page
        
        # Test attention with cache-friendly layout
        query = mx.random.normal((1, 8, 16, 64))
        output = paged_attention(query, [page_table], allocator)
        assert output.shape == (1, 8, 16, 64)
    
    def test_batch_size_optimization(self):
        """Test optimization for different batch sizes."""
        allocator = AdvancedPageAllocator(
            total_pages=100, page_size=32, num_heads=4, head_dim=32
        )
        
        # Test with various batch sizes
        batch_sizes = [1, 4, 8, 16]
        
        for batch_size in batch_sizes:
            page_tables = []
            
            for i in range(batch_size):
                page_table = PageTable(page_size=32)
                seq_len = 64 + i * 16  # Varying sequence lengths
                page_table.append_tokens(seq_len, allocator)
                
                # Fill with data
                for entry in page_table.entries:
                    page_block = allocator.page_blocks[entry.physical_page_id]
                    page_block.key_data = mx.random.normal(page_block.key_data.shape)
                    page_block.value_data = mx.random.normal(page_block.value_data.shape)
                
                page_tables.append(page_table)
            
            # Test attention computation
            query = mx.random.normal((batch_size, 4, 32, 32))
            
            start_time = time.time()
            output = paged_attention(query, page_tables, allocator)
            computation_time = time.time() - start_time
            
            assert output.shape == (batch_size, 4, 32, 32)
            assert computation_time < 20.0  # Should scale reasonably
            
            # Clean up for next iteration
            for page_table in page_tables:
                for entry in page_table.entries:
                    allocator.free_page(entry.physical_page_id)
    
    def test_memory_pool_efficiency(self):
        """Test memory pool efficiency with realistic usage patterns."""
        allocator = AdvancedPageAllocator(
            total_pages=80, page_size=32, num_heads=8, head_dim=64
        )
        
        # Simulate realistic usage: sequences of varying lengths, frequent allocation/deallocation
        active_tables = []
        total_allocations = 0
        total_deallocations = 0
        
        for round_num in range(10):
            # Allocation phase: create new sequences
            new_tables = []
            for i in range(3):
                page_table = PageTable(page_size=32)
                seq_len = 32 + (round_num * 16) + (i * 8)  # Growing sequences
                
                success = page_table.append_tokens(seq_len, allocator)
                if success:
                    new_tables.append(page_table)
                    total_allocations += 1
            
            active_tables.extend(new_tables)
            
            # Deallocation phase: remove some old sequences
            if len(active_tables) > 6:
                to_remove = active_tables[:2]
                for page_table in to_remove:
                    for entry in page_table.entries:
                        allocator.free_page(entry.physical_page_id)
                    total_deallocations += 1
                
                active_tables = active_tables[2:]
            
            # Test attention with current active sequences
            if active_tables:
                query = mx.random.normal((len(active_tables), 8, 16, 64))
                output = paged_attention(query, active_tables, allocator)
                assert output.shape == (len(active_tables), 8, 16, 64)
        
        # Check that memory pool handled dynamic allocation well
        stats = allocator.get_memory_usage()
        assert total_allocations > 10
        assert total_deallocations > 5
        assert stats["utilization"] < 1.0  # Shouldn't be completely full


if __name__ == "__main__":
    pytest.main([__file__])
