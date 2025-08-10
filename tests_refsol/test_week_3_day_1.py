"""
Tests for Week 3, Day 1: Paged Attention implementation.

Tests the paged attention system with page management, memory efficiency,
and copy-on-write sharing mechanisms.
"""

import pytest
import mlx.core as mx
import time
from src.tiny_llm_ref.paged_attention import (
    PageBlock, PageTableEntry, PageAllocator, PageTable, 
    paged_attention, analyze_memory_efficiency
)


class TestPageBlock:
    """Test PageBlock data structure."""
    
    def test_page_block_creation(self):
        """Test creating a page block."""
        page = PageBlock(
            page_id=0,
            tokens_stored=0,
            max_tokens=16,
            ref_count=1,
            key_data=mx.zeros((8, 16, 64)),
            value_data=mx.zeros((8, 16, 64))
        )
        
        assert page.page_id == 0
        assert page.tokens_stored == 0
        assert page.max_tokens == 16
        assert page.ref_count == 1
        assert not page.is_dirty


class TestPageAllocator:
    """Test PageAllocator functionality."""
    
    def test_allocator_initialization(self):
        """Test page allocator initialization."""
        allocator = PageAllocator(
            total_pages=100,
            page_size=16,
            num_heads=8,
            head_dim=64
        )
        
        assert allocator.total_pages == 100
        assert allocator.page_size == 16
        assert len(allocator.free_pages) == 100
        assert len(allocator.page_blocks) == 100
        
        # All pages should start free
        for page_id in range(100):
            assert page_id in allocator.free_pages
            assert allocator.page_blocks[page_id].ref_count == 0
    
    def test_page_allocation(self):
        """Test allocating pages."""
        allocator = PageAllocator(total_pages=10, page_size=16, num_heads=8, head_dim=64)
        
        # Allocate a page
        page_id = allocator.allocate_page()
        assert page_id is not None
        assert page_id not in allocator.free_pages
        assert allocator.page_blocks[page_id].ref_count == 1
        
        # Allocate all remaining pages
        allocated_pages = [page_id]
        for _ in range(9):
            page_id = allocator.allocate_page()
            assert page_id is not None
            allocated_pages.append(page_id)
        
        # Should be out of pages now
        assert allocator.allocate_page() is None
    
    def test_page_deallocation(self):
        """Test freeing pages."""
        allocator = PageAllocator(total_pages=10, page_size=16, num_heads=8, head_dim=64)
        
        # Allocate and free a page
        page_id = allocator.allocate_page()
        assert page_id not in allocator.free_pages
        
        allocator.free_page(page_id)
        assert page_id in allocator.free_pages
        assert allocator.page_blocks[page_id].ref_count == 0
    
    def test_page_sharing(self):
        """Test page sharing with reference counting."""
        allocator = PageAllocator(total_pages=10, page_size=16, num_heads=8, head_dim=64)
        
        page_id = allocator.allocate_page()
        assert allocator.page_blocks[page_id].ref_count == 1
        
        # Share the page
        allocator.share_page(page_id)
        assert allocator.page_blocks[page_id].ref_count == 2
        
        # Free once - should still be allocated
        allocator.free_page(page_id)
        assert allocator.page_blocks[page_id].ref_count == 1
        assert page_id not in allocator.free_pages
        
        # Free again - should return to free pool
        allocator.free_page(page_id)
        assert allocator.page_blocks[page_id].ref_count == 0
        assert page_id in allocator.free_pages
    
    def test_memory_usage_stats(self):
        """Test memory usage statistics."""
        allocator = PageAllocator(total_pages=10, page_size=16, num_heads=8, head_dim=64)
        
        stats = allocator.get_memory_usage()
        assert stats["total_pages"] == 10
        assert stats["allocated_pages"] == 0
        assert stats["free_pages"] == 10
        assert stats["utilization"] == 0.0
        
        # Allocate some pages
        for _ in range(5):
            allocator.allocate_page()
        
        stats = allocator.get_memory_usage()
        assert stats["allocated_pages"] == 5
        assert stats["free_pages"] == 5
        assert stats["utilization"] == 0.5


class TestPageTable:
    """Test PageTable logical-to-physical mapping."""
    
    def test_page_table_initialization(self):
        """Test page table initialization."""
        page_table = PageTable(page_size=16)
        assert page_table.page_size == 16
        assert len(page_table.entries) == 0
        assert page_table.sequence_length == 0
    
    def test_append_tokens(self):
        """Test appending tokens to page table."""
        allocator = PageAllocator(total_pages=10, page_size=16, num_heads=8, head_dim=64)
        page_table = PageTable(page_size=16)
        
        # Append tokens within one page
        success = page_table.append_tokens(10, allocator)
        assert success
        assert page_table.sequence_length == 10
        assert len(page_table.entries) == 1
        
        # Append more tokens to fill the page
        success = page_table.append_tokens(6, allocator)
        assert success
        assert page_table.sequence_length == 16
        assert len(page_table.entries) == 1
        
        # Append tokens requiring new page
        success = page_table.append_tokens(10, allocator)
        assert success
        assert page_table.sequence_length == 26
        assert len(page_table.entries) == 2
    
    def test_logical_to_physical_mapping(self):
        """Test logical to physical address translation."""
        allocator = PageAllocator(total_pages=10, page_size=16, num_heads=8, head_dim=64)
        page_table = PageTable(page_size=16)
        
        # Set up page table with multiple pages
        page_table.append_tokens(20, allocator)
        
        # Test mappings
        page_id, offset = page_table.logical_to_physical(5)
        assert offset == 5
        
        page_id, offset = page_table.logical_to_physical(15)
        assert offset == 15
        
        page_id, offset = page_table.logical_to_physical(18)
        assert offset == 2  # 18 - 16 = 2 in second page
        
        # Test invalid position
        with pytest.raises(ValueError):
            page_table.logical_to_physical(25)
    
    def test_copy_prefix(self):
        """Test copying prefix from another page table."""
        allocator = PageAllocator(total_pages=20, page_size=16, num_heads=8, head_dim=64)
        
        # Create source page table
        source_table = PageTable(page_size=16)
        source_table.append_tokens(30, allocator)
        
        # Create target page table and copy prefix
        target_table = PageTable(page_size=16)
        target_table.copy_prefix(source_table, 20, allocator)
        
        assert target_table.sequence_length == 20
        # Should have shared some pages
        assert len(target_table.entries) >= 1
    
    def test_copy_on_write(self):
        """Test copy-on-write mechanism."""
        allocator = PageAllocator(total_pages=20, page_size=16, num_heads=8, head_dim=64)
        
        # Create source page table
        source_table = PageTable(page_size=16)
        source_table.append_tokens(16, allocator)
        
        # Copy prefix (should share pages)
        target_table = PageTable(page_size=16)
        target_table.copy_prefix(source_table, 16, allocator)
        
        # Get original page id
        original_page_id, _ = target_table.logical_to_physical(5)
        original_ref_count = allocator.page_blocks[original_page_id].ref_count
        
        # Perform copy-on-write
        success = target_table.copy_on_write(5, allocator)
        assert success
        
        # Should have new page id after COW
        new_page_id, _ = target_table.logical_to_physical(5)
        assert new_page_id != original_page_id
        assert allocator.page_blocks[original_page_id].ref_count == original_ref_count - 1


class TestPagedAttention:
    """Test paged attention computation."""
    
    def test_paged_attention_basic(self):
        """Test basic paged attention computation."""
        # Set up page allocator and tables
        allocator = PageAllocator(total_pages=20, page_size=16, num_heads=4, head_dim=32)
        
        # Create page tables for batch
        page_tables = []
        for seq_len in [24, 32, 16]:
            page_table = PageTable(page_size=16)
            page_table.append_tokens(seq_len, allocator)
            
            # Fill some dummy KV data
            for entry in page_table.entries:
                page_block = allocator.page_blocks[entry.physical_page_id]
                page_block.key_data = mx.random.normal(page_block.key_data.shape)
                page_block.value_data = mx.random.normal(page_block.value_data.shape)
            
            page_tables.append(page_table)
        
        # Create query tensor
        batch_size = len(page_tables)
        num_heads = 4
        seq_len = 8
        head_dim = 32
        query = mx.random.normal((batch_size, num_heads, seq_len, head_dim))
        
        # Compute paged attention
        output = paged_attention(query, page_tables, allocator)
        
        assert output.shape == (batch_size, num_heads, seq_len, head_dim)
        assert not mx.isnan(output).any()
    
    def test_paged_attention_empty_cache(self):
        """Test paged attention with empty cache."""
        allocator = PageAllocator(total_pages=10, page_size=16, num_heads=4, head_dim=32)
        page_tables = [PageTable(page_size=16)]  # Empty page table
        
        query = mx.random.normal((1, 4, 8, 32))
        
        # Should handle empty cache gracefully
        output = paged_attention(query, page_tables, allocator)
        assert output.shape == (1, 4, 8, 32)


class TestMemoryEfficiency:
    """Test memory efficiency analysis."""
    
    def test_memory_efficiency_analysis(self):
        """Test memory efficiency comparison."""
        # Test parameters
        traditional_batch_size = 8
        traditional_max_seq_len = 1024
        paged_sequence_lengths = [512, 256, 768, 128, 900, 300, 600, 400]
        page_size = 64
        num_heads = 16
        head_dim = 64
        
        efficiency = analyze_memory_efficiency(
            traditional_batch_size,
            traditional_max_seq_len,
            paged_sequence_lengths,
            page_size,
            num_heads,
            head_dim
        )
        
        assert "traditional_memory_mb" in efficiency
        assert "paged_memory_mb" in efficiency
        assert "memory_savings_ratio" in efficiency
        assert "memory_savings_percent" in efficiency
        assert efficiency["memory_savings_ratio"] > 1.0  # Should save memory
        assert efficiency["memory_savings_percent"] > 0
    
    def test_memory_efficiency_edge_cases(self):
        """Test memory efficiency with edge cases."""
        # Case where paged might use more memory (very short sequences)
        efficiency = analyze_memory_efficiency(
            traditional_batch_size=4,
            traditional_max_seq_len=64,
            paged_sequence_lengths=[60, 62, 58, 61],  # Almost full utilization
            page_size=64,
            num_heads=8,
            head_dim=32
        )
        
        # Should still compute valid metrics
        assert efficiency["memory_savings_ratio"] > 0
        assert "paged_utilization" in efficiency


class TestPagedAttentionIntegration:
    """Integration tests for complete paged attention system."""
    
    def test_concurrent_requests(self):
        """Test handling multiple concurrent requests."""
        allocator = PageAllocator(total_pages=50, page_size=16, num_heads=8, head_dim=64)
        
        # Create multiple requests with shared prefixes
        base_page_table = PageTable(page_size=16)
        base_page_table.append_tokens(32, allocator)  # Common prefix
        
        # Fill base with dummy data
        for entry in base_page_table.entries:
            page_block = allocator.page_blocks[entry.physical_page_id]
            page_block.key_data = mx.random.normal(page_block.key_data.shape)
            page_block.value_data = mx.random.normal(page_block.value_data.shape)
        
        # Create derived requests sharing the prefix
        derived_tables = []
        for i in range(5):
            derived_table = PageTable(page_size=16)
            derived_table.copy_prefix(base_page_table, 32, allocator)
            derived_table.append_tokens(8 + i * 4, allocator)  # Different extensions
            
            # Fill new parts with dummy data
            for entry in derived_table.entries:
                if entry.logical_start >= 32:  # New parts only
                    page_block = allocator.page_blocks[entry.physical_page_id]
                    page_block.key_data = mx.random.normal(page_block.key_data.shape)
                    page_block.value_data = mx.random.normal(page_block.value_data.shape)
            
            derived_tables.append(derived_table)
        
        # Test attention computation
        batch_size = len(derived_tables)
        query = mx.random.normal((batch_size, 8, 12, 64))
        
        output = paged_attention(query, derived_tables, allocator)
        assert output.shape == (batch_size, 8, 12, 64)
        
        # Verify memory sharing
        stats = allocator.get_memory_usage()
        # Should use fewer pages than without sharing
        expected_pages_without_sharing = 5 * 3  # 5 requests * ~3 pages each
        assert stats["allocated_pages"] < expected_pages_without_sharing
    
    def test_memory_pressure(self):
        """Test behavior under memory pressure."""
        # Small allocator to test out-of-memory scenarios
        allocator = PageAllocator(total_pages=5, page_size=16, num_heads=4, head_dim=32)
        
        page_tables = []
        
        # Try to allocate more than available
        for i in range(3):
            page_table = PageTable(page_size=16)
            success = page_table.append_tokens(20, allocator)  # Needs 2 pages each
            
            if success:
                page_tables.append(page_table)
            else:
                break
        
        # Should have created some tables but not all
        assert len(page_tables) <= 3
        
        # Should still be able to compute attention with available tables
        if page_tables:
            query = mx.random.normal((len(page_tables), 4, 8, 32))
            output = paged_attention(query, page_tables, allocator)
            assert output.shape[0] == len(page_tables)
    
    def test_performance_comparison(self):
        """Test performance comparison with traditional attention."""
        # This is a basic performance comparison test
        allocator = PageAllocator(total_pages=100, page_size=32, num_heads=8, head_dim=64)
        
        # Set up paged attention
        page_tables = []
        for seq_len in [128, 256, 192]:
            page_table = PageTable(page_size=32)
            page_table.append_tokens(seq_len, allocator)
            
            for entry in page_table.entries:
                page_block = allocator.page_blocks[entry.physical_page_id]
                page_block.key_data = mx.random.normal(page_block.key_data.shape)
                page_block.value_data = mx.random.normal(page_block.value_data.shape)
            
            page_tables.append(page_table)
        
        batch_size = len(page_tables)
        query = mx.random.normal((batch_size, 8, 32, 64))
        
        # Time paged attention
        start_time = time.time()
        paged_output = paged_attention(query, page_tables, allocator)
        paged_time = time.time() - start_time
        
        # Verify output validity
        assert paged_output.shape == (batch_size, 8, 32, 64)
        assert not mx.isnan(paged_output).any()
        
        # Performance should be reasonable (not timing out)
        assert paged_time < 10.0  # Should complete within 10 seconds


if __name__ == "__main__":
    pytest.main([__file__])
