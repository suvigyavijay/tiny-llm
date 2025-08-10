import pytest
from tiny_llm_ref.paged_attention import Page, PageTable, CacheManager


class TestPagedAttention:
    @pytest.mark.parametrize("num_pages", [10, 100, 1000])
    @pytest.mark.parametrize("page_size", [8, 16, 32])
    def test_cache_manager_init(self, num_pages, page_size):
        manager = CacheManager(num_pages=num_pages, page_size=page_size, head_dim=64, num_heads=8)
        assert len(manager.free_pages) == num_pages
        assert manager.free_pages[0].page_size == page_size

    def test_allocate_free_all_pages(self):
        manager = CacheManager(num_pages=10, page_size=16, head_dim=64, num_heads=8)
        pages = [manager.allocate_page() for _ in range(10)]
        assert len(manager.free_pages) == 0
        for page in pages:
            manager.free_page(page)
        assert len(manager.free_pages) == 10

    @pytest.mark.parametrize("num_sequences", [1, 5, 10])
    def test_add_multiple_sequences(self, num_sequences):
        manager = CacheManager(num_pages=num_sequences, page_size=16, head_dim=64, num_heads=8)
        for i in range(num_sequences):
            manager.add_sequence(seq_id=i)
        assert len(manager.page_tables) == num_sequences

    @pytest.mark.parametrize("extend_sizes", [[2, 3], [1, 1, 1, 1, 1]])
    def test_extend_sequence_multiple_times(self, extend_sizes):
        manager = CacheManager(num_pages=sum(extend_sizes), page_size=16, head_dim=64, num_heads=8)
        for size in extend_sizes:
            manager.extend_sequence(seq_id=1, num_pages=size)
        
        page_table = manager.get_sequence_page_table(seq_id=1)
        pages = page_table.get_sequence_pages(seq_id=1)
        assert len(pages) == sum(extend_sizes)

    def test_remove_sequence(self):
        manager = CacheManager(num_pages=10, page_size=16, head_dim=64, num_heads=8)
        manager.extend_sequence(seq_id=1, num_pages=5)
        manager.remove_sequence(seq_id=1)
        assert 1 not in manager.page_tables
        assert len(manager.free_pages) == 10

    def test_get_invalid_sequence(self):
        manager = CacheManager(num_pages=10, page_size=16, head_dim=64, num_heads=8)
        with pytest.raises(KeyError):
            manager.get_sequence_page_table(seq_id=1)
