import pytest
from tiny_llm_ref.paged_attention import Page, PageTable, CacheManager

class TestPagedAttention:
    def test_cache_manager_init(self):
        manager = CacheManager(num_pages=100, page_size=16)
        assert len(manager.free_pages) == 100
        assert manager.free_pages[0].page_size == 16

    def test_allocate_free_page(self):
        manager = CacheManager(num_pages=10, page_size=16)
        page = manager.allocate_page()
        assert isinstance(page, Page)
        assert len(manager.free_pages) == 9
        manager.free_page(page)
        assert len(manager.free_pages) == 10

    def test_add_sequence(self):
        manager = CacheManager(num_pages=10, page_size=16)
        manager.add_sequence(seq_id=1)
        assert 1 in manager.page_tables
        assert isinstance(manager.page_tables[1], PageTable)

    def test_extend_sequence(self):
        manager = CacheManager(num_pages=10, page_size=16)
        manager.extend_sequence(seq_id=1, num_pages=5)
        assert len(manager.free_pages) == 5
        page_table = manager.get_sequence_page_table(seq_id=1)
        pages = page_table.get_sequence_pages(seq_id=1)
        assert len(pages) == 5

    def test_out_of_memory(self):
        manager = CacheManager(num_pages=1, page_size=16)
        manager.allocate_page()
        with pytest.raises(MemoryError):
            manager.allocate_page()
