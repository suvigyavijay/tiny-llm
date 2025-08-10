from typing import List, Dict
import mlx.core as mx

class Page:
    def __init__(self, page_id: int, page_size: int, head_dim: int, num_heads: int):
        self.page_id = page_id
        self.page_size = page_size
        self.key_cache = mx.zeros((num_heads, page_size, head_dim))
        self.value_cache = mx.zeros((num_heads, page_size, head_dim))

class PageTable:
    def __init__(self):
        self.table: Dict[int, List[Page]] = {}

    def add_sequence(self, seq_id: int):
        self.table[seq_id] = []

    def add_page_to_sequence(self, seq_id: int, page: Page):
        if seq_id not in self.table:
            self.add_sequence(seq_id)
        self.table[seq_id].append(page)

    def get_sequence_pages(self, seq_id: int) -> List[Page]:
        return self.table.get(seq_id, [])

class CacheManager:
    def __init__(self, num_pages: int, page_size: int, head_dim: int, num_heads: int):
        self.free_pages: List[Page] = [Page(i, page_size, head_dim, num_heads) for i in range(num_pages)]
        self.page_tables: Dict[int, PageTable] = {}

    def allocate_page(self) -> Page:
        if not self.free_pages:
            # For now, we'll assume we don't run out of memory.
            # In a real system, this would trigger a more complex memory management strategy.
            raise MemoryError("Out of memory")
        return self.free_pages.pop(0)

    def free_page(self, page: Page):
        self.free_pages.append(page)

    def add_sequence(self, seq_id: int):
        self.page_tables[seq_id] = PageTable()

    def extend_sequence(self, seq_id: int, num_pages: int = 1):
        if seq_id not in self.page_tables:
            self.add_sequence(seq_id)
        
        for _ in range(num_pages):
            page = self.allocate_page()
            self.page_tables[seq_id].add_page_to_sequence(seq_id, page)

    def remove_sequence(self, seq_id: int):
        if seq_id in self.page_tables:
            page_table = self.page_tables.pop(seq_id)
            pages = page_table.get_sequence_pages(seq_id)
            for page in pages:
                self.free_page(page)

    def get_sequence_page_table(self, seq_id: int) -> PageTable:
        if seq_id not in self.page_tables:
            raise KeyError(f"Sequence ID {seq_id} not found")
        return self.page_tables.get(seq_id)

    def get_sequence_page_ids(self, seq_id: int) -> List[int]:
        if seq_id not in self.page_tables:
            raise KeyError(f"Sequence ID {seq_id} not found")
        return [page.page_id for page in self.page_tables[seq_id].get_sequence_pages(seq_id)]

    def get_cache(self) -> (mx.array, mx.array):
        # This is not an efficient way to get the cache, but it's fine for testing.
        all_pages = self.free_pages + [page for pt in self.page_tables.values() for pages in pt.table.values() for page in pages]
        all_pages.sort(key=lambda p: p.page_id)
        
        k_cache = mx.stack([p.key_cache for p in all_pages])
        v_cache = mx.stack([p.value_cache for p in all_pages])
        
        return k_cache, v_cache
