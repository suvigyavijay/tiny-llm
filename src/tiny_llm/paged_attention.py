from typing import List

class Page:
    def __init__(self, id: int):
        self.id = id

class PageTable:
    def __init__(self, seq_id: int):
        self.seq_id = seq_id
        self.pages: List[Page] = []

    def append(self, page: Page):
        self.pages.append(page)

class CacheManager:
    def __init__(self, num_pages: int, page_size: int, head_dim: int, num_heads: int):
        # TODO: implement
        pass

    def add_sequence(self, seq_id: int):
        # TODO: implement
        pass

    def extend_sequence(self, seq_id: int):
        # TODO: implement
        pass

    def remove_sequence(self, seq_id: int):
        # TODO: implement
        pass

    def get_sequence_page_table(self, seq_id: int) -> PageTable:
        # TODO: implement
        pass

    def get_num_free_pages(self) -> int:
        # TODO: implement
        pass
