import mlx.core as mx
from typing import List, Tuple


class VectorStore:
    """
    A simple in-memory vector store for RAG.
    """
    def __init__(self):
        self.embeddings: List[mx.array] = []
        self.texts: List[str] = []

    def add(self, text: str, embedding: mx.array):
        """
        Add a text and its embedding to the store.
        """
        pass

    def search(self, query_embedding: mx.array, k: int = 1) -> List[Tuple[str, float]]:
        """
        Find top-k most similar texts to the query_embedding using cosine similarity.
        
        Returns:
            List of (text, score) tuples.
        """
        pass
