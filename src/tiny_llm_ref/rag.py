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
        self.embeddings.append(embedding)
        self.texts.append(text)

    def search(self, query_embedding: mx.array, k: int = 1) -> List[Tuple[str, float]]:
        if not self.embeddings:
            return []
            
        # Stack embeddings: [N, D]
        db = mx.stack(self.embeddings)
        
        # Normalize for cosine similarity
        db_norm = db / mx.linalg.norm(db, axis=1, keepdims=True)
        q_norm = query_embedding / mx.linalg.norm(query_embedding)
        
        # Scores: [N]
        scores = db_norm @ q_norm
        
        # Top K
        indices = mx.argsort(scores)[::-1][:k]
        
        results = []
        for idx in indices.tolist():
            results.append((self.texts[idx], scores[idx].item()))
            
        return results
