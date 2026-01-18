import mlx.core as mx
from typing import List, Tuple, Callable


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


class RAGPipeline:
    def __init__(self, vector_store: VectorStore, embedding_func: Callable, llm_func: Callable):
        self.store = vector_store
        self.embed = embedding_func
        self.llm = llm_func

    def query(self, question: str) -> str:
        # 1. Embed query
        q_emb = self.embed(question)
        
        # 2. Retrieve
        hits = self.store.search(q_emb, k=2)
        context = "\n".join([text for text, score in hits])
        
        # 3. Construct Prompt
        prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
        
        # 4. Generate
        return self.llm(prompt)
