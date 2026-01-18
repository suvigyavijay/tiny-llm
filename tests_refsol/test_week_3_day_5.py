import pytest
import mlx.core as mx
from .tiny_llm_base import *
from .utils import *


@pytest.mark.parametrize("stream", AVAILABLE_STREAMS, ids=AVAILABLE_STREAMS_IDS)
def test_vector_store_add_and_search(stream: mx.Stream):
    """Test adding documents and searching."""
    with mx.stream(stream):
        store = VectorStore()
        store.add("Cat", mx.array([1.0, 0.0]))
        store.add("Dog", mx.array([0.9, 0.1]))
        store.add("Car", mx.array([0.0, 1.0]))
        
        query = mx.array([1.0, 0.05])
        results = store.search(query, k=2)
        
        assert len(results) == 2
        assert results[0][0] == "Cat"


@pytest.mark.parametrize("k", [1, 3, 5])
def test_vector_store_top_k(k: int):
    """Test that search returns exactly k results."""
    store = VectorStore()
    for i in range(10):
        store.add(f"doc_{i}", mx.random.normal((4,)))
    
    query = mx.random.normal((4,))
    results = store.search(query, k=k)
    
    assert len(results) == k


def test_rag_pipeline():
    """Test end-to-end RAG pipeline."""
    store = VectorStore()
    store.add("Paris is the capital of France.", mx.array([1.0, 0.0]))
    store.add("Berlin is the capital of Germany.", mx.array([0.0, 1.0]))
    
    def mock_embed(text):
        if "France" in text:
            return mx.array([1.0, 0.0])
        return mx.array([0.0, 1.0])
        
    def mock_llm(prompt):
        if "Paris" in prompt:
            return "Paris"
        return "Unknown"
        
    rag = RAGPipeline(store, mock_embed, mock_llm)
    answer = rag.query("capital of France")
    
    assert answer == "Paris"
