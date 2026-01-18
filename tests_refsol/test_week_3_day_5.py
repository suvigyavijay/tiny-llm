import pytest
import mlx.core as mx
from tiny_llm_ref.rag import VectorStore


def test_vector_store_add_and_search():
    """Test basic add and search functionality."""
    store = VectorStore()
    
    # Add some vectors
    store.add("Cat", mx.array([1.0, 0.0, 0.0]))
    store.add("Dog", mx.array([0.9, 0.1, 0.0]))
    store.add("Car", mx.array([0.0, 0.0, 1.0]))
    
    # Query close to Cat
    query = mx.array([1.0, 0.05, 0.0])
    results = store.search(query, k=2)
    
    assert len(results) == 2
    assert results[0][0] == "Cat"
    assert results[1][0] == "Dog"


def test_vector_store_empty():
    """Test search on empty store."""
    store = VectorStore()
    results = store.search(mx.array([1.0, 0.0]), k=5)
    assert results == []


def test_vector_store_cosine_similarity():
    """Test that search uses cosine similarity (direction matters, not magnitude)."""
    store = VectorStore()
    
    # Add vectors with different magnitudes but same direction
    store.add("Small", mx.array([0.1, 0.1]))
    store.add("Large", mx.array([10.0, 10.0]))
    store.add("Different", mx.array([1.0, -1.0]))
    
    # Query in same direction as Small/Large
    query = mx.array([5.0, 5.0])
    results = store.search(query, k=3)
    
    # Small and Large should have same similarity (both ~1.0)
    # Different should be 0
    assert results[0][0] in ["Small", "Large"]
    assert results[1][0] in ["Small", "Large"]
    assert results[2][0] == "Different"
