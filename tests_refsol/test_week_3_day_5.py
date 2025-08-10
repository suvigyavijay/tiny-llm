"""
Tests for Week 3, Day 5: RAG Pipeline implementation.

Tests the Retrieval-Augmented Generation system including document processing,
vector search, context assembly, and complete RAG query pipeline.
"""

import pytest
import mlx.core as mx
import time
from unittest.mock import Mock, MagicMock
from src.tiny_llm_ref.rag_pipeline import (
    Document, DocumentProcessor, VectorStore, HybridRetriever,
    ContextAssembler, RAGSystem, SimpleEmbeddingModel, evaluate_rag_system
)


class MockEmbeddingModel:
    """Mock embedding model for testing."""
    
    def __init__(self, dimension=128):
        self.dimension = dimension
        self.call_count = 0
    
    def encode(self, text):
        """Create deterministic embedding based on text hash."""
        self.call_count += 1
        hash_val = hash(text)
        mx.random.seed(abs(hash_val) % (2**31))
        embedding = mx.random.normal((self.dimension,))
        return embedding / mx.linalg.norm(embedding)


class MockLLMModel:
    """Mock LLM for testing RAG generation."""
    
    def __init__(self):
        self.call_count = 0
        self.last_prompt = None
    
    def generate(self, prompt):
        """Generate mock response."""
        self.call_count += 1
        self.last_prompt = prompt
        
        # Simple mock response based on prompt content
        if "cannot find" in prompt.lower():
            return "I cannot find the answer in the provided context."
        elif "context:" in prompt.lower():
            return "Based on the provided context, the answer is: mock response."
        else:
            return "This is a mock response to your query."
    
    def stream_generate(self, prompt):
        """Generate streaming mock response."""
        response = self.generate(prompt)
        for chunk in response.split():
            yield chunk + " "


class TestDocument:
    """Test Document data structure."""
    
    def test_document_creation(self):
        """Test creating a document."""
        doc = Document(
            id="doc1",
            content="This is a test document.",
            metadata={"source": "test", "author": "tester"},
            embedding=mx.random.normal((128,))
        )
        
        assert doc.id == "doc1"
        assert doc.content == "This is a test document."
        assert doc.metadata["source"] == "test"
        assert doc.embedding.shape == (128,)
    
    def test_document_without_embedding(self):
        """Test document creation without embedding."""
        doc = Document(
            id="doc2",
            content="Another test document.",
            metadata={"type": "test"}
        )
        
        assert doc.embedding is None


class TestDocumentProcessor:
    """Test DocumentProcessor functionality."""
    
    def test_processor_initialization(self):
        """Test document processor initialization."""
        embedding_model = MockEmbeddingModel(dimension=64)
        processor = DocumentProcessor(
            embedding_model=embedding_model,
            chunk_size=100,
            overlap=20
        )
        
        assert processor.embedding_model == embedding_model
        assert processor.chunk_size == 100
        assert processor.overlap == 20
    
    def test_chunk_text_basic(self):
        """Test basic text chunking."""
        embedding_model = MockEmbeddingModel()
        processor = DocumentProcessor(embedding_model, chunk_size=50, overlap=10)
        
        text = "This is a sentence. This is another sentence. And here is a third sentence that is longer."
        chunks = processor.chunk_text(text)
        
        assert len(chunks) > 1
        assert all(len(chunk) <= 50 for chunk in chunks if chunk)
        assert all(isinstance(chunk, str) for chunk in chunks)
    
    def test_chunk_text_short(self):
        """Test chunking with text shorter than chunk size."""
        embedding_model = MockEmbeddingModel()
        processor = DocumentProcessor(embedding_model, chunk_size=200, overlap=20)
        
        text = "Short text."
        chunks = processor.chunk_text(text)
        
        assert len(chunks) == 1
        assert chunks[0] == text
    
    def test_chunk_text_overlap(self):
        """Test that chunking creates proper overlap."""
        embedding_model = MockEmbeddingModel()
        processor = DocumentProcessor(embedding_model, chunk_size=20, overlap=5)
        
        text = "A" * 50  # 50 characters
        chunks = processor.chunk_text(text)
        
        assert len(chunks) > 1
        # Check overlap exists (simplified check)
        for i in range(1, len(chunks)):
            # Should have some overlap between consecutive chunks
            assert len(chunks[i]) > 0
    
    def test_process_document(self):
        """Test complete document processing."""
        embedding_model = MockEmbeddingModel(dimension=64)
        processor = DocumentProcessor(embedding_model, chunk_size=50, overlap=10)
        
        text = "This is a test document. It has multiple sentences. This should be chunked appropriately."
        metadata = {"doc_id": "test_doc", "title": "Test Document"}
        
        documents = processor.process_document(text, metadata)
        
        assert len(documents) > 0
        assert all(isinstance(doc, Document) for doc in documents)
        assert all(doc.embedding is not None for doc in documents)
        assert all(doc.embedding.shape == (64,) for doc in documents)
        assert all("chunk_id" in doc.metadata for doc in documents)
        assert all("chunk_count" in doc.metadata for doc in documents)
        
        # Check metadata propagation
        for doc in documents:
            assert doc.metadata["doc_id"] == "test_doc"
            assert doc.metadata["title"] == "Test Document"


class TestVectorStore:
    """Test VectorStore functionality."""
    
    def test_vector_store_initialization(self):
        """Test vector store initialization."""
        store = VectorStore(dimension=128)
        
        assert store.dimension == 128
        assert len(store.documents) == 0
        assert store.embeddings is None
    
    def test_add_documents(self):
        """Test adding documents to vector store."""
        store = VectorStore(dimension=64)
        
        docs = [
            Document("1", "First doc", {}, mx.random.normal((64,))),
            Document("2", "Second doc", {}, mx.random.normal((64,))),
            Document("3", "Third doc", {}, mx.random.normal((64,)))
        ]
        
        store.add_documents(docs)
        
        assert len(store.documents) == 3
        assert store.embeddings.shape == (3, 64)
        assert store.get_document_count() == 3
    
    def test_add_documents_incremental(self):
        """Test incremental document addition."""
        store = VectorStore(dimension=32)
        
        # Add first batch
        docs1 = [Document("1", "Doc 1", {}, mx.random.normal((32,)))]
        store.add_documents(docs1)
        assert store.get_document_count() == 1
        
        # Add second batch
        docs2 = [
            Document("2", "Doc 2", {}, mx.random.normal((32,))),
            Document("3", "Doc 3", {}, mx.random.normal((32,)))
        ]
        store.add_documents(docs2)
        assert store.get_document_count() == 3
        assert store.embeddings.shape == (3, 32)
    
    def test_similarity_search(self):
        """Test similarity search functionality."""
        store = VectorStore(dimension=64)
        
        # Create documents with known embeddings
        embedding1 = mx.array([1.0] + [0.0] * 63)  # Unit vector in first dimension
        embedding2 = mx.array([0.0, 1.0] + [0.0] * 62)  # Unit vector in second dimension
        embedding3 = mx.array([0.7, 0.7] + [0.0] * 62)  # Diagonal
        
        docs = [
            Document("1", "Doc 1", {}, embedding1),
            Document("2", "Doc 2", {}, embedding2),
            Document("3", "Doc 3", {}, embedding3)
        ]
        
        store.add_documents(docs)
        
        # Query similar to first document
        query_embedding = mx.array([0.9, 0.1] + [0.0] * 62)
        query_embedding = query_embedding / mx.linalg.norm(query_embedding)
        
        results = store.similarity_search(query_embedding, top_k=2)
        
        assert len(results) == 2
        assert all(isinstance(result, tuple) for result in results)
        assert all(len(result) == 2 for result in results)
        
        doc, score = results[0]
        assert isinstance(doc, Document)
        assert isinstance(score, float)
        assert -1 <= score <= 1  # Cosine similarity bounds
    
    def test_similarity_search_empty_store(self):
        """Test similarity search on empty store."""
        store = VectorStore(dimension=64)
        
        query_embedding = mx.random.normal((64,))
        results = store.similarity_search(query_embedding, top_k=5)
        
        assert len(results) == 0
    
    def test_clear_store(self):
        """Test clearing vector store."""
        store = VectorStore(dimension=32)
        
        docs = [Document("1", "Doc", {}, mx.random.normal((32,)))]
        store.add_documents(docs)
        assert store.get_document_count() == 1
        
        store.clear()
        assert store.get_document_count() == 0
        assert len(store.documents) == 0
        assert store.embeddings is None


class TestHybridRetriever:
    """Test HybridRetriever functionality."""
    
    def test_hybrid_retriever_initialization(self):
        """Test hybrid retriever initialization."""
        store = VectorStore(dimension=64)
        retriever = HybridRetriever(store, enable_reranking=True)
        
        assert retriever.vector_store == store
        assert retriever.enable_reranking == True
        assert len(retriever.term_frequencies) == 0
    
    def test_sparse_index_building(self):
        """Test sparse retrieval index building."""
        store = VectorStore(dimension=32)
        docs = [
            Document("1", "The quick brown fox jumps", {}, mx.random.normal((32,))),
            Document("2", "The lazy dog sleeps peacefully", {}, mx.random.normal((32,))),
            Document("3", "Brown dogs are quick animals", {}, mx.random.normal((32,)))
        ]
        store.add_documents(docs)
        
        retriever = HybridRetriever(store)
        retriever._build_sparse_index()
        
        assert len(retriever.term_frequencies) == 3  # 3 documents
        assert len(retriever.document_frequencies) > 0
        assert retriever.total_documents == 3
        
        # Check that common words appear in document frequencies
        assert "the" in retriever.document_frequencies
        assert "quick" in retriever.document_frequencies
    
    def test_sparse_retrieval(self):
        """Test sparse retrieval functionality."""
        store = VectorStore(dimension=32)
        docs = [
            Document("1", "Machine learning algorithms", {}, mx.random.normal((32,))),
            Document("2", "Deep learning neural networks", {}, mx.random.normal((32,))),
            Document("3", "Natural language processing", {}, mx.random.normal((32,)))
        ]
        store.add_documents(docs)
        
        retriever = HybridRetriever(store)
        results = retriever._sparse_retrieval("machine learning", top_k=2)
        
        assert len(results) <= 2
        assert all(isinstance(result, tuple) for result in results)
        assert all(len(result) == 2 for result in results)
        
        if results:
            doc_idx, score = results[0]
            assert isinstance(doc_idx, int)
            assert isinstance(score, float)
            assert 0 <= doc_idx < len(store.documents)
    
    def test_hybrid_retrieve(self):
        """Test complete hybrid retrieval."""
        store = VectorStore(dimension=64)
        
        # Create documents with embeddings
        docs = [
            Document("1", "Python programming language", {}, mx.random.normal((64,))),
            Document("2", "Java programming tutorial", {}, mx.random.normal((64,))),
            Document("3", "Machine learning with Python", {}, mx.random.normal((64,)))
        ]
        store.add_documents(docs)
        
        retriever = HybridRetriever(store)
        
        query = "Python programming"
        query_embedding = mx.random.normal((64,))
        query_embedding = query_embedding / mx.linalg.norm(query_embedding)
        
        results = retriever.retrieve(query, query_embedding, top_k=2)
        
        assert len(results) <= 2
        assert all(isinstance(doc, Document) for doc in results)


class TestContextAssembler:
    """Test ContextAssembler functionality."""
    
    def test_context_assembler_initialization(self):
        """Test context assembler initialization."""
        assembler = ContextAssembler(max_context_length=1000)
        
        assert assembler.max_context_length == 1000
    
    def test_assemble_context_basic(self):
        """Test basic context assembly."""
        assembler = ContextAssembler(max_context_length=500)
        
        docs = [
            Document("1", "First document content.", {"source": "doc1"}),
            Document("2", "Second document content.", {"source": "doc2"}),
            Document("3", "Third document content.", {"source": "doc3"})
        ]
        
        query = "Test query"
        context = assembler.assemble_context(query, docs)
        
        assert isinstance(context, str)
        assert len(context) <= 500
        assert "First document content" in context
        assert "Document 1:" in context
    
    def test_assemble_context_length_limit(self):
        """Test context assembly respects length limits."""
        assembler = ContextAssembler(max_context_length=50)
        
        docs = [
            Document("1", "Very long document content that exceeds the limit", {}),
            Document("2", "Another long document with lots of content", {}),
            Document("3", "Third document with even more content", {})
        ]
        
        context = assembler.assemble_context("query", docs)
        
        assert len(context) <= 50
    
    def test_create_rag_prompt(self):
        """Test RAG prompt creation."""
        assembler = ContextAssembler(max_context_length=1000)
        
        query = "What is machine learning?"
        context = "Machine learning is a subset of AI. It involves algorithms that learn from data."
        
        prompt = assembler.create_rag_prompt(query, context)
        
        assert isinstance(prompt, str)
        assert query in prompt
        assert context in prompt
        assert "Context:" in prompt
        assert "Question:" in prompt
        assert "Answer:" in prompt


class TestRAGSystem:
    """Test complete RAG system."""
    
    def test_rag_system_initialization(self):
        """Test RAG system initialization."""
        llm_model = MockLLMModel()
        embedding_model = MockEmbeddingModel(dimension=64)
        vector_store = VectorStore(dimension=64)
        
        rag = RAGSystem(
            llm_model=llm_model,
            embedding_model=embedding_model,
            vector_store=vector_store,
            max_context_length=2000
        )
        
        assert rag.llm_model == llm_model
        assert rag.embedding_model == embedding_model
        assert rag.query_count == 0
        assert rag.total_retrieved_docs == 0
    
    def test_add_documents(self):
        """Test adding documents to RAG system."""
        llm_model = MockLLMModel()
        embedding_model = MockEmbeddingModel(dimension=32)
        vector_store = VectorStore(dimension=32)
        
        rag = RAGSystem(llm_model, embedding_model, vector_store)
        
        texts = [
            "Machine learning is a subset of artificial intelligence.",
            "Deep learning uses neural networks with multiple layers.",
            "Natural language processing handles human language data."
        ]
        metadatas = [
            {"source": "ml_basics", "topic": "ML"},
            {"source": "dl_guide", "topic": "DL"},
            {"source": "nlp_intro", "topic": "NLP"}
        ]
        
        rag.add_documents(texts, metadatas)
        
        assert vector_store.get_document_count() > 0
        assert embedding_model.call_count > 0
    
    def test_rag_query_basic(self):
        """Test basic RAG query processing."""
        llm_model = MockLLMModel()
        embedding_model = MockEmbeddingModel(dimension=32)
        vector_store = VectorStore(dimension=32)
        
        rag = RAGSystem(llm_model, embedding_model, vector_store)
        
        # Add some documents
        texts = ["Python is a programming language.", "Machine learning uses Python."]
        metadatas = [{"source": "python_doc"}, {"source": "ml_doc"}]
        rag.add_documents(texts, metadatas)
        
        # Query the system
        result = rag.query("What is Python?", top_k=2, return_sources=True)
        
        assert "answer" in result
        assert "query" in result
        assert "retrieved_documents" in result
        assert "context_length" in result
        assert "sources" in result
        
        assert result["query"] == "What is Python?"
        assert isinstance(result["answer"], str)
        assert result["retrieved_documents"] >= 0
        assert isinstance(result["sources"], list)
        
        assert rag.query_count == 1
        assert llm_model.call_count == 1
    
    def test_rag_query_without_sources(self):
        """Test RAG query without returning sources."""
        llm_model = MockLLMModel()
        embedding_model = MockEmbeddingModel(dimension=32)
        vector_store = VectorStore(dimension=32)
        
        rag = RAGSystem(llm_model, embedding_model, vector_store)
        
        # Add document
        rag.add_documents(["Test document content."], [{"source": "test"}])
        
        result = rag.query("Test query", return_sources=False)
        
        assert "sources" not in result
        assert "answer" in result
    
    def test_rag_query_empty_store(self):
        """Test RAG query with empty document store."""
        llm_model = MockLLMModel()
        embedding_model = MockEmbeddingModel(dimension=32)
        vector_store = VectorStore(dimension=32)
        
        rag = RAGSystem(llm_model, embedding_model, vector_store)
        
        result = rag.query("What is machine learning?")
        
        assert result["retrieved_documents"] == 0
        assert isinstance(result["answer"], str)
        assert llm_model.call_count == 1
    
    def test_stream_response(self):
        """Test streaming RAG responses."""
        llm_model = MockLLMModel()
        embedding_model = MockEmbeddingModel(dimension=32)
        vector_store = VectorStore(dimension=32)
        
        rag = RAGSystem(llm_model, embedding_model, vector_store)
        
        # Add document
        rag.add_documents(["Streaming test content."], [{"source": "stream_test"}])
        
        # Test streaming
        response_chunks = list(rag.stream_response("Test streaming query"))
        
        assert len(response_chunks) > 0
        assert all(isinstance(chunk, str) for chunk in response_chunks)
    
    def test_get_statistics(self):
        """Test RAG system statistics."""
        llm_model = MockLLMModel()
        embedding_model = MockEmbeddingModel(dimension=32)
        vector_store = VectorStore(dimension=32)
        
        rag = RAGSystem(llm_model, embedding_model, vector_store)
        
        # Add documents and make queries
        rag.add_documents(["Doc 1", "Doc 2"], [{"id": "1"}, {"id": "2"}])
        rag.query("Query 1")
        rag.query("Query 2")
        
        stats = rag.get_statistics()
        
        assert "total_queries" in stats
        assert "total_documents_in_store" in stats
        assert "average_documents_per_query" in stats
        assert "total_retrieved_documents" in stats
        
        assert stats["total_queries"] == 2
        assert stats["total_documents_in_store"] > 0


class TestSimpleEmbeddingModel:
    """Test SimpleEmbeddingModel functionality."""
    
    def test_simple_embedding_model(self):
        """Test simple embedding model."""
        model = SimpleEmbeddingModel(dimension=64)
        
        assert model.dimension == 64
        
        text = "Test text for embedding"
        embedding = model.encode(text)
        
        assert embedding.shape == (64,)
        assert not mx.isnan(embedding).any()
        
        # Should be normalized
        norm = mx.linalg.norm(embedding)
        assert abs(float(norm) - 1.0) < 1e-6
    
    def test_embedding_deterministic(self):
        """Test that embeddings are deterministic."""
        model = SimpleEmbeddingModel(dimension=32)
        
        text = "Deterministic test"
        embedding1 = model.encode(text)
        embedding2 = model.encode(text)
        
        assert mx.allclose(embedding1, embedding2)
    
    def test_different_texts_different_embeddings(self):
        """Test that different texts produce different embeddings."""
        model = SimpleEmbeddingModel(dimension=32)
        
        embedding1 = model.encode("First text")
        embedding2 = model.encode("Second text")
        
        # Should be different (very low probability of being identical)
        assert not mx.allclose(embedding1, embedding2, atol=1e-6)


class TestRAGEvaluation:
    """Test RAG system evaluation."""
    
    def test_evaluate_rag_system(self):
        """Test RAG system evaluation."""
        llm_model = MockLLMModel()
        embedding_model = MockEmbeddingModel(dimension=32)
        vector_store = VectorStore(dimension=32)
        
        rag = RAGSystem(llm_model, embedding_model, vector_store)
        
        # Add knowledge base
        texts = [
            "The capital of France is Paris.",
            "Python is a programming language.",
            "Machine learning is a subset of AI."
        ]
        metadatas = [{"source": f"doc{i}"} for i in range(len(texts))]
        rag.add_documents(texts, metadatas)
        
        # Test questions and answers
        test_questions = [
            "What is the capital of France?",
            "What is Python?"
        ]
        ground_truth_answers = [
            "Paris",
            "programming language"
        ]
        
        evaluation = evaluate_rag_system(rag, test_questions, ground_truth_answers)
        
        assert "accuracy" in evaluation
        assert "average_context_length" in evaluation
        assert "average_retrieved_documents" in evaluation
        assert "total_questions_evaluated" in evaluation
        
        assert 0 <= evaluation["accuracy"] <= 1
        assert evaluation["average_context_length"] >= 0
        assert evaluation["average_retrieved_documents"] >= 0
        assert evaluation["total_questions_evaluated"] == len(test_questions)
    
    def test_evaluate_empty_questions(self):
        """Test evaluation with empty question list."""
        llm_model = MockLLMModel()
        embedding_model = MockEmbeddingModel(dimension=32)
        vector_store = VectorStore(dimension=32)
        
        rag = RAGSystem(llm_model, embedding_model, vector_store)
        
        evaluation = evaluate_rag_system(rag, [], [])
        
        assert evaluation["total_questions_evaluated"] == 0
        assert evaluation["accuracy"] == 0  # 0/0 handled as 0


class TestRAGIntegration:
    """Integration tests for complete RAG pipeline."""
    
    def test_end_to_end_rag_pipeline(self):
        """Test complete end-to-end RAG pipeline."""
        llm_model = MockLLMModel()
        embedding_model = MockEmbeddingModel(dimension=64)
        vector_store = VectorStore(dimension=64)
        
        rag = RAGSystem(llm_model, embedding_model, vector_store, max_context_length=1000)
        
        # Add comprehensive knowledge base
        documents = [
            ("Machine learning is a method of data analysis that automates analytical model building.", 
             {"topic": "ML", "difficulty": "basic"}),
            ("Deep learning is part of a broader family of machine learning methods based on artificial neural networks.", 
             {"topic": "DL", "difficulty": "intermediate"}),
            ("Python is a high-level, interpreted programming language with dynamic semantics.", 
             {"topic": "Python", "difficulty": "basic"}),
            ("TensorFlow is an open-source software library for machine learning applications.", 
             {"topic": "Tools", "difficulty": "intermediate"})
        ]
        
        texts, metadatas = zip(*documents)
        rag.add_documents(list(texts), list(metadatas))
        
        # Test various queries
        queries = [
            "What is machine learning?",
            "Tell me about Python programming.",
            "How does deep learning work?",
            "What tools are available for ML?"
        ]
        
        for query in queries:
            result = rag.query(query, top_k=3, return_sources=True)
            
            # Verify response structure
            assert isinstance(result["answer"], str)
            assert len(result["answer"]) > 0
            assert result["retrieved_documents"] > 0
            assert len(result["sources"]) > 0
            assert result["context_length"] > 0
            
            # Verify sources contain relevant information
            for source in result["sources"]:
                assert "content" in source
                assert "metadata" in source
        
        # Check system statistics
        stats = rag.get_statistics()
        assert stats["total_queries"] == len(queries)
        assert stats["total_documents_in_store"] > 0
        assert stats["average_documents_per_query"] > 0
    
    def test_rag_with_document_updates(self):
        """Test RAG system with dynamic document updates."""
        llm_model = MockLLMModel()
        embedding_model = MockEmbeddingModel(dimension=32)
        vector_store = VectorStore(dimension=32)
        
        rag = RAGSystem(llm_model, embedding_model, vector_store)
        
        # Initial knowledge base
        rag.add_documents(
            ["Initial document about topic A."],
            [{"version": "1.0", "topic": "A"}]
        )
        
        # Query with initial knowledge
        result1 = rag.query("What about topic A?")
        initial_doc_count = result1["retrieved_documents"]
        
        # Add more documents
        rag.add_documents(
            ["Additional information about topic A.", "New document about topic B."],
            [{"version": "1.1", "topic": "A"}, {"version": "1.0", "topic": "B"}]
        )
        
        # Query again - should retrieve from expanded knowledge base
        result2 = rag.query("What about topic A?")
        
        assert vector_store.get_document_count() > initial_doc_count
        assert result2["retrieved_documents"] >= result1["retrieved_documents"]
    
    def test_rag_performance_characteristics(self):
        """Test RAG system performance characteristics."""
        llm_model = MockLLMModel()
        embedding_model = MockEmbeddingModel(dimension=64)
        vector_store = VectorStore(dimension=64)
        
        rag = RAGSystem(llm_model, embedding_model, vector_store)
        
        # Add documents of varying sizes
        large_documents = []
        for i in range(10):
            content = f"Document {i}. " + "Content. " * 100  # ~800 characters each
            large_documents.append((content, {"doc_id": f"large_{i}", "size": "large"}))
        
        texts, metadatas = zip(*large_documents)
        
        # Time document addition
        start_time = time.time()
        rag.add_documents(list(texts), list(metadatas))
        indexing_time = time.time() - start_time
        
        # Time query processing
        start_time = time.time()
        result = rag.query("Tell me about the documents", top_k=5)
        query_time = time.time() - start_time
        
        # Performance should be reasonable
        assert indexing_time < 30.0  # Should index within 30 seconds
        assert query_time < 10.0     # Should query within 10 seconds
        
        # Should retrieve documents
        assert result["retrieved_documents"] > 0
        assert len(result["sources"]) > 0
    
    def test_rag_context_length_management(self):
        """Test RAG system handles context length limits."""
        llm_model = MockLLMModel()
        embedding_model = MockEmbeddingModel(dimension=32)
        vector_store = VectorStore(dimension=32)
        
        # Small context limit for testing
        rag = RAGSystem(llm_model, embedding_model, vector_store, max_context_length=200)
        
        # Add documents that would exceed context limit
        long_documents = []
        for i in range(5):
            content = f"Very long document {i}. " + "More content here. " * 20
            long_documents.append((content, {"doc_id": f"long_{i}"}))
        
        texts, metadatas = zip(*long_documents)
        rag.add_documents(list(texts), list(metadatas))
        
        # Query should respect context length limit
        result = rag.query("What are these documents about?", top_k=10)
        
        # Context should be truncated to fit limit
        assert result["context_length"] <= 200
        assert isinstance(result["answer"], str)


if __name__ == "__main__":
    pytest.main([__file__])
