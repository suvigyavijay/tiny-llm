"""
Retrieval-Augmented Generation (RAG) Pipeline implementation.

Implements a complete RAG system combining document retrieval with LLM generation
to enable knowledge-enhanced responses without model retraining.
"""

import mlx.core as mx
from typing import List, Dict, Optional, Any, Tuple
import math
import re
import json
from dataclasses import dataclass


@dataclass
class Document:
    """Represents a document chunk in the knowledge base."""
    id: str
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[mx.array] = None


class DocumentProcessor:
    """Processes and chunks documents for RAG knowledge base."""
    
    def __init__(self, embedding_model: Any, chunk_size: int = 512, overlap: int = 50):
        """
        Initialize document processor.
        
        Args:
            embedding_model: Model for generating embeddings
            chunk_size: Size of text chunks in characters
            overlap: Overlap between chunks in characters
        """
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: Input text to chunk
            
        Returns:
            List of text chunks
        """
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            # Try to break at sentence boundaries
            if end < len(text):
                # Look for sentence endings
                sentence_end = text.rfind('.', start, end)
                if sentence_end > start + self.chunk_size // 2:
                    end = sentence_end + 1
                else:
                    # Fall back to word boundaries
                    word_end = text.rfind(' ', start, end)
                    if word_end > start + self.chunk_size // 2:
                        end = word_end
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Move start position with overlap
            start = max(start + 1, end - self.overlap)
            
            if end >= len(text):
                break
        
        return chunks
    
    def process_document(self, text: str, metadata: Dict[str, Any]) -> List[Document]:
        """
        Process document into embedded chunks.
        
        Args:
            text: Document text content
            metadata: Document metadata (title, source, etc.)
            
        Returns:
            List of Document objects with embeddings
        """
        chunks = self.chunk_text(text)
        documents = []
        
        for i, chunk in enumerate(chunks):
            # Generate embedding for chunk
            embedding = self.embedding_model.encode(chunk)
            
            # Create document with metadata
            doc_metadata = metadata.copy()
            doc_metadata.update({
                'chunk_id': i,
                'chunk_count': len(chunks),
                'chunk_text_length': len(chunk)
            })
            
            doc = Document(
                id=f"{metadata.get('doc_id', 'unknown')}_{i}",
                content=chunk,
                metadata=doc_metadata,
                embedding=embedding
            )
            documents.append(doc)
        
        return documents


class VectorStore:
    """In-memory vector store for document embeddings and similarity search."""
    
    def __init__(self, dimension: int):
        """
        Initialize vector store.
        
        Args:
            dimension: Embedding dimension
        """
        self.dimension = dimension
        self.documents: List[Document] = []
        self.embeddings: Optional[mx.array] = None
        
    def add_documents(self, documents: List[Document]):
        """
        Add documents to the vector store.
        
        Args:
            documents: List of documents to add
        """
        self.documents.extend(documents)
        
        # Rebuild embedding matrix
        embeddings_list = [doc.embedding for doc in self.documents if doc.embedding is not None]
        if embeddings_list:
            self.embeddings = mx.stack(embeddings_list)
        
    def similarity_search(self, query_embedding: mx.array, top_k: int = 5) -> List[Tuple[Document, float]]:
        """
        Find most similar documents using cosine similarity.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of top results to return
            
        Returns:
            List of (document, similarity_score) tuples
        """
        if self.embeddings is None or len(self.documents) == 0:
            return []
        
        # Normalize embeddings
        query_norm = query_embedding / mx.linalg.norm(query_embedding)
        doc_norms = self.embeddings / mx.linalg.norm(self.embeddings, axis=1, keepdims=True)
        
        # Compute cosine similarities
        similarities = mx.matmul(doc_norms, query_norm)
        
        # Get top-k indices
        top_indices = mx.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            doc = self.documents[int(idx)]
            score = float(similarities[int(idx)])
            results.append((doc, score))
        
        return results
    
    def get_document_count(self) -> int:
        """Get total number of documents in store."""
        return len(self.documents)
    
    def clear(self):
        """Clear all documents from the store."""
        self.documents = []
        self.embeddings = None


class HybridRetriever:
    """Retriever combining dense and sparse retrieval methods."""
    
    def __init__(self, vector_store: VectorStore, enable_reranking: bool = False):
        """
        Initialize hybrid retriever.
        
        Args:
            vector_store: Vector store for dense retrieval
            enable_reranking: Whether to apply reranking
        """
        self.vector_store = vector_store
        self.enable_reranking = enable_reranking
        
        # Simple BM25-like scoring for sparse retrieval
        self.term_frequencies = {}
        self.document_frequencies = {}
        self.total_documents = 0
        
    def _build_sparse_index(self):
        """Build sparse retrieval index (simplified BM25)."""
        self.term_frequencies = {}
        self.document_frequencies = {}
        self.total_documents = len(self.vector_store.documents)
        
        for doc_idx, doc in enumerate(self.vector_store.documents):
            # Simple tokenization
            tokens = re.findall(r'\b\w+\b', doc.content.lower())
            
            # Track term frequencies
            doc_term_freq = {}
            for token in tokens:
                doc_term_freq[token] = doc_term_freq.get(token, 0) + 1
            
            self.term_frequencies[doc_idx] = doc_term_freq
            
            # Track document frequencies
            for token in set(tokens):
                self.document_frequencies[token] = self.document_frequencies.get(token, 0) + 1
    
    def _sparse_retrieval(self, query: str, top_k: int = 20) -> List[Tuple[int, float]]:
        """
        Perform sparse retrieval using BM25-like scoring.
        
        Args:
            query: Query string
            top_k: Number of results to return
            
        Returns:
            List of (document_index, score) tuples
        """
        if not self.term_frequencies:
            self._build_sparse_index()
        
        query_tokens = re.findall(r'\b\w+\b', query.lower())
        scores = {}
        
        k1, b = 1.5, 0.75  # BM25 parameters
        avgdl = sum(len(tf) for tf in self.term_frequencies.values()) / max(1, len(self.term_frequencies))
        
        for doc_idx, doc_tf in self.term_frequencies.items():
            score = 0.0
            doc_len = sum(doc_tf.values())
            
            for token in query_tokens:
                if token in doc_tf and token in self.document_frequencies:
                    tf = doc_tf[token]
                    df = self.document_frequencies[token]
                    idf = math.log((self.total_documents - df + 0.5) / (df + 0.5))
                    
                    numerator = tf * (k1 + 1)
                    denominator = tf + k1 * (1 - b + b * (doc_len / avgdl))
                    score += idf * (numerator / denominator)
            
            if score > 0:
                scores[doc_idx] = score
        
        # Sort by score and return top-k
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_scores[:top_k]
    
    def retrieve(self, query: str, query_embedding: mx.array, top_k: int = 10) -> List[Document]:
        """
        Retrieve documents using hybrid approach.
        
        Args:
            query: Query string
            query_embedding: Query embedding
            top_k: Number of documents to retrieve
            
        Returns:
            List of retrieved documents
        """
        # Dense retrieval
        dense_results = self.vector_store.similarity_search(query_embedding, top_k * 2)
        
        # Sparse retrieval
        sparse_results = self._sparse_retrieval(query, top_k * 2)
        
        # Combine results (simple score fusion)
        combined_scores = {}
        
        # Add dense scores
        for doc, score in dense_results:
            doc_idx = self.vector_store.documents.index(doc)
            combined_scores[doc_idx] = score * 0.6  # Weight dense retrieval
        
        # Add sparse scores
        for doc_idx, score in sparse_results:
            if doc_idx in combined_scores:
                combined_scores[doc_idx] += score * 0.4  # Weight sparse retrieval
            else:
                combined_scores[doc_idx] = score * 0.4
        
        # Sort and return top-k documents
        sorted_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        
        retrieved_docs = []
        for doc_idx, score in sorted_results[:top_k]:
            retrieved_docs.append(self.vector_store.documents[doc_idx])
        
        return retrieved_docs


class ContextAssembler:
    """Assembles retrieved documents into context for LLM prompts."""
    
    def __init__(self, max_context_length: int = 4096):
        """
        Initialize context assembler.
        
        Args:
            max_context_length: Maximum context length in characters
        """
        self.max_context_length = max_context_length
    
    def assemble_context(self, query: str, documents: List[Document]) -> str:
        """
        Create context from retrieved documents.
        
        Args:
            query: User query
            documents: Retrieved documents
            
        Returns:
            Assembled context string
        """
        context_parts = []
        current_length = 0
        
        for i, doc in enumerate(documents):
            # Format document
            doc_text = f"Document {i + 1}:\n{doc.content}\n\n"
            
            # Check if adding this document would exceed limit
            if current_length + len(doc_text) > self.max_context_length:
                break
            
            context_parts.append(doc_text)
            current_length += len(doc_text)
        
        return "".join(context_parts)
    
    def create_rag_prompt(self, query: str, context: str) -> str:
        """
        Create final RAG prompt with instructions.
        
        Args:
            query: User query
            context: Assembled context
            
        Returns:
            Complete prompt for LLM
        """
        prompt = f"""Use the following context to answer the question. If the answer cannot be found in the context, say "I cannot find the answer in the provided context."

Context:
{context}

Question: {query}

Answer:"""
        
        return prompt


class RAGSystem:
    """Complete RAG system integrating all components."""
    
    def __init__(self, llm_model: Any, embedding_model: Any, 
                 vector_store: VectorStore, max_context_length: int = 4096):
        """
        Initialize RAG system.
        
        Args:
            llm_model: Language model for generation
            embedding_model: Model for generating embeddings
            vector_store: Vector store for document retrieval
            max_context_length: Maximum context length
        """
        self.llm_model = llm_model
        self.embedding_model = embedding_model
        self.retriever = HybridRetriever(vector_store)
        self.context_assembler = ContextAssembler(max_context_length)
        
        # Track query statistics
        self.query_count = 0
        self.total_retrieved_docs = 0
        
    def add_documents(self, texts: List[str], metadatas: List[Dict[str, Any]]):
        """
        Add documents to the RAG knowledge base.
        
        Args:
            texts: List of document texts
            metadatas: List of metadata dictionaries
        """
        processor = DocumentProcessor(self.embedding_model)
        
        all_docs = []
        for text, metadata in zip(texts, metadatas):
            docs = processor.process_document(text, metadata)
            all_docs.extend(docs)
        
        self.retriever.vector_store.add_documents(all_docs)
    
    def query(self, user_query: str, top_k: int = 5, return_sources: bool = True) -> Dict[str, Any]:
        """
        Process query through RAG pipeline.
        
        Args:
            user_query: User's question
            top_k: Number of documents to retrieve
            return_sources: Whether to return source documents
            
        Returns:
            Dictionary with answer and metadata
        """
        self.query_count += 1
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode(user_query)
        
        # Retrieve relevant documents
        retrieved_docs = self.retriever.retrieve(user_query, query_embedding, top_k)
        self.total_retrieved_docs += len(retrieved_docs)
        
        # Assemble context
        context = self.context_assembler.assemble_context(user_query, retrieved_docs)
        
        # Create RAG prompt
        prompt = self.context_assembler.create_rag_prompt(user_query, context)
        
        # Generate response
        response = self.llm_model.generate(prompt)
        
        result = {
            "answer": response,
            "query": user_query,
            "retrieved_documents": len(retrieved_docs),
            "context_length": len(context)
        }
        
        if return_sources:
            result["sources"] = [
                {
                    "content": doc.content[:200] + "..." if len(doc.content) > 200 else doc.content,
                    "metadata": doc.metadata
                }
                for doc in retrieved_docs
            ]
        
        return result
    
    def stream_response(self, user_query: str, top_k: int = 5):
        """
        Stream RAG response for better user experience.
        
        Args:
            user_query: User's question
            top_k: Number of documents to retrieve
            
        Yields:
            Response chunks as they're generated
        """
        # Generate query embedding
        query_embedding = self.embedding_model.encode(user_query)
        
        # Retrieve relevant documents
        retrieved_docs = self.retriever.retrieve(user_query, query_embedding, top_k)
        
        # Assemble context
        context = self.context_assembler.assemble_context(user_query, retrieved_docs)
        
        # Create RAG prompt
        prompt = self.context_assembler.create_rag_prompt(user_query, context)
        
        # Stream response from LLM
        for chunk in self.llm_model.stream_generate(prompt):
            yield chunk
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get RAG system usage statistics."""
        avg_docs_per_query = self.total_retrieved_docs / max(1, self.query_count)
        
        return {
            "total_queries": self.query_count,
            "total_documents_in_store": self.retriever.vector_store.get_document_count(),
            "average_documents_per_query": avg_docs_per_query,
            "total_retrieved_documents": self.total_retrieved_docs
        }


class SimpleEmbeddingModel:
    """Simple embedding model for testing (normally would use a real model)."""
    
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
    
    def encode(self, text: str) -> mx.array:
        """Create simple hash-based embedding for testing."""
        # Simple deterministic embedding based on text hash
        hash_val = hash(text)
        
        # Create pseudo-random embedding
        mx.random.seed(abs(hash_val) % (2**31))
        embedding = mx.random.normal((self.dimension,))
        
        # Normalize
        return embedding / mx.linalg.norm(embedding)


def evaluate_rag_system(rag_system: RAGSystem, 
                       test_questions: List[str],
                       ground_truth_answers: List[str]) -> Dict[str, float]:
    """
    Evaluate RAG system performance.
    
    Args:
        rag_system: RAG system to evaluate
        test_questions: List of test questions
        ground_truth_answers: List of expected answers
        
    Returns:
        Evaluation metrics
    """
    correct_answers = 0
    total_context_length = 0
    total_retrieved_docs = 0
    
    for question, expected_answer in zip(test_questions, ground_truth_answers):
        result = rag_system.query(question)
        
        # Simple exact match evaluation (in practice, would use more sophisticated metrics)
        if expected_answer.lower() in result["answer"].lower():
            correct_answers += 1
        
        total_context_length += result["context_length"]
        total_retrieved_docs += result["retrieved_documents"]
    
    num_questions = len(test_questions)
    
    return {
        "accuracy": correct_answers / num_questions,
        "average_context_length": total_context_length / num_questions,
        "average_retrieved_documents": total_retrieved_docs / num_questions,
        "total_questions_evaluated": num_questions
    }
