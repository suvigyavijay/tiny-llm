# Week 3 Day 5: Retrieval Augmented Generation (RAG)

In this chapter, we will build a **Retrieval Augmented Generation (RAG)** pipeline. We will implement a simple vector store to retrieve relevant context and inject it into the model's prompt to reduce hallucinations.

LLMs hallucinate and lack up-to-date knowledge. RAG solves this by retrieving relevant documents from a knowledge base and injecting them into the prompt.

**📚 Readings**

- [Retrieval Augmented Generation (RAG) Explained - Pinecone](https://www.pinecone.io/learn/retrieval-augmented-generation/)
- [Building RAG with LangChain - LangChain Docs](https://python.langchain.com/docs/use_cases/question_answering/)

## Concept: Prompt Injection

Instead of:
`User: Who won the World Cup in 2030?`
`Model: I don't know.`

We do:
`System: Use this context: "Brazil won the 2030 World Cup."`
`User: Who won the World Cup in 2030?`
`Model: Brazil.`

To find the right context, we use **Vector Embeddings**.

## Pipeline

1.  **Ingestion**: Text -> Chunks -> Embeddings -> Vector DB.
2.  **Retrieval**: Query -> Embedding -> ANN Search -> Top-K Chunks.
3.  **Generation**: System Prompt + Retrieved Context + Query -> LLM Answer.

## Task 1: Simple Vector Store

```
src/tiny_llm/rag.py
```

Implement a simple in-memory `VectorStore`.
- `add(text, embedding)`
- `search(query_embedding, k)`: Uses cosine similarity.

### Code Walkthrough

```python
class VectorStore:
    def search(self, query, k):
        # 1. Normalize query
        # 2. Compute dot product with all stored embeddings
        # 3. Return top K indices
        pass
```

## Testing

```bash
pdm run test-refsol tests_refsol/test_week_3_day_5.py
```

{{#include copyright.md}}
