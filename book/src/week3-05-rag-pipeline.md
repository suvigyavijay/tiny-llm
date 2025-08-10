# Week 3 Day 5: RAG Pipeline

Retrieval-Augmented Generation (RAG) is a powerful technique that enhances the capabilities of LLMs by allowing them to access external knowledge sources. This enables them to provide more accurate and up-to-date information, overcoming the limitations of their static training data.

In this chapter, you will implement a simple RAG pipeline. This will give you a practical understanding of how RAG works and how it can be used to build more powerful and knowledgeable language models.

[📚 Reading: Retrieval-Augmented Generation (RAG)](https://aws.amazon.com/what-is/retrieval-augmented-generation/)

## Task 1: Implement the RAG Pipeline

Your task is to implement a RAG pipeline that consists of a knowledge base, a retriever, and a generator.

```
src/tiny_llm/rag.py
```

The implementation will involve the following components:
- **`KnowledgeBase`**: A simple class that stores a collection of documents.
- **`Retriever`**: A component that retrieves relevant documents from the knowledge base based on a query. In this simplified implementation, the retriever will be part of the `KnowledgeBase` class.
- **`RagPipeline`**: The main class that orchestrates the RAG process. It will take a query, retrieve relevant documents, and then use the generator (an LLM) to produce a response based on the query and the retrieved documents.

You can run the following tests to verify your implementation:

```
pdm run test --week 3 --day 5
```

This simplified RAG pipeline will serve as a starting point for building more sophisticated systems. In a real-world application, you would use more advanced techniques for retrieval, such as vector search, and you would likely use a much larger and more diverse knowledge base.

{{#include copyright.md}}
