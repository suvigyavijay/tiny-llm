from typing import List, Dict

class KnowledgeBase:
    def __init__(self):
        self.documents: Dict[int, str] = {}
        self.next_doc_id = 0

    def add_document(self, text: str):
        self.documents[self.next_doc_id] = text
        self.next_doc_id += 1

    def retrieve(self, query: str, top_k: int = 1) -> List[str]:
        # This is a simplified retriever that uses keyword matching.
        # A real implementation would use a more sophisticated retrieval method,
        # such as TF-IDF or a vector-based search.
        stop_words = {'a', 'an', 'the', 'is', 'in', 'it', 'of', 'for', 'what', 'and', 'to'}
        query_words = {word.lower().strip("?,.") for word in query.split()}
        search_words = query_words - stop_words

        if not search_words:
            return []

        # Score documents based on the number of matching keywords
        scored_docs = []
        for doc_id, doc_text in self.documents.items():
            # A simple way to find words in a doc
            doc_words = set(doc_text.lower().split())
            matched_words = search_words.intersection(doc_words)
            if matched_words:
                scored_docs.append((len(matched_words), doc_text))

        # Sort by score (descending)
        scored_docs.sort(key=lambda x: x[0], reverse=True)

        # Return the text of the top_k documents
        return [doc for score, doc in scored_docs[:top_k]]

class RagPipeline:
    def __init__(self, model, knowledge_base: KnowledgeBase):
        self.model = model
        self.knowledge_base = knowledge_base

    def generate(self, query: str) -> str:
        retrieved_docs = self.knowledge_base.retrieve(query, top_k=3)
        
        prompt = "Context:\n"
        if retrieved_docs:
            for doc in retrieved_docs:
                prompt += f"- {doc}\n"
        
        prompt += f"\nQuestion: {query}\nAnswer:"
        
        return self.model.generate(prompt)
