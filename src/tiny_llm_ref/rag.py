from typing import List, Dict

class KnowledgeBase:
    def __init__(self):
        self.documents: Dict[int, str] = {}
        self.next_doc_id = 0

    def add_document(self, text: str):
        self.documents[self.next_doc_id] = text
        self.next_doc_id += 1

    def retrieve(self, query: str, top_k: int = 1) -> List[str]:
        # This is a simplified retriever that just returns the first k documents.
        # A real implementation would use a more sophisticated retrieval method,
        # such as TF-IDF or a vector-based search.
        return list(self.documents.values())[:top_k]

class RagPipeline:
    def __init__(self, model, knowledge_base: KnowledgeBase):
        self.model = model
        self.knowledge_base = knowledge_base

    def generate(self, query: str) -> str:
        retrieved_docs = self.knowledge_base.retrieve(query)
        
        prompt = "Context:\n"
        for doc in retrieved_docs:
            prompt += f"- {doc}\n"
        prompt += f"\nQuestion: {query}\nAnswer:"
        
        return self.model.generate(prompt)
