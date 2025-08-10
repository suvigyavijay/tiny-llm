import pytest
from tiny_llm_ref.rag import KnowledgeBase, RagPipeline

class MockModel:
    def generate(self, prompt):
        return f"Generated response based on: {prompt}"

class TestRagPipeline:
    def test_rag_pipeline(self):
        kb = KnowledgeBase()
        kb.add_document("The sky is blue.")
        kb.add_document("The grass is green.")
        
        model = MockModel()
        rag = RagPipeline(model, kb)
        
        response = rag.generate("What color is the sky?")
        
        assert "The sky is blue" in response
        assert "What color is the sky?" in response

    def test_empty_knowledge_base(self):
        kb = KnowledgeBase()
        model = MockModel()
        rag = RagPipeline(model, kb)
        
        response = rag.generate("What color is the sky?")
        
        assert "Context:\n\nQuestion: What color is the sky?\nAnswer:" in response
