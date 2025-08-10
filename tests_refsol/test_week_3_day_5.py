import pytest
from tiny_llm_ref.rag import KnowledgeBase, RagPipeline

class MockModel:
    def generate(self, prompt):
        return f"Generated response based on: {prompt}"

class TestRagPipeline:
    @pytest.fixture
    def setup_kb(self):
        kb = KnowledgeBase()
        kb.add_document("The sky is blue.")
        kb.add_document("The grass is green.")
        kb.add_document("The sun is yellow.")
        return kb

    def test_rag_pipeline_basic(self, setup_kb):
        model = MockModel()
        rag = RagPipeline(model, setup_kb)
        
        response = rag.generate("What color is the sky?")
        
        assert "The sky is blue" in response
        assert "What color is the sky?" in response

    def test_rag_pipeline_no_relevant_docs(self, setup_kb):
        model = MockModel()
        rag = RagPipeline(model, setup_kb)
        
        response = rag.generate("What is the meaning of life?")
        
        assert "Context:\n\nQuestion: What is the meaning of life?\nAnswer:" in response

    def test_rag_pipeline_multiple_docs(self, setup_kb):
        model = MockModel()
        rag = RagPipeline(model, setup_kb)
        
        response = rag.generate("What color is the grass and sun?")
        
        assert "The grass is green" in response
        assert "The sun is yellow" in response
        assert "What color is the grass and sun?" in response

    def test_empty_knowledge_base(self):
        kb = KnowledgeBase()
        model = MockModel()
        rag = RagPipeline(model, kb)
        
        response = rag.generate("What color is the sky?")
        
        assert "Context:\n\nQuestion: What color is the sky?\nAnswer:" in response
