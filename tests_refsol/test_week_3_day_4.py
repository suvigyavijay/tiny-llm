import pytest
import mlx.core as mx
from tiny_llm_ref.speculative import speculative_decode

class MockModel:
    def __init__(self, generated_tokens, token_probs):
        self.generated_tokens = generated_tokens
        self.token_probs = token_probs

    def generate(self, prompt, num_tokens):
        return self.generated_tokens

    def get_probabilities(self, prompt, tokens):
        return self.token_probs

class TestSpeculativeDecoding:
    def test_speculative_decode_accept_all(self):
        draft_tokens = mx.array([1, 2, 3, 4])
        target_probs = mx.array([[0.1, 0.9, 0.0, 0.0, 0.0],
                                 [0.0, 0.1, 0.9, 0.0, 0.0],
                                 [0.0, 0.0, 0.1, 0.9, 0.0],
                                 [0.0, 0.0, 0.0, 0.1, 0.9]])
        
        draft_model = MockModel(draft_tokens, None)
        target_model = MockModel(None, target_probs)
        
        accepted_tokens = speculative_decode(draft_model, target_model, "prompt")
        
        assert mx.array_equal(mx.array(accepted_tokens), draft_tokens)

    def test_speculative_decode_reject_some(self):
        draft_tokens = mx.array([1, 2, 3, 4])
        target_probs = mx.array([[0.1, 0.9, 0.0, 0.0, 0.0],
                                 [0.9, 0.1, 0.0, 0.0, 0.0],
                                 [0.0, 0.0, 0.1, 0.9, 0.0],
                                 [0.0, 0.0, 0.0, 0.1, 0.9]])
        
        draft_model = MockModel(draft_tokens, None)
        target_model = MockModel(None, target_probs)
        
        accepted_tokens = speculative_decode(draft_model, target_model, "prompt")
        
        assert mx.array_equal(mx.array(accepted_tokens), mx.array([1]))
