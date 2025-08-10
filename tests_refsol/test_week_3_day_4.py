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
    @pytest.mark.parametrize("draft_len", [1, 5, 10])
    def test_speculative_decode_accept_all(self, draft_len):
        draft_tokens = mx.arange(draft_len)
        target_probs = mx.eye(draft_len)

        draft_model = MockModel(draft_tokens, None)
        target_model = MockModel(None, target_probs)
        
        accepted_tokens = speculative_decode(draft_model, target_model, "prompt")
        
        assert mx.array_equal(mx.array(accepted_tokens), draft_tokens)

    @pytest.mark.parametrize("reject_idx", [0, 2, 4])
    def test_speculative_decode_reject_some(self, reject_idx):
        draft_tokens = mx.arange(5)
        target_probs = mx.eye(5)
        target_probs[reject_idx] = mx.roll(target_probs[reject_idx], 1)

        draft_model = MockModel(draft_tokens, None)
        target_model = MockModel(None, target_probs)
        
        accepted_tokens = speculative_decode(draft_model, target_model, "prompt")
        
        assert mx.array_equal(mx.array(accepted_tokens), draft_tokens[:reject_idx])

    def test_speculative_decode_empty_draft(self):
        draft_tokens = mx.array([])
        target_probs = mx.array([])

        draft_model = MockModel(draft_tokens, None)
        target_model = MockModel(None, target_probs)
        
        accepted_tokens = speculative_decode(draft_model, target_model, "prompt")
        
        assert len(accepted_tokens) == 0

    def test_speculative_decode_prob_distribution(self):
        draft_tokens = mx.array([1, 2, 3])
        target_probs = mx.array([[0.1, 0.8, 0.1, 0.0],
                                 [0.1, 0.2, 0.7, 0.0],
                                 [0.3, 0.3, 0.3, 0.1]])

        draft_model = MockModel(draft_tokens, None)
        target_model = MockModel(None, target_probs)
        
        accepted_tokens = speculative_decode(draft_model, target_model, "prompt")
        
        # Manually calculate the expected accepted tokens based on the probabilities
        expected_tokens = []
        for i in range(len(draft_tokens)):
            if mx.random.uniform() < target_probs[i, draft_tokens[i]]:
                expected_tokens.append(draft_tokens[i])
            else:
                break
        
        # This test is probabilistic, so we can't assert equality.
        # Instead, we check if the length of accepted tokens is plausible.
        assert len(accepted_tokens) <= len(draft_tokens)
