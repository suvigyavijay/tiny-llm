import pytest
import mlx.core as mx
from tiny_llm_ref.speculative import speculative_decode


class MockModel:
    """A mock model that returns predetermined tokens."""
    def __init__(self, token_sequence: list):
        self.token_sequence = token_sequence
        self.call_count = 0
    
    def __call__(self, x: mx.array) -> mx.array:
        # Return logits that argmax to the desired token
        l = x.shape[1]
        target = self.token_sequence[min(l - 1, len(self.token_sequence) - 1)]
        logits = mx.zeros((1, l, 100))
        # Set high logit for target token at last position
        for i in range(l):
            t = self.token_sequence[min(i, len(self.token_sequence) - 1)]
            logits = logits.at[0, i, t].set(10.0)
        return logits


def test_speculative_all_accepted():
    """Test when all draft tokens are accepted."""
    # Both models predict: 10, 11, 12, then target predicts 13
    target = MockModel([10, 11, 12, 13])
    draft = MockModel([10, 11, 12])
    
    prompt = mx.array([[1]])  # Single token prompt
    accepted, correction = speculative_decode(target, draft, prompt, k=3)
    
    mx.eval(accepted)
    mx.eval(correction)
    
    assert len(accepted) == 3
    assert accepted[0].item() == 10
    assert accepted[1].item() == 11
    assert accepted[2].item() == 12
    assert correction.item() == 13


def test_speculative_rejection():
    """Test when draft token is rejected."""
    # Draft predicts 10, 11, 99 (wrong)
    # Target predicts 10, 11, 12
    target = MockModel([10, 11, 12])
    draft = MockModel([10, 11, 99])
    
    prompt = mx.array([[1]])
    accepted, correction = speculative_decode(target, draft, prompt, k=3)
    
    mx.eval(accepted)
    mx.eval(correction)
    
    # First 2 should be accepted, third rejected and corrected to 12
    assert len(accepted) == 2
    assert accepted[0].item() == 10
    assert accepted[1].item() == 11
    assert correction.item() == 12
