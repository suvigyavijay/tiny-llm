import pytest
import mlx.core as mx
from .tiny_llm_base import *
from .utils import *


def test_speculative_decode_accepts_correct():
    """Test speculative decode accepts when draft matches target."""
    
    # Mock models that always predict the same token
    def mock_target(tokens):
        # Return logits that favor token 5
        logits = mx.full((1, 10), -10.0)
        logits = logits.at[:, 5].add(20.0)
        return logits
    
    def mock_draft(tokens):
        # Same as target
        return mock_target(tokens)
    
    prompt = mx.array([[1, 2, 3]])
    accepted, correction = speculative_decode(mock_target, mock_draft, prompt, k=3)
    
    # All drafts should be accepted when draft == target
    assert len(accepted) == 3


def test_speculative_decode_rejects_wrong():
    """Test speculative decode rejects when draft differs from target."""
    
    def mock_target(tokens):
        logits = mx.full((1, 10), -10.0)
        logits = logits.at[:, 5].add(20.0)  # Target wants 5
        return logits
    
    def mock_draft(tokens):
        logits = mx.full((1, 10), -10.0)
        logits = logits.at[:, 7].add(20.0)  # Draft predicts 7
        return logits
    
    prompt = mx.array([[1, 2, 3]])
    accepted, correction = speculative_decode(mock_target, mock_draft, prompt, k=3)
    
    # Draft should be rejected
    assert len(accepted) == 0
    assert correction is not None
