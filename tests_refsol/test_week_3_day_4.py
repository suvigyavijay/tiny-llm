"""
Tests for Week 3, Day 4: Speculative Decoding implementation.

Tests the speculative decoding system with draft model generation,
verification, and acceptance/rejection mechanisms.
"""

import pytest
import mlx.core as mx
import time
from unittest.mock import Mock, MagicMock
from src.tiny_llm_ref.speculative_decoding import (
    SpeculativeDecoder, AdaptiveSpeculator, batched_speculative_decode,
    benchmark_speculative_decoding
)


class MockModel:
    """Mock model for testing speculative decoding."""
    
    def __init__(self, vocab_size=1000, hidden_dim=64, is_draft=False):
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.is_draft = is_draft
        self.call_count = 0
    
    def __call__(self, tokens):
        """Generate mock logits."""
        self.call_count += 1
        batch_size, seq_len = tokens.shape
        
        # Create mock logits with some patterns
        if self.is_draft:
            # Draft model: simpler, more uniform distribution
            logits = mx.random.normal((batch_size, seq_len, self.vocab_size)) * 0.5
        else:
            # Target model: more peaked distribution
            logits = mx.random.normal((batch_size, seq_len, self.vocab_size)) * 1.0
        
        # Add some deterministic patterns based on input tokens
        for i in range(batch_size):
            for j in range(seq_len):
                if j > 0:
                    prev_token = int(tokens[i, j-1])
                    # Make next token somewhat predictable
                    next_token = (prev_token + 1) % self.vocab_size
                    logits = logits.at[i, j, next_token].add(2.0)
        
        return logits
    
    def reset_call_count(self):
        self.call_count = 0


class TestSpeculativeDecoder:
    """Test SpeculativeDecoder functionality."""
    
    def test_decoder_initialization(self):
        """Test speculative decoder initialization."""
        draft_model = MockModel(is_draft=True)
        target_model = MockModel(is_draft=False)
        
        decoder = SpeculativeDecoder(
            draft_model=draft_model,
            target_model=target_model,
            lookahead=4,
            acceptance_threshold=0.8
        )
        
        assert decoder.draft_model == draft_model
        assert decoder.target_model == target_model
        assert decoder.lookahead == 4
        assert decoder.acceptance_threshold == 0.8
        assert decoder.total_draft_tokens == 0
        assert decoder.total_accepted_tokens == 0
    
    def test_draft_generation(self):
        """Test draft token generation."""
        draft_model = MockModel(vocab_size=100, is_draft=True)
        target_model = MockModel(vocab_size=100, is_draft=False)
        
        decoder = SpeculativeDecoder(draft_model, target_model, lookahead=3)
        
        prompt_tokens = mx.array([[10, 20, 30]])
        draft_tokens = decoder.draft_generation(prompt_tokens, num_tokens=3, temperature=1.0)
        
        assert len(draft_tokens) == 3
        assert all(isinstance(token, int) for token in draft_tokens)
        assert all(0 <= token < 100 for token in draft_tokens)
        assert decoder.total_draft_tokens == 3
    
    def test_draft_generation_greedy(self):
        """Test greedy draft generation (temperature=0)."""
        draft_model = MockModel(vocab_size=100, is_draft=True)
        target_model = MockModel(vocab_size=100, is_draft=False)
        
        decoder = SpeculativeDecoder(draft_model, target_model)
        
        prompt_tokens = mx.array([[10, 20]])
        
        # Generate twice with temperature=0 (should be deterministic)
        draft_tokens1 = decoder.draft_generation(prompt_tokens, num_tokens=2, temperature=0.0)
        draft_tokens2 = decoder.draft_generation(prompt_tokens, num_tokens=2, temperature=0.0)
        
        assert draft_tokens1 == draft_tokens2  # Should be identical
    
    def test_verify_speculation(self):
        """Test speculation verification."""
        draft_model = MockModel(vocab_size=50, is_draft=True)
        target_model = MockModel(vocab_size=50, is_draft=False)
        
        decoder = SpeculativeDecoder(draft_model, target_model)
        
        base_tokens = mx.array([[1, 2, 3]])
        draft_tokens = [4, 5, 6]
        
        num_accepted, corrected_tokens = decoder.verify_speculation(
            base_tokens, draft_tokens, temperature=1.0
        )
        
        assert 0 <= num_accepted <= len(draft_tokens)
        assert len(corrected_tokens) <= 1  # At most one correction
        assert decoder.total_accepted_tokens >= 0
        assert decoder.verification_calls == 1
    
    def test_verify_speculation_empty_draft(self):
        """Test verification with empty draft tokens."""
        draft_model = MockModel(vocab_size=50, is_draft=True)
        target_model = MockModel(vocab_size=50, is_draft=False)
        
        decoder = SpeculativeDecoder(draft_model, target_model)
        
        base_tokens = mx.array([[1, 2, 3]])
        draft_tokens = []
        
        num_accepted, corrected_tokens = decoder.verify_speculation(
            base_tokens, draft_tokens, temperature=1.0
        )
        
        assert num_accepted == 0
        assert len(corrected_tokens) == 0
    
    def test_generate_basic(self):
        """Test basic token generation."""
        draft_model = MockModel(vocab_size=50, is_draft=True)
        target_model = MockModel(vocab_size=50, is_draft=False)
        
        decoder = SpeculativeDecoder(draft_model, target_model, lookahead=2)
        
        prompt_tokens = mx.array([[10, 20]])
        generated_tokens = decoder.generate(prompt_tokens, max_tokens=5, temperature=1.0)
        
        assert len(generated_tokens) <= 5
        assert all(isinstance(token, int) for token in generated_tokens)
        assert all(0 <= token < 50 for token in generated_tokens)
    
    def test_generate_max_tokens(self):
        """Test generation respects max_tokens limit."""
        draft_model = MockModel(vocab_size=50, is_draft=True)
        target_model = MockModel(vocab_size=50, is_draft=False)
        
        decoder = SpeculativeDecoder(draft_model, target_model, lookahead=3)
        
        prompt_tokens = mx.array([[1, 2]])
        max_tokens = 10
        
        generated_tokens = decoder.generate(prompt_tokens, max_tokens=max_tokens, temperature=1.0)
        
        assert len(generated_tokens) <= max_tokens
    
    def test_acceptance_rate_tracking(self):
        """Test acceptance rate computation."""
        draft_model = MockModel(vocab_size=50, is_draft=True)
        target_model = MockModel(vocab_size=50, is_draft=False)
        
        decoder = SpeculativeDecoder(draft_model, target_model)
        
        # Manually set statistics for testing
        decoder.total_draft_tokens = 100
        decoder.total_accepted_tokens = 75
        
        acceptance_rate = decoder.get_acceptance_rate()
        assert acceptance_rate == 0.75
        
        # Test with zero draft tokens
        decoder.total_draft_tokens = 0
        acceptance_rate = decoder.get_acceptance_rate()
        assert acceptance_rate == 0.0
    
    def test_speedup_estimate(self):
        """Test speedup estimation."""
        draft_model = MockModel(vocab_size=50, is_draft=True)
        target_model = MockModel(vocab_size=50, is_draft=False)
        
        decoder = SpeculativeDecoder(draft_model, target_model, lookahead=4)
        
        # Test with different acceptance rates
        decoder.total_draft_tokens = 100
        decoder.total_accepted_tokens = 80  # 80% acceptance
        
        speedup = decoder.get_speedup_estimate()
        assert speedup > 1.0  # Should provide speedup
        
        # Test with zero acceptance
        decoder.total_accepted_tokens = 0
        speedup = decoder.get_speedup_estimate()
        assert speedup >= 1.0  # Should be at least 1x
    
    def test_reset_statistics(self):
        """Test statistics reset."""
        draft_model = MockModel(vocab_size=50, is_draft=True)
        target_model = MockModel(vocab_size=50, is_draft=False)
        
        decoder = SpeculativeDecoder(draft_model, target_model)
        
        # Set some statistics
        decoder.total_draft_tokens = 100
        decoder.total_accepted_tokens = 75
        decoder.verification_calls = 10
        
        # Reset
        decoder.reset_statistics()
        
        assert decoder.total_draft_tokens == 0
        assert decoder.total_accepted_tokens == 0
        assert decoder.verification_calls == 0


class TestAdaptiveSpeculator:
    """Test AdaptiveSpeculator functionality."""
    
    def test_adaptive_speculator_initialization(self):
        """Test adaptive speculator initialization."""
        speculator = AdaptiveSpeculator(
            initial_lookahead=4,
            min_lookahead=1,
            max_lookahead=8
        )
        
        assert speculator.current_lookahead == 4
        assert speculator.min_lookahead == 1
        assert speculator.max_lookahead == 8
        assert len(speculator.acceptance_history) == 0
    
    def test_lookahead_adjustment_high_acceptance(self):
        """Test lookahead increases with high acceptance rate."""
        speculator = AdaptiveSpeculator(
            initial_lookahead=4,
            min_lookahead=1,
            max_lookahead=8
        )
        
        # Feed high acceptance rates
        for _ in range(5):
            speculator.update_lookahead(0.9)  # 90% acceptance
        
        # Should increase lookahead
        assert speculator.get_lookahead() > 4
        assert speculator.get_lookahead() <= 8
    
    def test_lookahead_adjustment_low_acceptance(self):
        """Test lookahead decreases with low acceptance rate."""
        speculator = AdaptiveSpeculator(
            initial_lookahead=4,
            min_lookahead=1,
            max_lookahead=8
        )
        
        # Feed low acceptance rates
        for _ in range(5):
            speculator.update_lookahead(0.2)  # 20% acceptance
        
        # Should decrease lookahead
        assert speculator.get_lookahead() < 4
        assert speculator.get_lookahead() >= 1
    
    def test_lookahead_bounds(self):
        """Test lookahead respects min/max bounds."""
        speculator = AdaptiveSpeculator(
            initial_lookahead=2,
            min_lookahead=1,
            max_lookahead=3
        )
        
        # Try to push beyond max
        for _ in range(10):
            speculator.update_lookahead(1.0)  # Perfect acceptance
        
        assert speculator.get_lookahead() <= 3
        
        # Try to push below min
        for _ in range(10):
            speculator.update_lookahead(0.0)  # Zero acceptance
        
        assert speculator.get_lookahead() >= 1
    
    def test_adaptive_window_management(self):
        """Test acceptance history window management."""
        speculator = AdaptiveSpeculator(initial_lookahead=4)
        speculator.window_size = 3  # Small window for testing
        
        # Add more data than window size
        for i in range(5):
            speculator.update_lookahead(i * 0.2)
        
        # Should only keep last 3 entries
        assert len(speculator.acceptance_history) == 3
        assert speculator.acceptance_history == [0.4, 0.6, 0.8]


class TestBatchedSpeculativeDecoding:
    """Test batched speculative decoding."""
    
    def test_batched_decode_basic(self):
        """Test basic batched speculative decoding."""
        draft_model = MockModel(vocab_size=50, is_draft=True)
        target_model = MockModel(vocab_size=50, is_draft=False)
        
        batch_prompts = [
            [1, 2, 3],
            [4, 5],
            [6, 7, 8, 9]
        ]
        
        results = batched_speculative_decode(
            batch_prompts=batch_prompts,
            draft_model=draft_model,
            target_model=target_model,
            max_tokens=5,
            lookahead=2
        )
        
        assert len(results) == len(batch_prompts)
        for result in results:
            assert isinstance(result, list)
            assert len(result) <= 5
            assert all(isinstance(token, int) for token in result)
    
    def test_batched_decode_empty_batch(self):
        """Test batched decoding with empty batch."""
        draft_model = MockModel(vocab_size=50, is_draft=True)
        target_model = MockModel(vocab_size=50, is_draft=False)
        
        results = batched_speculative_decode(
            batch_prompts=[],
            draft_model=draft_model,
            target_model=target_model,
            max_tokens=5,
            lookahead=2
        )
        
        assert len(results) == 0
    
    def test_batched_decode_single_prompt(self):
        """Test batched decoding with single prompt."""
        draft_model = MockModel(vocab_size=50, is_draft=True)
        target_model = MockModel(vocab_size=50, is_draft=False)
        
        batch_prompts = [[1, 2, 3, 4]]
        
        results = batched_speculative_decode(
            batch_prompts=batch_prompts,
            draft_model=draft_model,
            target_model=target_model,
            max_tokens=3,
            lookahead=2
        )
        
        assert len(results) == 1
        assert len(results[0]) <= 3


class TestBenchmarkSpeculativeDecoding:
    """Test speculative decoding benchmarking."""
    
    def test_benchmark_basic(self):
        """Test basic benchmarking functionality."""
        draft_model = MockModel(vocab_size=50, is_draft=True)
        target_model = MockModel(vocab_size=50, is_draft=False)
        
        test_prompts = [
            [1, 2, 3],
            [4, 5, 6, 7],
            [8, 9]
        ]
        
        stats = benchmark_speculative_decoding(
            draft_model=draft_model,
            target_model=target_model,
            test_prompts=test_prompts,
            max_tokens=5
        )
        
        # Check required statistics
        assert "standard_time_per_prompt" in stats
        assert "speculative_time_per_prompt" in stats
        assert "speedup" in stats
        assert "average_acceptance_rate" in stats
        assert "tokens_per_second_standard" in stats
        assert "tokens_per_second_speculative" in stats
        assert "efficiency_gain" in stats
        
        # Check reasonable values
        assert stats["standard_time_per_prompt"] > 0
        assert stats["speculative_time_per_prompt"] > 0
        assert stats["speedup"] > 0
        assert 0 <= stats["average_acceptance_rate"] <= 1
        assert stats["tokens_per_second_standard"] > 0
        assert stats["tokens_per_second_speculative"] > 0
    
    def test_benchmark_model_call_counts(self):
        """Test that benchmark tracks model calls correctly."""
        draft_model = MockModel(vocab_size=50, is_draft=True)
        target_model = MockModel(vocab_size=50, is_draft=False)
        
        test_prompts = [[1, 2], [3, 4]]
        max_tokens = 3
        
        # Reset call counts
        draft_model.reset_call_count()
        target_model.reset_call_count()
        
        stats = benchmark_speculative_decoding(
            draft_model=draft_model,
            target_model=target_model,
            test_prompts=test_prompts,
            max_tokens=max_tokens
        )
        
        # Should have made calls to both models
        assert draft_model.call_count > 0
        assert target_model.call_count > 0
        
        # Target model should be called for both standard and speculative runs
        # Minimum calls: len(test_prompts) for standard + some for speculative
        assert target_model.call_count >= len(test_prompts)
    
    def test_benchmark_empty_prompts(self):
        """Test benchmarking with empty prompt list."""
        draft_model = MockModel(vocab_size=50, is_draft=True)
        target_model = MockModel(vocab_size=50, is_draft=False)
        
        stats = benchmark_speculative_decoding(
            draft_model=draft_model,
            target_model=target_model,
            test_prompts=[],
            max_tokens=5
        )
        
        # Should handle empty case gracefully
        assert stats["average_acceptance_rate"] == 0


class TestSpeculativeDecodingIntegration:
    """Integration tests for speculative decoding system."""
    
    def test_end_to_end_generation(self):
        """Test complete end-to-end generation pipeline."""
        # Create models with different characteristics
        draft_model = MockModel(vocab_size=100, is_draft=True)
        target_model = MockModel(vocab_size=100, is_draft=False)
        
        decoder = SpeculativeDecoder(
            draft_model=draft_model,
            target_model=target_model,
            lookahead=3
        )
        
        prompt_tokens = mx.array([[1, 5, 10]])
        
        # Generate a longer sequence
        generated_tokens = decoder.generate(
            prompt_tokens, 
            max_tokens=20, 
            temperature=0.8
        )
        
        assert len(generated_tokens) <= 20
        assert all(0 <= token < 100 for token in generated_tokens)
        
        # Check that some speculation occurred
        assert decoder.total_draft_tokens > 0
        assert decoder.verification_calls > 0
        
        # Acceptance rate should be reasonable
        acceptance_rate = decoder.get_acceptance_rate()
        assert 0 <= acceptance_rate <= 1
    
    def test_adaptive_speculation_integration(self):
        """Test integration with adaptive speculation."""
        draft_model = MockModel(vocab_size=50, is_draft=True)
        target_model = MockModel(vocab_size=50, is_draft=False)
        
        # Start with adaptive speculator
        adaptive_speculator = AdaptiveSpeculator(
            initial_lookahead=2,
            min_lookahead=1,
            max_lookahead=5
        )
        
        # Run multiple generations to adapt
        for i in range(5):
            decoder = SpeculativeDecoder(
                draft_model=draft_model,
                target_model=target_model,
                lookahead=adaptive_speculator.get_lookahead()
            )
            
            prompt_tokens = mx.array([[i * 2, i * 2 + 1]])
            generated_tokens = decoder.generate(prompt_tokens, max_tokens=5)
            
            # Update adaptive speculation based on acceptance rate
            acceptance_rate = decoder.get_acceptance_rate()
            adaptive_speculator.update_lookahead(acceptance_rate)
        
        # Adaptive speculator should have adjusted
        final_lookahead = adaptive_speculator.get_lookahead()
        assert 1 <= final_lookahead <= 5
    
    def test_performance_characteristics(self):
        """Test performance characteristics of speculative decoding."""
        draft_model = MockModel(vocab_size=100, is_draft=True)
        target_model = MockModel(vocab_size=100, is_draft=False)
        
        # Test different lookahead values
        lookahead_values = [1, 2, 4, 8]
        results = {}
        
        for lookahead in lookahead_values:
            decoder = SpeculativeDecoder(
                draft_model=draft_model,
                target_model=target_model,
                lookahead=lookahead
            )
            
            prompt_tokens = mx.array([[10, 20, 30]])
            
            start_time = time.time()
            generated_tokens = decoder.generate(prompt_tokens, max_tokens=10)
            generation_time = time.time() - start_time
            
            results[lookahead] = {
                "time": generation_time,
                "tokens": len(generated_tokens),
                "acceptance_rate": decoder.get_acceptance_rate(),
                "speedup_estimate": decoder.get_speedup_estimate()
            }
        
        # All configurations should generate valid tokens
        for lookahead, result in results.items():
            assert result["tokens"] > 0
            assert result["time"] > 0
            assert 0 <= result["acceptance_rate"] <= 1
            assert result["speedup_estimate"] >= 1.0
    
    def test_temperature_effects(self):
        """Test effects of different temperature settings."""
        draft_model = MockModel(vocab_size=50, is_draft=True)
        target_model = MockModel(vocab_size=50, is_draft=False)
        
        decoder = SpeculativeDecoder(draft_model, target_model, lookahead=3)
        prompt_tokens = mx.array([[1, 2, 3]])
        
        temperatures = [0.0, 0.5, 1.0, 1.5]
        results = {}
        
        for temp in temperatures:
            decoder.reset_statistics()
            
            generated_tokens = decoder.generate(
                prompt_tokens, 
                max_tokens=8, 
                temperature=temp
            )
            
            results[temp] = {
                "tokens": generated_tokens,
                "acceptance_rate": decoder.get_acceptance_rate()
            }
        
        # All temperatures should produce valid outputs
        for temp, result in results.items():
            assert len(result["tokens"]) > 0
            assert all(0 <= token < 50 for token in result["tokens"])
            assert 0 <= result["acceptance_rate"] <= 1
        
        # Greedy (temp=0) should be deterministic
        decoder.reset_statistics()
        tokens1 = decoder.generate(prompt_tokens, max_tokens=5, temperature=0.0)
        decoder.reset_statistics()
        tokens2 = decoder.generate(prompt_tokens, max_tokens=5, temperature=0.0)
        assert tokens1 == tokens2


if __name__ == "__main__":
    pytest.main([__file__])
