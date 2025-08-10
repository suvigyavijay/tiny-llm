"""
Tests for Week 3, Day 7: Long Context Handling implementation.

Tests various techniques for efficiently processing very long sequences
including sliding window, hierarchical attention, and context management.
"""

import pytest
import mlx.core as mx
import time
from unittest.mock import Mock, patch
from src.tiny_llm_ref.long_context import (
    SlidingWindowAttention, HierarchicalAttention, RingAttention,
    ContextCompressor, StreamingProcessor, AdaptiveContextManager,
    benchmark_long_context_techniques
)


class TestSlidingWindowAttention:
    """Test SlidingWindowAttention functionality."""
    
    def test_sliding_window_initialization(self):
        """Test sliding window attention initialization."""
        attention = SlidingWindowAttention(window_size=128, global_tokens=4)
        
        assert attention.window_size == 128
        assert attention.global_tokens == 4
    
    def test_create_sliding_mask_basic(self):
        """Test basic sliding window mask creation."""
        attention = SlidingWindowAttention(window_size=4, global_tokens=0)
        
        mask = attention.create_sliding_mask(seq_len=6)
        
        assert mask.shape == (6, 6)
        
        # Check causal property (upper triangle should be -inf)
        for i in range(6):
            for j in range(i + 1, 6):
                assert mask[i, j] == -mx.inf
        
        # Check sliding window property
        for i in range(6):
            # Should attend to tokens within window
            start = max(0, i - 3)  # window_size - 1
            for j in range(start, i + 1):
                assert mask[i, j] == 0.0
    
    def test_create_sliding_mask_with_global_tokens(self):
        """Test sliding window mask with global tokens."""
        attention = SlidingWindowAttention(window_size=3, global_tokens=2)
        
        mask = attention.create_sliding_mask(seq_len=5)
        
        # Global tokens (first 2) should attend to everything up to their position
        assert mask[0, 0] == 0.0
        assert mask[1, 0] == 0.0
        assert mask[1, 1] == 0.0
        
        # All tokens should attend to global tokens
        for i in range(5):
            assert mask[i, 0] == 0.0
            assert mask[i, 1] == 0.0
    
    def test_sliding_attention_computation(self):
        """Test sliding window attention computation."""
        attention = SlidingWindowAttention(window_size=4, global_tokens=1)
        
        batch_size, num_heads, seq_len, head_dim = 2, 4, 8, 16
        q = mx.random.normal((batch_size, num_heads, seq_len, head_dim))
        k = mx.random.normal((batch_size, num_heads, seq_len, head_dim))
        v = mx.random.normal((batch_size, num_heads, seq_len, head_dim))
        
        output = attention.sliding_attention(q, k, v)
        
        assert output.shape == (batch_size, num_heads, seq_len, head_dim)
        assert not mx.isnan(output).any()
    
    def test_sliding_attention_with_scale(self):
        """Test sliding window attention with custom scale."""
        attention = SlidingWindowAttention(window_size=4)
        
        batch_size, num_heads, seq_len, head_dim = 1, 2, 6, 8
        q = mx.random.normal((batch_size, num_heads, seq_len, head_dim))
        k = mx.random.normal((batch_size, num_heads, seq_len, head_dim))
        v = mx.random.normal((batch_size, num_heads, seq_len, head_dim))
        
        scale = 0.5
        output = attention.sliding_attention(q, k, v, scale=scale)
        
        assert output.shape == (batch_size, num_heads, seq_len, head_dim)


class TestHierarchicalAttention:
    """Test HierarchicalAttention functionality."""
    
    def test_hierarchical_attention_initialization(self):
        """Test hierarchical attention initialization."""
        levels = [32, 128, 512]
        attention = HierarchicalAttention(levels=levels)
        
        assert attention.levels == levels
        assert attention.num_levels == 3
    
    def test_create_hierarchical_masks(self):
        """Test hierarchical mask creation."""
        attention = HierarchicalAttention(levels=[4, 8])
        
        masks = attention.create_hierarchical_masks(seq_len=10)
        
        assert len(masks) == 2
        assert all(mask.shape == (10, 10) for mask in masks)
        
        # Check that each mask has different attention patterns
        mask1, mask2 = masks
        
        # Both should be causal
        for mask in masks:
            for i in range(10):
                for j in range(i + 1, 10):
                    assert mask[i, j] == -mx.inf
    
    def test_hierarchical_forward_basic(self):
        """Test hierarchical attention forward pass."""
        attention = HierarchicalAttention(levels=[4, 8])
        
        batch_size, num_heads, seq_len, head_dim = 1, 2, 12, 16
        q = mx.random.normal((batch_size, num_heads, seq_len, head_dim))
        k = mx.random.normal((batch_size, num_heads, seq_len, head_dim))
        v = mx.random.normal((batch_size, num_heads, seq_len, head_dim))
        
        output = attention.hierarchical_forward(q, k, v)
        
        assert output.shape == (batch_size, num_heads, seq_len, head_dim)
        assert not mx.isnan(output).any()
    
    def test_hierarchical_forward_with_weights(self):
        """Test hierarchical attention with custom level weights."""
        attention = HierarchicalAttention(levels=[4, 8, 16])
        
        batch_size, num_heads, seq_len, head_dim = 1, 2, 10, 8
        q = mx.random.normal((batch_size, num_heads, seq_len, head_dim))
        k = mx.random.normal((batch_size, num_heads, seq_len, head_dim))
        v = mx.random.normal((batch_size, num_heads, seq_len, head_dim))
        
        layer_weights = [0.5, 0.3, 0.2]
        output = attention.hierarchical_forward(q, k, v, layer_weights=layer_weights)
        
        assert output.shape == (batch_size, num_heads, seq_len, head_dim)


class TestRingAttention:
    """Test RingAttention functionality."""
    
    def test_ring_attention_initialization(self):
        """Test ring attention initialization."""
        ring_attention = RingAttention(ring_size=4, block_size=128)
        
        assert ring_attention.ring_size == 4
        assert ring_attention.block_size == 128
    
    def test_partition_sequence(self):
        """Test sequence partitioning for ring processing."""
        ring_attention = RingAttention(ring_size=3, block_size=64)
        
        partitions = ring_attention.partition_sequence(seq_len=100)
        
        assert len(partitions) == 3
        assert all(isinstance(partition, tuple) for partition in partitions)
        assert all(len(partition) == 2 for partition in partitions)
        
        # Check partitions cover the sequence
        starts, ends = zip(*partitions)
        assert starts[0] == 0
        assert ends[-1] == 100
        
        # Check partitions don't overlap (except boundaries)
        for i in range(len(partitions) - 1):
            assert partitions[i][1] <= partitions[i + 1][0]
    
    def test_partition_sequence_exact_division(self):
        """Test partitioning with exact division."""
        ring_attention = RingAttention(ring_size=4, block_size=64)
        
        partitions = ring_attention.partition_sequence(seq_len=120)  # 30 per partition
        
        assert len(partitions) == 4
        assert partitions[0] == (0, 30)
        assert partitions[1] == (30, 60)
        assert partitions[2] == (60, 90)
        assert partitions[3] == (90, 120)
    
    def test_ring_attention_step(self):
        """Test single ring attention step."""
        ring_attention = RingAttention(ring_size=2, block_size=32)
        
        batch_size, num_heads, seq_len, head_dim = 1, 2, 16, 8
        q_block = mx.random.normal((batch_size, num_heads, seq_len, head_dim))
        k_block = mx.random.normal((batch_size, num_heads, seq_len, head_dim))
        v_block = mx.random.normal((batch_size, num_heads, seq_len, head_dim))
        
        output, weights = ring_attention.ring_attention_step(
            q_block, k_block, v_block, ring_position=0, causal_offset=0
        )
        
        assert output.shape == (batch_size, num_heads, seq_len, head_dim)
        assert weights.shape == (batch_size, num_heads, seq_len, seq_len)
        assert not mx.isnan(output).any()
        assert not mx.isnan(weights).any()
    
    def test_ring_attention_full(self):
        """Test complete ring attention computation."""
        ring_attention = RingAttention(ring_size=2, block_size=16)
        
        batch_size, num_heads, seq_len, head_dim = 1, 2, 24, 8
        q = mx.random.normal((batch_size, num_heads, seq_len, head_dim))
        k = mx.random.normal((batch_size, num_heads, seq_len, head_dim))
        v = mx.random.normal((batch_size, num_heads, seq_len, head_dim))
        
        output = ring_attention.ring_attention(q, k, v)
        
        assert output.shape == (batch_size, num_heads, seq_len, head_dim)
        assert not mx.isnan(output).any()


class TestContextCompressor:
    """Test ContextCompressor functionality."""
    
    def test_context_compressor_initialization(self):
        """Test context compressor initialization."""
        compressor = ContextCompressor(compression_ratio=0.6, importance_method="attention")
        
        assert compressor.compression_ratio == 0.6
        assert compressor.importance_method == "attention"
    
    def test_compute_attention_importance(self):
        """Test attention-based importance computation."""
        compressor = ContextCompressor(importance_method="attention")
        
        seq_len = 10
        tokens = mx.random.normal((seq_len, 64))
        attention_weights = mx.random.uniform((4, 8, seq_len, seq_len))  # [layers, heads, seq, seq]
        
        importance = compressor.compute_attention_importance(tokens, attention_weights)
        
        assert importance.shape == (seq_len,)
        assert mx.all(importance >= 0)
    
    def test_compute_gradient_importance(self):
        """Test gradient-based importance computation."""
        compressor = ContextCompressor(importance_method="gradient")
        
        seq_len, hidden_dim = 8, 32
        tokens = mx.random.normal((seq_len, hidden_dim))
        gradients = mx.random.normal((seq_len, hidden_dim))
        
        importance = compressor.compute_gradient_importance(tokens, gradients)
        
        assert importance.shape == (seq_len,)
        assert mx.all(importance >= 0)
    
    def test_compress_context_basic(self):
        """Test basic context compression."""
        compressor = ContextCompressor(compression_ratio=0.5)
        
        seq_len = 20
        tokens = mx.random.normal((seq_len,))  # 1D token sequence
        importance_scores = mx.random.uniform(shape=(seq_len,))
        
        compressed_tokens, position_mapping = compressor.compress_context(
            tokens, importance_scores
        )
        
        expected_length = int(seq_len * 0.5)
        assert len(compressed_tokens) <= expected_length + 2  # +2 for preserved positions
        assert position_mapping.shape == (seq_len,)
    
    def test_compress_context_with_preserved_positions(self):
        """Test compression with preserved positions."""
        compressor = ContextCompressor(compression_ratio=0.4)
        
        seq_len = 15
        tokens = mx.arange(seq_len)  # Sequential tokens for easy verification
        importance_scores = mx.random.uniform(shape=(seq_len,))
        
        preserve_positions = [0, 5, seq_len - 1]
        compressed_tokens, position_mapping = compressor.compress_context(
            tokens, importance_scores, preserve_positions=preserve_positions
        )
        
        # Preserved positions should be in compressed tokens
        assert 0 in compressed_tokens  # First token
        assert seq_len - 1 in compressed_tokens  # Last token
        assert 5 in compressed_tokens  # Middle preserved token
    
    def test_compress_context_2d_tokens(self):
        """Test compression with 2D token arrays."""
        compressor = ContextCompressor(compression_ratio=0.6)
        
        batch_size, seq_len = 2, 12
        tokens = mx.random.normal((batch_size, seq_len))
        importance_scores = mx.random.uniform(shape=(seq_len,))
        
        compressed_tokens, position_mapping = compressor.compress_context(
            tokens, importance_scores
        )
        
        expected_length = int(seq_len * 0.6)
        assert compressed_tokens.shape[0] == batch_size
        assert compressed_tokens.shape[1] <= expected_length + 2


class TestStreamingProcessor:
    """Test StreamingProcessor functionality."""
    
    def test_streaming_processor_initialization(self):
        """Test streaming processor initialization."""
        processor = StreamingProcessor(chunk_size=512, overlap=64)
        
        assert processor.chunk_size == 512
        assert processor.overlap == 64
    
    def test_chunk_sequence_basic(self):
        """Test basic sequence chunking."""
        processor = StreamingProcessor(chunk_size=8, overlap=2)
        
        sequence = mx.arange(20)  # 20-element sequence
        chunks = processor.chunk_sequence(sequence)
        
        assert len(chunks) > 1
        assert all(isinstance(chunk, tuple) for chunk in chunks)
        assert all(len(chunk) == 3 for chunk in chunks)  # (chunk, start, end)
        
        # Check chunk coverage
        first_chunk, first_start, first_end = chunks[0]
        assert first_start == 0
        assert len(first_chunk) == min(8, 20)
        
        # Check overlap
        if len(chunks) > 1:
            second_chunk, second_start, second_end = chunks[1]
            overlap_size = first_end - second_start
            assert overlap_size == processor.overlap
    
    def test_chunk_sequence_2d(self):
        """Test chunking 2D sequences."""
        processor = StreamingProcessor(chunk_size=6, overlap=1)
        
        sequence = mx.random.normal((3, 15))  # [batch, seq_len]
        chunks = processor.chunk_sequence(sequence)
        
        for chunk, start, end in chunks:
            assert chunk.shape[0] == 3  # Batch dimension preserved
            assert chunk.shape[1] == end - start
    
    def test_chunk_sequence_short(self):
        """Test chunking sequence shorter than chunk size."""
        processor = StreamingProcessor(chunk_size=10, overlap=2)
        
        sequence = mx.arange(5)
        chunks = processor.chunk_sequence(sequence)
        
        assert len(chunks) == 1
        chunk, start, end = chunks[0]
        assert start == 0
        assert end == 5
        assert len(chunk) == 5
    
    def test_process_stream(self):
        """Test streaming processing."""
        processor = StreamingProcessor(chunk_size=4, overlap=1)
        
        # Mock model
        mock_model = Mock()
        mock_model.return_value = mx.array([1.0])  # Simple output
        
        # Create token stream
        def token_generator():
            for i in range(3):
                yield mx.array([i, i+1])
        
        results = list(processor.process_stream(token_generator(), mock_model))
        
        assert len(results) > 0
        assert mock_model.call_count > 0
    
    def test_merge_chunk_outputs_basic(self):
        """Test merging chunk outputs."""
        processor = StreamingProcessor(chunk_size=4, overlap=1)
        
        # Mock chunk outputs
        chunk_outputs = [
            mx.array([[1.0, 2.0], [3.0, 4.0]]),  # 2 tokens, 2 features
            mx.array([[4.5, 5.0], [6.0, 7.0]]),  # Overlapping chunk
            mx.array([[7.5, 8.0]])               # Final chunk
        ]
        
        chunk_positions = [(0, 2), (1, 3), (2, 3)]
        
        merged = processor.merge_chunk_outputs(chunk_outputs, chunk_positions)
        
        assert merged.shape[0] == 3  # Total length
        assert merged.shape[1] == 2  # Feature dimension
        assert not mx.isnan(merged).any()
    
    def test_merge_chunk_outputs_empty(self):
        """Test merging empty chunk outputs."""
        processor = StreamingProcessor(chunk_size=4, overlap=1)
        
        merged = processor.merge_chunk_outputs([], [])
        
        assert merged.size == 0


class TestAdaptiveContextManager:
    """Test AdaptiveContextManager functionality."""
    
    def test_adaptive_context_manager_initialization(self):
        """Test adaptive context manager initialization."""
        manager = AdaptiveContextManager(max_context=1000, summary_ratio=0.2)
        
        assert manager.max_context == 1000
        assert manager.summary_ratio == 0.2
        assert len(manager.conversation_segments) == 0
    
    def test_add_message(self):
        """Test adding messages to conversation."""
        manager = AdaptiveContextManager(max_context=1000)
        
        manager.add_message("Hello, how are you?", role="user")
        manager.add_message("I'm doing well, thank you!", role="assistant")
        
        assert len(manager.conversation_segments) == 2
        assert manager.conversation_segments[0]["content"] == "Hello, how are you?"
        assert manager.conversation_segments[0]["role"] == "user"
        assert manager.conversation_segments[1]["role"] == "assistant"
    
    def test_estimate_token_count(self):
        """Test token count estimation."""
        manager = AdaptiveContextManager(max_context=1000)
        
        text = "This is a test sentence with multiple words."
        token_count = manager.estimate_token_count(text)
        
        assert token_count > 0
        assert isinstance(token_count, (int, float))
        
        # Longer text should have more tokens
        longer_text = text + " " + text
        longer_count = manager.estimate_token_count(longer_text)
        assert longer_count > token_count
    
    def test_extract_key_points(self):
        """Test key point extraction."""
        manager = AdaptiveContextManager(max_context=1000)
        
        text = "This is important information. Some filler text here. This is also critical data. More filler. Essential details here."
        key_points = manager.extract_key_points(text)
        
        assert isinstance(key_points, list)
        assert len(key_points) <= 3  # Should extract top points
        assert all(isinstance(point, str) for point in key_points)
        
        # Should prefer sentences with importance keywords
        key_text = " ".join(key_points).lower()
        assert "important" in key_text or "critical" in key_text or "essential" in key_text
    
    def test_summarize_old_context(self):
        """Test context summarization."""
        manager = AdaptiveContextManager(max_context=1000)
        
        segments = [
            {"content": "Important point about machine learning.", "role": "user"},
            {"content": "Key insight about neural networks.", "role": "assistant"},
            {"content": "Critical information about data processing.", "role": "user"}
        ]
        
        summary = manager.summarize_old_context(segments)
        
        assert isinstance(summary, str)
        assert len(summary) > 0
        assert "summary" in summary.lower()
    
    def test_manage_context_within_limit(self):
        """Test context management when within limits."""
        manager = AdaptiveContextManager(max_context=1000)
        
        # Add short messages
        manager.add_message("Short message 1")
        manager.add_message("Short message 2")
        
        context = manager.manage_context("New short message")
        
        # Should return full conversation since it's within limits
        assert "Short message 1" in context
        assert "Short message 2" in context
        assert "New short message" in context
    
    def test_manage_context_exceeds_limit(self):
        """Test context management when exceeding limits."""
        manager = AdaptiveContextManager(max_context=50, summary_ratio=0.5)  # Very small limit
        
        # Add long messages to exceed limit
        long_message = "This is a very long message that contains lots of text to exceed the context limit."
        
        for i in range(5):
            manager.add_message(f"{long_message} Message {i}")
        
        context = manager.manage_context("Final message")
        
        # Should have summary + recent context
        assert len(context) > 0
        assert "summary" in context.lower() or len(manager.conversation_segments) < 5


class TestLongContextBenchmarking:
    """Test long context benchmarking functionality."""
    
    def test_benchmark_long_context_techniques(self):
        """Test benchmarking different long context techniques."""
        # Mock models
        models = {
            "standard": Mock(),
            "sliding_window": Mock(),
        }
        
        # Create test sequences of different lengths
        test_sequences = [
            mx.random.normal((1, 512)),
            mx.random.normal((1, 1024)),
            mx.random.normal((1, 2048))
        ]
        
        results = benchmark_long_context_techniques(models, test_sequences)
        
        assert isinstance(results, dict)
        # Should have results for each technique tested
        for technique_name in results:
            technique_result = results[technique_name]
            assert "avg_time" in technique_result
            assert "sequences_processed" in technique_result
            assert technique_result["sequences_processed"] >= 0
    
    def test_benchmark_empty_sequences(self):
        """Test benchmarking with empty sequence list."""
        models = {"test_model": Mock()}
        
        results = benchmark_long_context_techniques(models, [])
        
        # Should handle empty case gracefully
        assert isinstance(results, dict)


class TestLongContextIntegration:
    """Integration tests for long context techniques."""
    
    def test_sliding_window_vs_standard_attention(self):
        """Test sliding window attention vs standard attention."""
        # Create a moderately long sequence
        batch_size, num_heads, seq_len, head_dim = 1, 4, 128, 32
        q = mx.random.normal((batch_size, num_heads, seq_len, head_dim))
        k = mx.random.normal((batch_size, num_heads, seq_len, head_dim))
        v = mx.random.normal((batch_size, num_heads, seq_len, head_dim))
        
        # Sliding window attention
        sliding_attention = SlidingWindowAttention(window_size=32)
        
        start_time = time.time()
        sliding_output = sliding_attention.sliding_attention(q, k, v)
        sliding_time = time.time() - start_time
        
        # Verify output
        assert sliding_output.shape == (batch_size, num_heads, seq_len, head_dim)
        assert not mx.isnan(sliding_output).any()
        assert sliding_time < 10.0  # Should complete reasonably quickly
    
    def test_hierarchical_attention_scaling(self):
        """Test hierarchical attention with different scales."""
        sequences = [64, 128, 256]
        
        for seq_len in sequences:
            attention = HierarchicalAttention(levels=[16, 64, seq_len])
            
            batch_size, num_heads, head_dim = 1, 2, 16
            q = mx.random.normal((batch_size, num_heads, seq_len, head_dim))
            k = mx.random.normal((batch_size, num_heads, seq_len, head_dim))
            v = mx.random.normal((batch_size, num_heads, seq_len, head_dim))
            
            output = attention.hierarchical_forward(q, k, v)
            
            assert output.shape == (batch_size, num_heads, seq_len, head_dim)
            assert not mx.isnan(output).any()
    
    def test_context_compression_efficiency(self):
        """Test context compression efficiency."""
        compressor = ContextCompressor(compression_ratio=0.3)
        
        # Test with different sequence lengths
        for seq_len in [100, 500, 1000]:
            tokens = mx.random.normal((seq_len, 64))
            importance = mx.random.uniform(shape=(seq_len,))
            
            start_time = time.time()
            compressed, mapping = compressor.compress_context(tokens, importance)
            compression_time = time.time() - start_time
            
            # Should achieve compression
            assert len(compressed) < seq_len
            
            # Should complete quickly
            assert compression_time < 5.0
            
            # Mapping should be valid
            assert mapping.shape == (seq_len,)
    
    def test_streaming_processor_with_real_chunks(self):
        """Test streaming processor with realistic chunk processing."""
        processor = StreamingProcessor(chunk_size=64, overlap=8)
        
        # Create a long sequence
        long_sequence = mx.random.normal((2048,))
        
        # Test chunking
        chunks = processor.chunk_sequence(long_sequence)
        
        assert len(chunks) > 1
        
        # Verify coverage
        total_coverage = 0
        for chunk, start, end in chunks:
            assert end - start == len(chunk)
            total_coverage = max(total_coverage, end)
        
        assert total_coverage == len(long_sequence)
        
        # Test merging mock outputs
        mock_outputs = []
        positions = []
        
        for chunk, start, end in chunks:
            # Mock output: mean of chunk as a simple transformation
            output = mx.array([[mx.mean(chunk), mx.std(chunk)]])
            mock_outputs.append(output)
            positions.append((start, end))
        
        merged = processor.merge_chunk_outputs(mock_outputs, positions)
        assert merged.shape[0] == len(long_sequence)
    
    def test_adaptive_context_manager_conversation_flow(self):
        """Test adaptive context manager with realistic conversation."""
        manager = AdaptiveContextManager(max_context=300, summary_ratio=0.3)
        
        # Simulate a long conversation
        conversation = [
            ("What is machine learning?", "user"),
            ("Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed.", "assistant"),
            ("Can you explain neural networks?", "user"),
            ("Neural networks are computing systems inspired by biological neural networks. They consist of layers of interconnected nodes that process information.", "assistant"),
            ("What about deep learning?", "user"),
            ("Deep learning is a subset of machine learning that uses neural networks with multiple hidden layers to model and understand complex patterns.", "assistant"),
            ("How do I get started with ML?", "user")
        ]
        
        # Add messages progressively
        for i, (message, role) in enumerate(conversation):
            if i == len(conversation) - 1:
                # Last message through manage_context
                context = manager.manage_context(message)
            else:
                manager.add_message(message, role)
        
        # Should have managed context appropriately
        assert isinstance(context, str)
        assert len(context) > 0
        
        # Should contain recent information
        assert "get started" in context.lower() or "ml" in context.lower()


if __name__ == "__main__":
    pytest.main([__file__])
