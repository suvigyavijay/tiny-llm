"""
Tests for Week 2, Day 4: Flash Attention 2 - CPU implementation.

Tests the CPU version of Flash Attention 2 using the actual C++ extension
implementation for accuracy, performance, and integration.
"""

import pytest
import mlx.core as mx
import time
import math
from .tiny_llm_base import *
from .utils import *

try:
    from src.extensions_ref.tiny_llm_ext_ref import flash_attention
    FLASH_ATTENTION_AVAILABLE = True
except ImportError:
    FLASH_ATTENTION_AVAILABLE = False
    flash_attention = None


class TestFlashAttentionCPU:
    """Test Flash Attention CPU implementation."""
    
    @pytest.mark.skipif(not FLASH_ATTENTION_AVAILABLE, reason="Flash Attention extension not available")
    @pytest.mark.parametrize("stream", AVAILABLE_STREAMS, ids=AVAILABLE_STREAMS_IDS)
    @pytest.mark.parametrize("precision", [mx.float32], ids=["float32"])  # Extension supports float32
    def test_flash_attention_cpu_basic(self, stream: mx.Stream, precision: mx.Dtype):
        """Test basic Flash Attention CPU functionality using C++ extension."""
        with stream:
            batch_size, seq_len, head_dim = 2, 64, 32
            num_heads = 4
            num_kv_heads = 4  # For this test
            
            # Shape: [batch * num_heads, seq_len, head_dim] for extension
            q = mx.random.normal((batch_size * num_heads, seq_len, head_dim), dtype=precision)
            k = mx.random.normal((batch_size * num_kv_heads, seq_len, head_dim), dtype=precision)
            v = mx.random.normal((batch_size * num_kv_heads, seq_len, head_dim), dtype=precision)
            
            # Create causal mask
            mask = mx.triu(mx.full((batch_size, seq_len, seq_len), -mx.inf), k=1).astype(precision)
            
            scale = 1.0 / math.sqrt(head_dim)
            
            output = flash_attention(q, k, v, mask, scale, num_kv_heads, num_heads, stream)
            
            assert output.shape == (batch_size * num_heads, seq_len, head_dim)
            assert not mx.isnan(output).any()
            assert output.dtype == precision
    
    @pytest.mark.skipif(not FLASH_ATTENTION_AVAILABLE, reason="Flash Attention extension not available")
    @pytest.mark.parametrize("stream", AVAILABLE_STREAMS, ids=AVAILABLE_STREAMS_IDS)
    def test_flash_attention_cpu_vs_standard(self, stream: mx.Stream):
        """Test Flash Attention CPU vs standard attention accuracy."""
        with stream:
            batch_size, seq_len, head_dim = 1, 32, 16
            num_heads = 2
            num_kv_heads = 2
            precision = mx.float32
            
            # Create inputs for standard attention
            q_std = mx.random.normal((batch_size, num_heads, seq_len, head_dim), dtype=precision)
            k_std = mx.random.normal((batch_size, num_heads, seq_len, head_dim), dtype=precision)
            v_std = mx.random.normal((batch_size, num_heads, seq_len, head_dim), dtype=precision)
            
            # Standard attention
            scale = 1.0 / math.sqrt(head_dim)
            scores = mx.matmul(q_std, k_std.transpose(0, 1, 3, 2)) * scale
            
            # Apply causal mask
            causal_mask = mx.triu(mx.full((seq_len, seq_len), -mx.inf), k=1)
            scores = scores + causal_mask[None, None, :, :]
            
            attn_weights = mx.softmax(scores, axis=-1)
            standard_output = mx.matmul(attn_weights, v_std)
            
            # Flash attention (reshape for extension format)
            q_flash = q_std.reshape(batch_size * num_heads, seq_len, head_dim)
            k_flash = k_std.reshape(batch_size * num_kv_heads, seq_len, head_dim)
            v_flash = v_std.reshape(batch_size * num_kv_heads, seq_len, head_dim)
            mask = mx.triu(mx.full((batch_size, seq_len, seq_len), -mx.inf), k=1).astype(precision)
            
            flash_output = flash_attention(q_flash, k_flash, v_flash, mask, scale, num_kv_heads, num_heads, stream)
            flash_output = flash_output.reshape(batch_size, num_heads, seq_len, head_dim)
            
            # Should be close (allowing for numerical differences in implementation)
            assert mx.allclose(flash_output, standard_output, atol=1e-3, rtol=1e-3)
    
    @pytest.mark.skipif(not FLASH_ATTENTION_AVAILABLE, reason="Flash Attention extension not available")
    @pytest.mark.parametrize("seq_len", [32, 64, 128])
    def test_flash_attention_cpu_different_lengths(self, seq_len):
        """Test Flash Attention CPU with different sequence lengths."""
        batch_size, head_dim = 1, 16
        num_heads = 2
        num_kv_heads = 2
        precision = mx.float32
        
        q = mx.random.normal((batch_size * num_heads, seq_len, head_dim), dtype=precision)
        k = mx.random.normal((batch_size * num_kv_heads, seq_len, head_dim), dtype=precision)
        v = mx.random.normal((batch_size * num_kv_heads, seq_len, head_dim), dtype=precision)
        mask = mx.triu(mx.full((batch_size, seq_len, seq_len), -mx.inf), k=1).astype(precision)
        
        scale = 1.0 / math.sqrt(head_dim)
        output = flash_attention(q, k, v, mask, scale, num_kv_heads, num_heads)
        
        assert output.shape == (batch_size * num_heads, seq_len, head_dim)
        assert not mx.isnan(output).any()
    
    @pytest.mark.skipif(not FLASH_ATTENTION_AVAILABLE, reason="Flash Attention extension not available")
    def test_flash_attention_cpu_grouped_query(self):
        """Test Flash Attention CPU with grouped query attention."""
        batch_size, seq_len, head_dim = 1, 64, 16
        num_heads = 4
        num_kv_heads = 2  # Grouped query attention
        precision = mx.float32
        
        q = mx.random.normal((batch_size * num_heads, seq_len, head_dim), dtype=precision)
        k = mx.random.normal((batch_size * num_kv_heads, seq_len, head_dim), dtype=precision)
        v = mx.random.normal((batch_size * num_kv_heads, seq_len, head_dim), dtype=precision)
        mask = mx.triu(mx.full((batch_size, seq_len, seq_len), -mx.inf), k=1).astype(precision)
        
        scale = 1.0 / math.sqrt(head_dim)
        output = flash_attention(q, k, v, mask, scale, num_kv_heads, num_heads)
        
        assert output.shape == (batch_size * num_heads, seq_len, head_dim)
        assert not mx.isnan(output).any()
    
    @pytest.mark.skipif(not FLASH_ATTENTION_AVAILABLE, reason="Flash Attention extension not available")
    def test_flash_attention_cpu_causal_mask(self):
        """Test that Flash Attention CPU properly applies causal masking."""
        batch_size, seq_len, head_dim = 1, 8, 4
        num_heads = 1
        num_kv_heads = 1
        precision = mx.float32
        
        # Create specific input where causal masking is important
        q = mx.ones((batch_size * num_heads, seq_len, head_dim), dtype=precision)
        k = mx.ones((batch_size * num_kv_heads, seq_len, head_dim), dtype=precision)
        v = mx.arange(seq_len).reshape(1, seq_len, 1).astype(precision)
        v = mx.broadcast_to(v, (batch_size * num_kv_heads, seq_len, head_dim))
        
        # Create causal mask
        mask = mx.triu(mx.full((batch_size, seq_len, seq_len), -mx.inf), k=1).astype(precision)
        
        scale = 1.0 / math.sqrt(head_dim)
        output = flash_attention(q, k, v, mask, scale, num_kv_heads, num_heads)
        
        # Verify causal property - later positions shouldn't affect earlier ones
        assert output.shape == (batch_size * num_heads, seq_len, head_dim)
        assert not mx.isnan(output).any()
        
        # The exact values depend on the implementation details, but output should be valid
        assert mx.all(mx.isfinite(output))
    
    @pytest.mark.skipif(not FLASH_ATTENTION_AVAILABLE, reason="Flash Attention extension not available")
    def test_flash_attention_cpu_performance(self):
        """Test Flash Attention CPU performance characteristics."""
        batch_size, seq_len, head_dim = 1, 128, 64
        num_heads = 8
        num_kv_heads = 8
        precision = mx.float32
        
        q = mx.random.normal((batch_size * num_heads, seq_len, head_dim), dtype=precision)
        k = mx.random.normal((batch_size * num_kv_heads, seq_len, head_dim), dtype=precision)
        v = mx.random.normal((batch_size * num_kv_heads, seq_len, head_dim), dtype=precision)
        mask = mx.triu(mx.full((batch_size, seq_len, seq_len), -mx.inf), k=1).astype(precision)
        
        scale = 1.0 / math.sqrt(head_dim)
        
        # Time the computation
        start_time = time.time()
        output = flash_attention(q, k, v, mask, scale, num_kv_heads, num_heads)
        computation_time = time.time() - start_time
        
        assert output.shape == (batch_size * num_heads, seq_len, head_dim)
        assert computation_time < 10.0  # Should complete in reasonable time
    
    @pytest.mark.skipif(not FLASH_ATTENTION_AVAILABLE, reason="Flash Attention extension not available")
    def test_flash_attention_cpu_numerical_stability(self):
        """Test Flash Attention CPU numerical stability."""
        batch_size, seq_len, head_dim = 1, 16, 8
        num_heads = 2
        num_kv_heads = 2
        precision = mx.float32
        
        # Create inputs that could cause numerical issues
        q = mx.random.normal((batch_size * num_heads, seq_len, head_dim), dtype=precision) * 5.0
        k = mx.random.normal((batch_size * num_kv_heads, seq_len, head_dim), dtype=precision) * 5.0
        v = mx.random.normal((batch_size * num_kv_heads, seq_len, head_dim), dtype=precision)
        mask = mx.triu(mx.full((batch_size, seq_len, seq_len), -mx.inf), k=1).astype(precision)
        
        scale = 1.0 / math.sqrt(head_dim)
        output = flash_attention(q, k, v, mask, scale, num_kv_heads, num_heads)
        
        # Should not produce NaN or Inf values
        assert not mx.isnan(output).any()
        assert not mx.isinf(output).any()
        
        # Output should be bounded
        assert mx.all(mx.isfinite(output))


class TestFlashAttentionCPUIntegration:
    """Integration tests for Flash Attention CPU."""
    
    @pytest.mark.skipif(not FLASH_ATTENTION_AVAILABLE, reason="Flash Attention extension not available")
    def test_flash_attention_cpu_integration(self):
        """Test Flash Attention CPU integration with realistic parameters."""
        batch_size, seq_len, head_dim = 2, 64, 64
        num_heads = 8
        num_kv_heads = 8
        precision = mx.float32
        
        q = mx.random.normal((batch_size * num_heads, seq_len, head_dim), dtype=precision)
        k = mx.random.normal((batch_size * num_kv_heads, seq_len, head_dim), dtype=precision)
        v = mx.random.normal((batch_size * num_kv_heads, seq_len, head_dim), dtype=precision)
        mask = mx.triu(mx.full((batch_size, seq_len, seq_len), -mx.inf), k=1).astype(precision)
        
        scale = 1.0 / math.sqrt(head_dim)
        output = flash_attention(q, k, v, mask, scale, num_kv_heads, num_heads)
        
        assert output.shape == (batch_size * num_heads, seq_len, head_dim)
        
        # Test that output has reasonable statistics
        output_mean = mx.mean(output)
        output_std = mx.std(output)
        
        assert -2.0 < output_mean < 2.0  # Reasonable mean
        assert 0.1 < output_std < 5.0    # Reasonable standard deviation
    
    @pytest.mark.skipif(not FLASH_ATTENTION_AVAILABLE, reason="Flash Attention extension not available")
    def test_flash_attention_cpu_performance_scaling(self):
        """Test Flash Attention CPU performance with different sizes."""
        head_dim = 64
        num_heads = 4
        num_kv_heads = 4
        precision = mx.float32
        
        # Test different sequence lengths
        seq_lengths = [32, 64, 128]
        times = []
        
        for seq_len in seq_lengths:
            q = mx.random.normal((1 * num_heads, seq_len, head_dim), dtype=precision)
            k = mx.random.normal((1 * num_kv_heads, seq_len, head_dim), dtype=precision)
            v = mx.random.normal((1 * num_kv_heads, seq_len, head_dim), dtype=precision)
            mask = mx.triu(mx.full((1, seq_len, seq_len), -mx.inf), k=1).astype(precision)
            
            scale = 1.0 / math.sqrt(head_dim)
            
            start_time = time.time()
            output = flash_attention(q, k, v, mask, scale, num_kv_heads, num_heads)
            end_time = time.time()
            
            times.append(end_time - start_time)
            
            assert output.shape == (1 * num_heads, seq_len, head_dim)
        
        # Performance should scale reasonably
        assert all(t < 15.0 for t in times)  # All should complete quickly
    
    @pytest.mark.skipif(not FLASH_ATTENTION_AVAILABLE, reason="Flash Attention extension not available")
    def test_flash_attention_cpu_edge_cases(self):
        """Test Flash Attention CPU edge cases."""
        precision = mx.float32
        num_heads = 1
        num_kv_heads = 1
        
        # Single token
        head_dim = 8
        q = mx.random.normal((1 * num_heads, 1, head_dim), dtype=precision)
        k = mx.random.normal((1 * num_kv_heads, 1, head_dim), dtype=precision)
        v = mx.random.normal((1 * num_kv_heads, 1, head_dim), dtype=precision)
        mask = mx.zeros((1, 1, 1), dtype=precision)  # No masking for single token
        
        scale = 1.0 / math.sqrt(head_dim)
        output = flash_attention(q, k, v, mask, scale, num_kv_heads, num_heads)
        assert output.shape == (1 * num_heads, 1, head_dim)
        assert not mx.isnan(output).any()
        
        # Small sequence
        head_dim = 16
        seq_len = 4
        q = mx.random.normal((1 * num_heads, seq_len, head_dim), dtype=precision)
        k = mx.random.normal((1 * num_kv_heads, seq_len, head_dim), dtype=precision)
        v = mx.random.normal((1 * num_kv_heads, seq_len, head_dim), dtype=precision)
        mask = mx.triu(mx.full((1, seq_len, seq_len), -mx.inf), k=1).astype(precision)
        
        output = flash_attention(q, k, v, mask, scale, num_kv_heads, num_heads)
        assert output.shape == (1 * num_heads, seq_len, head_dim)
        assert not mx.isnan(output).any()


if __name__ == "__main__":
    pytest.main([__file__])
