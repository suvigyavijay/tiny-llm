"""
Tests for Week 2, Day 5: Flash Attention 2 - GPU implementation.

Tests GPU-specific aspects of Flash Attention 2 including Metal kernel usage,
device memory management, and performance characteristics on GPU vs CPU.
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


class TestFlashAttentionGPU:
    """Test Flash Attention GPU implementation."""
    
    @pytest.mark.skipif(not FLASH_ATTENTION_AVAILABLE, reason="Flash Attention extension not available")
    def test_flash_attention_gpu_availability(self):
        """Test that GPU/Metal is available for Flash Attention."""
        if not mx.metal.is_available():
            pytest.skip("Metal GPU not available")
        
        # Basic availability test
        assert mx.metal.is_available()
        
        # Test GPU device creation
        with mx.gpu:
            x = mx.array([1.0, 2.0, 3.0])
            assert x.device() == mx.gpu
    
    @pytest.mark.skipif(not FLASH_ATTENTION_AVAILABLE, reason="Flash Attention extension not available")
    def test_flash_attention_gpu_vs_cpu_performance(self):
        """Compare Flash Attention performance between GPU and CPU."""
        if not mx.metal.is_available():
            pytest.skip("Metal GPU not available")
            
        batch_size, seq_len, head_dim = 1, 256, 64
        num_heads = 8
        num_kv_heads = 8
        precision = mx.float32
        
        # Create inputs
        q = mx.random.normal((batch_size * num_heads, seq_len, head_dim), dtype=precision)
        k = mx.random.normal((batch_size * num_kv_heads, seq_len, head_dim), dtype=precision)
        v = mx.random.normal((batch_size * num_kv_heads, seq_len, head_dim), dtype=precision)
        mask = mx.triu(mx.full((batch_size, seq_len, seq_len), -mx.inf), k=1).astype(precision)
        scale = 1.0 / math.sqrt(head_dim)
        
        # Time GPU version
        with mx.gpu:
            q_gpu = q.astype(precision)
            k_gpu = k.astype(precision)
            v_gpu = v.astype(precision)
            mask_gpu = mask.astype(precision)
            
            start_time = time.time()
            gpu_output = flash_attention(q_gpu, k_gpu, v_gpu, mask_gpu, scale, num_kv_heads, num_heads, mx.gpu)
            mx.eval(gpu_output)  # Force evaluation
            gpu_time = time.time() - start_time
        
        # Time CPU version
        with mx.cpu:
            q_cpu = q.astype(precision)
            k_cpu = k.astype(precision)
            v_cpu = v.astype(precision)
            mask_cpu = mask.astype(precision)
            
            start_time = time.time()
            cpu_output = flash_attention(q_cpu, k_cpu, v_cpu, mask_cpu, scale, num_kv_heads, num_heads, mx.cpu)
            mx.eval(cpu_output)  # Force evaluation
            cpu_time = time.time() - start_time
        
        # Verify outputs are close
        assert mx.allclose(gpu_output, cpu_output, atol=1e-3, rtol=1e-3)
        
        # Both should complete in reasonable time
        assert gpu_time < 20.0
        assert cpu_time < 20.0
        
        print(f"GPU time: {gpu_time:.4f}s, CPU time: {cpu_time:.4f}s")
    
    @pytest.mark.skipif(not FLASH_ATTENTION_AVAILABLE, reason="Flash Attention extension not available")
    def test_flash_attention_gpu_memory_management(self):
        """Test GPU memory management with Flash Attention."""
        if not mx.metal.is_available():
            pytest.skip("Metal GPU not available")
            
        # Test with multiple sequence lengths to stress memory management
        sequence_lengths = [128, 256, 512]
        batch_size = 2
        head_dim = 64
        num_heads = 4
        num_kv_heads = 4
        precision = mx.float32
        
        with mx.gpu:
            for seq_len in sequence_lengths:
                q = mx.random.normal((batch_size * num_heads, seq_len, head_dim), dtype=precision)
                k = mx.random.normal((batch_size * num_kv_heads, seq_len, head_dim), dtype=precision)
                v = mx.random.normal((batch_size * num_kv_heads, seq_len, head_dim), dtype=precision)
                mask = mx.triu(mx.full((batch_size, seq_len, seq_len), -mx.inf), k=1).astype(precision)
                
                scale = 1.0 / math.sqrt(head_dim)
                output = flash_attention(q, k, v, mask, scale, num_kv_heads, num_heads, mx.gpu)
                
                assert output.shape == (batch_size * num_heads, seq_len, head_dim)
                assert not mx.isnan(output).any()
                assert output.device() == mx.gpu
                
                # Force cleanup
                del q, k, v, mask, output
    
    @pytest.mark.skipif(not FLASH_ATTENTION_AVAILABLE, reason="Flash Attention extension not available")
    def test_flash_attention_gpu_large_batch(self):
        """Test Flash Attention with large batch sizes on GPU."""
        if not mx.metal.is_available():
            pytest.skip("Metal GPU not available")
            
        # Test larger batch size that benefits from GPU parallelism
        batch_size = 8
        seq_len = 128
        head_dim = 64
        num_heads = 8
        num_kv_heads = 8
        precision = mx.float32
        
        with mx.gpu:
            q = mx.random.normal((batch_size * num_heads, seq_len, head_dim), dtype=precision)
            k = mx.random.normal((batch_size * num_kv_heads, seq_len, head_dim), dtype=precision)
            v = mx.random.normal((batch_size * num_kv_heads, seq_len, head_dim), dtype=precision)
            mask = mx.triu(mx.full((batch_size, seq_len, seq_len), -mx.inf), k=1).astype(precision)
            
            scale = 1.0 / math.sqrt(head_dim)
            output = flash_attention(q, k, v, mask, scale, num_kv_heads, num_heads, mx.gpu)
            
            assert output.shape == (batch_size * num_heads, seq_len, head_dim)
            assert not mx.isnan(output).any()
            assert output.device() == mx.gpu
    
    @pytest.mark.skipif(not FLASH_ATTENTION_AVAILABLE, reason="Flash Attention extension not available")
    def test_flash_attention_gpu_grouped_query(self):
        """Test Flash Attention GPU with grouped query attention."""
        if not mx.metal.is_available():
            pytest.skip("Metal GPU not available")
            
        batch_size = 2
        seq_len = 64
        head_dim = 32
        num_heads = 8
        num_kv_heads = 2  # Grouped query attention (4:1 ratio)
        precision = mx.float32
        
        with mx.gpu:
            q = mx.random.normal((batch_size * num_heads, seq_len, head_dim), dtype=precision)
            k = mx.random.normal((batch_size * num_kv_heads, seq_len, head_dim), dtype=precision)
            v = mx.random.normal((batch_size * num_kv_heads, seq_len, head_dim), dtype=precision)
            mask = mx.triu(mx.full((batch_size, seq_len, seq_len), -mx.inf), k=1).astype(precision)
            
            scale = 1.0 / math.sqrt(head_dim)
            output = flash_attention(q, k, v, mask, scale, num_kv_heads, num_heads, mx.gpu)
            
            assert output.shape == (batch_size * num_heads, seq_len, head_dim)
            assert not mx.isnan(output).any()
            assert output.device() == mx.gpu
    
    @pytest.mark.skipif(not FLASH_ATTENTION_AVAILABLE, reason="Flash Attention extension not available")
    def test_flash_attention_gpu_device_consistency(self):
        """Test that Flash Attention maintains device consistency."""
        if not mx.metal.is_available():
            pytest.skip("Metal GPU not available")
            
        batch_size = 1
        seq_len = 32
        head_dim = 16
        num_heads = 2
        num_kv_heads = 2
        precision = mx.float32
        
        # Test GPU computation
        with mx.gpu:
            q = mx.random.normal((batch_size * num_heads, seq_len, head_dim), dtype=precision)
            k = mx.random.normal((batch_size * num_kv_heads, seq_len, head_dim), dtype=precision)
            v = mx.random.normal((batch_size * num_kv_heads, seq_len, head_dim), dtype=precision)
            mask = mx.triu(mx.full((batch_size, seq_len, seq_len), -mx.inf), k=1).astype(precision)
            
            # Ensure all inputs are on GPU
            assert q.device() == mx.gpu
            assert k.device() == mx.gpu
            assert v.device() == mx.gpu
            assert mask.device() == mx.gpu
            
            scale = 1.0 / math.sqrt(head_dim)
            output = flash_attention(q, k, v, mask, scale, num_kv_heads, num_heads, mx.gpu)
            
            # Output should remain on GPU
            assert output.device() == mx.gpu
            assert output.shape == (batch_size * num_heads, seq_len, head_dim)
            assert not mx.isnan(output).any()
    
    @pytest.mark.skipif(not FLASH_ATTENTION_AVAILABLE, reason="Flash Attention extension not available")
    def test_flash_attention_gpu_numerical_stability_large_scale(self):
        """Test Flash Attention GPU numerical stability with large scale factors."""
        if not mx.metal.is_available():
            pytest.skip("Metal GPU not available")
            
        batch_size = 1
        seq_len = 64
        head_dim = 32
        num_heads = 4
        num_kv_heads = 4
        precision = mx.float32
        
        with mx.gpu:
            # Create inputs with larger magnitudes to test numerical stability
            q = mx.random.normal((batch_size * num_heads, seq_len, head_dim), dtype=precision) * 2.0
            k = mx.random.normal((batch_size * num_kv_heads, seq_len, head_dim), dtype=precision) * 2.0
            v = mx.random.normal((batch_size * num_kv_heads, seq_len, head_dim), dtype=precision)
            mask = mx.triu(mx.full((batch_size, seq_len, seq_len), -mx.inf), k=1).astype(precision)
            
            scale = 1.0 / math.sqrt(head_dim)
            output = flash_attention(q, k, v, mask, scale, num_kv_heads, num_heads, mx.gpu)
            
            # Should handle numerical challenges gracefully
            assert not mx.isnan(output).any()
            assert not mx.isinf(output).any()
            assert mx.all(mx.isfinite(output))
            assert output.device() == mx.gpu


class TestFlashAttentionGPUIntegration:
    """Integration tests for Flash Attention GPU with transformer components."""
    
    @pytest.mark.skipif(not FLASH_ATTENTION_AVAILABLE, reason="Flash Attention extension not available")
    def test_flash_attention_gpu_transformer_integration(self):
        """Test Flash Attention GPU integration with transformer-like patterns."""
        if not mx.metal.is_available():
            pytest.skip("Metal GPU not available")
            
        # Simulate transformer attention computation pattern
        batch_size = 2
        seq_len = 128
        model_dim = 512
        num_heads = 8
        head_dim = model_dim // num_heads
        num_kv_heads = num_heads  # Multi-head attention
        precision = mx.float32
        
        with mx.gpu:
            # Simulate transformer input processing
            x = mx.random.normal((batch_size, seq_len, model_dim), dtype=precision)
            
            # Simulate Q, K, V projections (reshape to extension format)
            q = x.reshape(batch_size * num_heads, seq_len, head_dim)
            k = x.reshape(batch_size * num_kv_heads, seq_len, head_dim)
            v = x.reshape(batch_size * num_kv_heads, seq_len, head_dim)
            
            # Create causal mask for transformer
            mask = mx.triu(mx.full((batch_size, seq_len, seq_len), -mx.inf), k=1).astype(precision)
            
            scale = 1.0 / math.sqrt(head_dim)
            attention_output = flash_attention(q, k, v, mask, scale, num_kv_heads, num_heads, mx.gpu)
            
            # Reshape back to transformer format
            output = attention_output.reshape(batch_size, seq_len, model_dim)
            
            assert output.shape == (batch_size, seq_len, model_dim)
            assert not mx.isnan(output).any()
            assert output.device() == mx.gpu
            
            # Test reasonable output statistics for transformer use
            output_std = mx.std(output)
            assert 0.1 < output_std < 10.0  # Reasonable variance for transformer layers
    
    @pytest.mark.skipif(not FLASH_ATTENTION_AVAILABLE, reason="Flash Attention extension not available")
    def test_flash_attention_gpu_inference_pattern(self):
        """Test Flash Attention GPU with inference-like usage patterns."""
        if not mx.metal.is_available():
            pytest.skip("Metal GPU not available")
            
        # Simulate inference with varying sequence lengths
        head_dim = 64
        num_heads = 8
        num_kv_heads = 8
        precision = mx.float32
        
        # Different sequence lengths as in inference
        sequence_lengths = [1, 16, 64, 128]  # From single token to longer sequences
        
        with mx.gpu:
            for seq_len in sequence_lengths:
                batch_size = 1  # Typical inference batch size
                
                q = mx.random.normal((batch_size * num_heads, seq_len, head_dim), dtype=precision)
                k = mx.random.normal((batch_size * num_kv_heads, seq_len, head_dim), dtype=precision)
                v = mx.random.normal((batch_size * num_kv_heads, seq_len, head_dim), dtype=precision)
                mask = mx.triu(mx.full((batch_size, seq_len, seq_len), -mx.inf), k=1).astype(precision)
                
                scale = 1.0 / math.sqrt(head_dim)
                
                # Time inference computation
                start_time = time.time()
                output = flash_attention(q, k, v, mask, scale, num_kv_heads, num_heads, mx.gpu)
                mx.eval(output)  # Force evaluation
                inference_time = time.time() - start_time
                
                assert output.shape == (batch_size * num_heads, seq_len, head_dim)
                assert not mx.isnan(output).any()
                
                # Inference should be fast, especially for shorter sequences
                if seq_len <= 64:
                    assert inference_time < 5.0
                else:
                    assert inference_time < 15.0
                
                print(f"Seq len {seq_len}: {inference_time:.4f}s")


if __name__ == "__main__":
    pytest.main([__file__])
