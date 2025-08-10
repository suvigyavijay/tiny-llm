import pytest
from .utils import *
from extensions_ref import tiny_llm_ext_ref
from tiny_llm_ref.attention import scaled_dot_product_attention_simple as scaled_dot_product_attention

def flash_attention_ref(q, k, v, mask, scale):
    if k.shape[0] != q.shape[0]:
        k = mx.repeat(k, q.shape[0] // k.shape[0], axis=0)
    if v.shape[0] != q.shape[0]:
        v = mx.repeat(v, q.shape[0] // v.shape[0], axis=0)
    # The reference implementation uses the standard scaled dot product attention
    return scaled_dot_product_attention(q, k, v, mask=mask, scale=scale)

@pytest.mark.skipif(
    not mx.metal.is_available(), reason="Metal is not available"
)
class TestFlashAttentionGPU:
    def test_flash_attention_gpu(self):
        q = mx.random.uniform(shape=(8, 128, 64)).astype(mx.float32)
        k = mx.random.uniform(shape=(8, 128, 64)).astype(mx.float32)
        v = mx.random.uniform(shape=(8, 128, 64)).astype(mx.float32)
        mask = mx.zeros((8, 128, 128), dtype=mx.float32)
        scale = 1.0 / np.sqrt(64)

        # Run the Flash Attention implementation
        with mx.stream(mx.gpu):
            flash_out = tiny_llm_ext_ref.flash_attention(q, k, v, mask, scale, 8, 8)

        # Run the reference implementation
        ref_out = flash_attention_ref(q, k, v, mask, scale)
        
        assert_allclose(flash_out, ref_out, atol=1e-5, rtol=1e-5, precision=mx.float32)

    def test_flash_attention_gpu_gqa(self):
        q = mx.random.uniform(shape=(8, 128, 64)).astype(mx.float32)
        k = mx.random.uniform(shape=(4, 128, 64)).astype(mx.float32)
        v = mx.random.uniform(shape=(4, 128, 64)).astype(mx.float32)
        mask = mx.zeros((8, 128, 128), dtype=mx.float32)
        scale = 1.0 / np.sqrt(64)

        # Run the Flash Attention implementation
        with mx.stream(mx.gpu):
            flash_out = tiny_llm_ext_ref.flash_attention(q, k, v, mask, scale, 4, 8)

        # Run the reference implementation
        ref_out = flash_attention_ref(q, k, v, mask, scale)
        
        assert_allclose(flash_out, ref_out, atol=1e-5, rtol=1e-5, precision=mx.float32)
