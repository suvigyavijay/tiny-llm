import pytest
from .utils import *
from extensions_ref import tiny_llm_ext_ref
from tiny_llm_ref.attention import scaled_dot_product_attention_simple as scaled_dot_product_attention
import mlx.nn as nn


def flash_attention_ref(q, k, v, mask, scale):
    B, H_q, L, D = q.shape
    B, H_kv, L, D = k.shape

    if H_q != H_kv:
        k = mx.repeat(k, H_q // H_kv, axis=1)
        v = mx.repeat(v, H_q // H_kv, axis=1)

    # The reference implementation uses the standard scaled dot product attention
    return scaled_dot_product_attention(q, k, v, mask=mask, scale=scale)


@pytest.mark.parametrize("stream", AVAILABLE_STREAMS, ids=AVAILABLE_STREAMS_IDS)
@pytest.mark.parametrize("precision", [mx.float16, mx.float32], ids=["f16", "f32"])
@pytest.mark.parametrize("B, H, L, D", [(1, 8, 256, 64), (2, 4, 512, 32)])
def test_flash_attention(stream, precision, B, H, L, D):
    with mx.stream(stream):
        q = mx.random.normal(shape=(B, H, L, D)).astype(precision)
        k = mx.random.normal(shape=(B, H, L, D)).astype(precision)
        v = mx.random.normal(shape=(B, H, L, D)).astype(precision)
        mask = mx.zeros((B, H, L, L), dtype=precision)
        scale = 1.0 / np.sqrt(D)

        # Run the Flash Attention implementation
        flash_out = tiny_llm_ext_ref.flash_attention(
            q.reshape(B * H, L, D).astype(mx.float32),
            k.reshape(B * H, L, D).astype(mx.float32),
            v.reshape(B * H, L, D).astype(mx.float32),
            mask.reshape(B * H, L, L).astype(mx.float32),
            scale, H, H
        ).reshape(B, H, L, D)

        # Run the reference implementation
        ref_out = flash_attention_ref(q, k, v, mask, scale)
        
        assert_allclose(flash_out, ref_out, atol=1e-1, rtol=1e-1, precision=precision)


@pytest.mark.parametrize("stream", AVAILABLE_STREAMS, ids=AVAILABLE_STREAMS_IDS)
@pytest.mark.parametrize("precision", [mx.float16, mx.float32], ids=["f16", "f32"])
@pytest.mark.parametrize("B, H_q, H_kv, L, D", [(1, 8, 4, 256, 64), (2, 4, 2, 512, 32)])
def test_flash_attention_gqa(stream, precision, B, H_q, H_kv, L, D):
    with mx.stream(stream):
        q = mx.random.normal(shape=(B, H_q, L, D)).astype(precision)
        k = mx.random.normal(shape=(B, H_kv, L, D)).astype(precision)
        v = mx.random.normal(shape=(B, H_kv, L, D)).astype(precision)
        mask = mx.zeros((B, H_q, L, L), dtype=precision)
        scale = 1.0 / np.sqrt(D)

        # Run the Flash Attention implementation
        flash_out = tiny_llm_ext_ref.flash_attention(
            q.reshape(B * H_q, L, D).astype(mx.float32),
            k.reshape(B * H_kv, L, D).astype(mx.float32),
            v.reshape(B * H_kv, L, D).astype(mx.float32),
            mask.reshape(B * H_q, L, L).astype(mx.float32),
            scale, H_kv, H_q
        ).reshape(B, H_q, L, D)

        # Run the reference implementation
        ref_out = flash_attention_ref(q, k, v, mask, scale)
        
        assert_allclose(flash_out, ref_out, atol=1e-1, rtol=1e-1, precision=precision)


@pytest.mark.parametrize("stream", AVAILABLE_STREAMS, ids=AVAILABLE_STREAMS_IDS)
@pytest.mark.parametrize("precision", [mx.float16, mx.float32], ids=["f16", "f32"])
@pytest.mark.parametrize("B, H, L, D", [(1, 8, 256, 64)])
def test_flash_attention_causal_mask(stream, precision, B, H, L, D):
    with mx.stream(stream):
        q = mx.random.normal(shape=(B, H, L, D)).astype(precision)
        k = mx.random.normal(shape=(B, H, L, D)).astype(precision)
        v = mx.random.normal(shape=(B, H, L, D)).astype(precision)
        mask = nn.MultiHeadAttention.create_additive_causal_mask(L).astype(precision)
        scale = 1.0 / np.sqrt(D)

        # Run the Flash Attention implementation
        flash_out = tiny_llm_ext_ref.flash_attention(
            q.reshape(B * H, L, D).astype(mx.float32),
            k.reshape(B * H, L, D).astype(mx.float32),
            v.reshape(B * H, L, D).astype(mx.float32),
            mx.broadcast_to(mask, (B * H, L, L)).astype(mx.float32),
            scale, H, H
        ).reshape(B, H, L, D)

        # Run the reference implementation
        ref_out = flash_attention_ref(q, k, v, mask, scale)

        assert_allclose(flash_out, ref_out, atol=1e-1, rtol=1e-1, precision=precision)
