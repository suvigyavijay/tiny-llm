import pytest
import mlx.core as mx
from .tiny_llm_base import *
from .utils import *


@pytest.mark.parametrize("stream", AVAILABLE_STREAMS, ids=AVAILABLE_STREAMS_IDS)
@pytest.mark.parametrize("M", [1, 8, 17])
@pytest.mark.parametrize("N", [32, 64, 128])
@pytest.mark.parametrize("K", [32, 64, 128])
@pytest.mark.parametrize("group_size", [32, 64])
@pytest.mark.parametrize("bits", [2, 4, 8])
def test_quantized_matmul(stream, M, N, K, group_size, bits):
    with mx.stream(stream):
        if N % group_size != 0:
            pytest.skip("N must be divisible by group_size")

        x = mx.random.normal(shape=(M, N)).astype(mx.float16)
        w = mx.random.normal(shape=(K, N)).astype(mx.float16)
        w_q, scales, biases = mx.quantize(w, group_size=group_size, bits=bits)

        user_out = quantized_matmul(
            a=x,
            b=w_q,
            scales=scales,
            biases=biases,
            group_size=group_size,
            bits=bits,
            transpose_b=True,
        )

        ref_out = mx.quantized_matmul(
            x,
            w_q,
            scales=scales,
            biases=biases,
            group_size=group_size,
            bits=bits,
            transpose=True,
        )
        assert_allclose(user_out, ref_out, atol=1e-1, rtol=1e-1, precision=mx.float16)


@pytest.mark.parametrize("stream", AVAILABLE_STREAMS, ids=AVAILABLE_STREAMS_IDS)
@pytest.mark.parametrize("M", [1, 8, 17])
@pytest.mark.parametrize("N", [32, 64, 128])
@pytest.mark.parametrize("K", [32, 64, 128])
@pytest.mark.parametrize("group_size", [32, 64])
@pytest.mark.parametrize("bits", [2, 4, 8])
def test_quantized_linear(stream, M, N, K, group_size, bits):
    with mx.stream(stream):
        if N % group_size != 0:
            pytest.skip("N must be divisible by group_size")

        x = mx.random.normal(shape=(M, N)).astype(mx.float16)
        weight = mx.random.normal(shape=(K, N)).astype(mx.float16)
        bias = mx.random.normal(shape=(K,)).astype(mx.float16)
        w_q, scales, biases = mx.quantize(weight, group_size=group_size, bits=bits)
        
        w = QuantizedWeights(scales, biases, group_size, bits, w_q)
        
        user_out = quantized_linear(x, w, bias)
        ref_out = mx.quantized_matmul(
            x,
            w_q,
            scales,
            biases,
            group_size=group_size,
            bits=bits,
            transpose=True,
        ) + bias
        
        assert_allclose(user_out, ref_out, atol=1e-1, rtol=1e-1, precision=mx.float16)
