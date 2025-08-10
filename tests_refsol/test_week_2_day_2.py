import pytest
import mlx.core as mx
from .tiny_llm_base import *
from .utils import *


def quantized_matmul_helper(
    stream: mx.Stream, identity_matrix: bool, precision: mx.Dtype
):
    shapes = [(3, 64, 5), (10, 128, 32), (1, 256, 128)]
    for M, N, K in shapes:
        with mx.stream(stream):
            if identity_matrix:
                input = mx.eye(N, dtype=precision)
            else:
                input = mx.random.normal(shape=(M, N), dtype=precision)
            weight = mx.random.normal(shape=(K, N), dtype=precision)
            w_q, scales, biases = mx.quantize(weight)
            user_out = quantized_matmul(
                scales=scales,
                biases=biases,
                group_size=64,
                bits=4,
                a=input,
                b=w_q,
                transpose_b=True,
            )
            ref_out = mx.quantized_matmul(
                input,
                w_q,
                scales,
                biases,
                group_size=64,
                bits=4,
                transpose=True,
            )
            assert_allclose(user_out, ref_out, precision, rtol=1e-1)


def test_task_1_quantized_linear_cpu():
    input = mx.random.normal(shape=(3, 64), dtype=mx.float16)
    weight = mx.random.normal(shape=(5, 64), dtype=mx.float16)
    bias = mx.random.normal(shape=(5,), dtype=mx.float16)
    w_q, scales, biases = mx.quantize(weight)
    
    w = QuantizedWeights(scales, biases, 64, 4, w_q)
    
    user_out = quantized_linear(input, w, bias)
    ref_out = mx.quantized_matmul(input, w_q, scales, biases, group_size=64, bits=4, transpose=True) + bias
    
    assert_allclose(user_out, ref_out, atol=1e-1, rtol=1e-1, precision=mx.float16)


def test_task_1_quantized_matmul_simple_f16_cpu():
    quantized_matmul_helper(mx.cpu, True, mx.float16)


def test_task_1_quantized_matmul_complex_f16_cpu():
    quantized_matmul_helper(mx.cpu, False, mx.float16)


def test_task_2_quantized_matmul_simple_f16_gpu():
    quantized_matmul_helper(mx.gpu, True, mx.float16)


def test_task_2_quantized_matmul_complex_f16_gpu():
    quantized_matmul_helper(mx.gpu, False, mx.float16)
