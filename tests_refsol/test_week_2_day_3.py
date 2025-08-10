import pytest
import mlx.core as mx
import numpy as np

def dequantize_ref_py(w_q, scales, biases, group_size, bits):
    w_q_np = np.array(w_q, copy=False)
    scales_np = np.array(scales, copy=False)
    biases_np = np.array(biases, copy=False)

    pack_per_int = 32 // bits
    mask = (1 << bits) - 1

    orig_cols = w_q_np.shape[1] * pack_per_int
    w_unpacked = np.zeros((w_q_np.shape[0], orig_cols), dtype=np.uint32)

    for r in range(w_q_np.shape[0]):
        for c in range(w_q_np.shape[1]):
            val = w_q_np[r, c]
            for i in range(pack_per_int):
                w_unpacked[r, c * pack_per_int + i] = (val >> (i * bits)) & mask

    w_unpacked = w_unpacked.astype(scales_np.dtype)

    scales_reshaped = np.repeat(scales_np, group_size, axis=1)
    biases_reshaped = np.repeat(biases_np, group_size, axis=1)

    dequantized = (w_unpacked * scales_reshaped) + biases_reshaped
    return dequantized


def quantized_matmul_ref_py(x, w_q, scales, biases, group_size, bits, transpose):
    x_np = np.array(x, copy=False)
    w_dq_np = dequantize_ref_py(w_q, scales, biases, group_size, bits)
    if transpose:
        w_dq_np = w_dq_np.T

    result = x_np @ w_dq_np
    return mx.array(result)

@pytest.mark.parametrize("M", [1, 8, 17])
@pytest.mark.parametrize("N", [32, 64, 128])
@pytest.mark.parametrize("K", [32, 64, 128])
@pytest.mark.parametrize("group_size", [32, 64])
@pytest.mark.parametrize("bits", [2, 4, 8])
def test_quantized_matmul_gpu(M, N, K, group_size, bits):
    if N % group_size != 0:
        pytest.skip("N must be divisible by group_size")

    x = mx.random.normal(shape=(M, N)).astype(mx.float16)
    w = mx.random.normal(shape=(K, N)).astype(mx.float16)
    w_q, scales, biases = mx.quantize(w, group_size=group_size, bits=bits)

    out_ref = quantized_matmul_ref_py(x, w_q, scales, biases, group_size, bits, transpose=True)

    out = mx.quantized_matmul(
        x,
        w_q,
        scales=scales,
        biases=biases,
        group_size=group_size,
        bits=bits,
        transpose=True,
    )

    assert out.shape == out_ref.shape
    assert np.allclose(np.array(out), np.array(out_ref), atol=1e-1)
