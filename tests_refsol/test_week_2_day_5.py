import pytest
import mlx.core as mx
from tiny_llm_ref.attention import flash_attention
import numpy as np

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)

def flash_attention_ref_py(q, k, v, scale):
    q_np = np.array(q, copy=False)
    k_np = np.array(k, copy=False)
    v_np = np.array(v, copy=False)

    B, H_q, L, E = q_np.shape
    B, H_k, S, E = k_np.shape
    
    if H_q != H_k:
        # Grouped Query Attention
        num_groups = H_q // H_k
        k_np = np.repeat(k_np, num_groups, axis=1)
        v_np = np.repeat(v_np, num_groups, axis=1)

    scores = (q_np @ k_np.transpose(0, 1, 3, 2)) * scale
    scores = softmax(scores)
    output = scores @ v_np
    return mx.array(output)

@pytest.mark.parametrize("H_q", [4, 8])
@pytest.mark.parametrize("H_k", [1, 2, 4])
@pytest.mark.parametrize("L", [16, 32])
@pytest.mark.parametrize("E", [32, 64])
@pytest.mark.parametrize("S", [16, 32])
@pytest.mark.parametrize("BATCH", [1, 2])
def test_flash_attention_gpu(H_q, H_k, L, E, S, BATCH):
    if H_q % H_k != 0:
        pytest.skip("H_q must be divisible by H_k")

    precision = mx.float32
    q_shape = (BATCH, H_q, L, E)
    kv_shape = (BATCH, H_k, S, E)
    scale = 0.9

    query = mx.random.uniform(shape=q_shape, dtype=precision)
    key = mx.random.uniform(shape=kv_shape, dtype=precision)
    value = mx.random.uniform(shape=kv_shape, dtype=precision)

    reference_output = flash_attention_ref_py(
        query,
        key,
        value,
        scale=scale,
    )
    user_output = flash_attention(
        query,
        key,
        value,
        scale=scale,
    )
    assert np.allclose(np.array(user_output), np.array(reference_output), atol=1e-5)
