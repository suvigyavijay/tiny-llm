import mlx.core as mx
from typing import Any
from extensions_ref import tiny_llm_ext_ref


class QuantizedWeights:
    """
    A container for quantized weights and their metadata.
    """
    def __init__(
        self,
        scales: mx.array,
        biases: mx.array,
        group_size: int,
        bits: int,
        weight: mx.array,
    ):
        self.scales = scales
        self.biases = biases
        self.group_size = group_size
        self.bits = bits
        self.weight = weight

    @staticmethod
    def from_mlx_layer(mlx_layer: Any) -> "QuantizedWeights":
        """
        Create a QuantizedWeights object from an mlx quantized layer.
        """
        return QuantizedWeights(
            scales=mlx_layer.scales,
            biases=mlx_layer.biases,
            group_size=mlx_layer.group_size,
            bits=mlx_layer.bits,
            weight=mlx_layer.weight,
        )


def quantized_linear(
    x: mx.array,
    w: QuantizedWeights,
    bias: mx.array | None = None,
) -> mx.array:
    """
    Performs a quantized linear operation.
    This is equivalent to `x @ w.T + bias`, but with a quantized weight matrix.
    """
    if bias is not None:
        return (
            quantized_matmul(
                w.scales, w.biases, w.group_size, w.bits, x, w.weight, True
            )
            + bias
        )
    else:
        return quantized_matmul(
            w.scales, w.biases, w.group_size, w.bits, x, w.weight, True
        )


def dequantize_linear(mx_layer: Any) -> mx.array:
    """
    Dequantize the weights of an mlx quantized layer.
    """
    w = mx.dequantize(
        mx_layer.weight,
        mx_layer.scales,
        mx_layer.biases,
        mx_layer.group_size,
        mx_layer.bits,
    )
    return w


def quantized_matmul(
    scales: mx.array,
    biases: mx.array,
    group_size: int,
    bits: int,
    a: mx.array,
    b: mx.array,
    transpose_b: bool = False,
) -> mx.array:
    """
    Performs quantized matrix multiplication.
    """
    *N, D = a.shape
    # The C++ kernel expects a 2D matrix, so we reshape the input.
    a = a.reshape(-1, D)
    # The C++ kernel expects contiguous arrays, so we ensure they are contiguous.
    a = mx.contiguous(a)
    b = mx.contiguous(b)
    return tiny_llm_ext_ref.quantized_matmul(
        scales, biases, group_size, bits, a, b, transpose_b
    ).reshape(*N, -1)
