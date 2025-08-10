import mlx.core as mx
from typing import Any


def dequantize_linear(mx_layer: Any) -> mx.array:
    w = mx.dequantize(
        mx_layer.weight,
        mx_layer.scales,
        mx_layer.biases,
        mx_layer.group_size,
        mx_layer.bits,
    )
    return w


class QuantizedWeights:
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
        return QuantizedWeights(
            scales=mlx_layer.scales,
            biases=mlx_layer.biases,
            group_size=mlx_layer.group_size,
            bits=mlx_layer.bits,
            weight=mlx_layer.weight,
        )


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
    Perform quantized matrix multiplication.
    
    TODO: Implement quantized matrix multiplication
    - This should call the C++/Metal extension for efficient computation
    - Use the tiny_llm_ext.quantized_matmul function from the extensions
    - Handle the tensor reshaping appropriately
    
    Args:
        scales: Per-group scale factors
        biases: Per-group bias terms
        group_size: Number of weights per quantization group
        bits: Number of bits for quantization (typically 4)
        a: Input tensor 
        b: Quantized weight tensor
        transpose_b: Whether to transpose b before multiplication
        
    Returns:
        Result of quantized matrix multiplication
    """
    pass


def quantized_linear(
    x: mx.array,
    w: QuantizedWeights,
    bias: mx.array | None = None,
) -> mx.array:
    """
    Perform quantized linear transformation.
    
    TODO: Implement quantized linear layer
    - Use quantized_matmul with the QuantizedWeights parameters
    - Add bias if provided
    - Handle proper tensor shapes for batched inputs
    
    Args:
        x: Input tensor [batch_size, seq_len, hidden_size]
        w: Quantized weights containing scales, biases, and quantized weight data
        bias: Optional bias term
        
    Returns:
        Output of linear transformation
    """
    pass
