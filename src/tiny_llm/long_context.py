import mlx.core as mx


def apply_linear_scaling_rope(freqs: mx.array, scale_factor: float = 1.0) -> mx.array:
    """
    Scale the RoPE frequencies for longer context windows.
    
    Args:
        freqs: Original RoPE frequencies (theta values).
        scale_factor: The ratio L_new / L_train (e.g., 2.0 for doubling context).
        
    Returns:
        Scaled frequencies.
    """
    pass
