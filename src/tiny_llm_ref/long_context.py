import mlx.core as mx


def apply_linear_scaling_rope(freqs: mx.array, scale_factor: float = 1.0) -> mx.array:
    """
    Scale the RoPE frequencies for longer context windows.
    
    The idea: if we want position 4096 to act like position 2048 (scale_factor=2),
    we need to slow down the rotation speed by dividing frequencies.
    
    RoPE uses: cos(theta * t), sin(theta * t)
    Scaling t by 1/s is equivalent to scaling theta by 1/s:
        cos(theta * (t/s)) = cos((theta/s) * t)
    """
    return freqs / scale_factor
