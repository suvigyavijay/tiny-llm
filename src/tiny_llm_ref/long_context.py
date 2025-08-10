import mlx.core as mx

def sliding_window_attention(q, k, v, window_size: int):
    """
    A simplified implementation of sliding window attention.
    """
    
    B, seq_len, D = q.shape
    output = mx.zeros_like(q)
    
    for i in range(seq_len):
        start = max(0, i - window_size + 1)
        end = i + 1
        
        # This is a highly simplified implementation. A real implementation
        # would use more efficient methods for slicing and computation.
        
        context_k = k[:, start:end, :]
        context_v = v[:, start:end, :]
        
        attention_scores = mx.matmul(q[:, i:i+1, :], context_k.transpose(0, 2, 1)) / mx.sqrt(D)
        
        # Apply causal mask
        mask = mx.full(attention_scores.shape, -1e9)
        mask[:, :, -(i-start+1):] = 0
        attention_scores = attention_scores + mask
        
        attention_weights = mx.softmax(attention_scores, axis=-1)
        
        output[:, i:i+1, :] = mx.matmul(attention_weights, context_v)
        
    return output
