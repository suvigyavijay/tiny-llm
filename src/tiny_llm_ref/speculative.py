import mlx.core as mx


def speculative_decode(target_model, draft_model, prompt: mx.array, k: int = 4):
    """
    Speculative decoding with greedy acceptance.
    """
    draft_tokens = []
    current_tokens = prompt
    
    # 1. Draft phase: generate k tokens with small model
    for _ in range(k):
        logits = draft_model(current_tokens)
        next_token = mx.argmax(logits[:, -1, :], axis=-1)
        draft_tokens.append(next_token)
        current_tokens = mx.concatenate([current_tokens, next_token[:, None]], axis=1)
        
    draft_tokens = mx.stack(draft_tokens, axis=1)  # [1, K]
    
    # 2. Verify phase: run target model on full sequence
    full_input = mx.concatenate([prompt, draft_tokens], axis=1)
    target_logits = target_model(full_input)  # [1, L+K, V]
    
    # 3. Acceptance: compare target predictions with draft tokens
    start_pos = prompt.shape[1] - 1
    accepted_count = 0
    
    for i in range(k):
        target_prediction = mx.argmax(target_logits[:, start_pos + i, :], axis=-1)
        if target_prediction.item() == draft_tokens[0, i].item():
            accepted_count += 1
        else:
            # Rejection: return accepted tokens + correction
            accepted = draft_tokens[0, :accepted_count] if accepted_count > 0 else mx.array([])
            return accepted, target_prediction
            
    # All accepted: return draft + next token from target
    next_token = mx.argmax(target_logits[:, -1, :], axis=-1)
    return draft_tokens[0], next_token
