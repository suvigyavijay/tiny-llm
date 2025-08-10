"""
Speculative Decoding implementation for accelerated LLM inference.

Implements speculative decoding that uses a smaller draft model to propose tokens
which are then verified in parallel by the target model, achieving significant speedups.
"""

import mlx.core as mx
from typing import List, Tuple, Optional, Any
import math
import time


class SpeculativeDecoder:
    """Main speculative decoding system coordinating draft and target models."""
    
    def __init__(self, 
                 draft_model: Any,      # Fast, smaller model
                 target_model: Any,     # Accurate, larger model  
                 lookahead: int = 4,
                 acceptance_threshold: float = 0.8):
        """
        Initialize speculative decoder.
        
        Args:
            draft_model: Fast model for generating proposals
            target_model: Accurate model for verification
            lookahead: Number of tokens to speculatively generate
            acceptance_threshold: Threshold for accepting speculative tokens
        """
        self.draft_model = draft_model
        self.target_model = target_model
        self.lookahead = lookahead
        self.acceptance_threshold = acceptance_threshold
        
        # Statistics tracking
        self.total_draft_tokens = 0
        self.total_accepted_tokens = 0
        self.verification_calls = 0
        
    def draft_generation(self, prompt_tokens: mx.array, num_tokens: int, 
                        temperature: float = 1.0) -> List[int]:
        """
        Generate draft tokens using the fast model.
        
        Args:
            prompt_tokens: Initial prompt tokens
            num_tokens: Number of tokens to generate
            temperature: Sampling temperature
            
        Returns:
            List of draft token IDs
        """
        current_tokens = prompt_tokens
        draft_tokens = []
        
        for _ in range(num_tokens):
            # Get logits from draft model
            logits = self.draft_model(current_tokens)
            
            # Sample next token
            if temperature == 0.0:
                next_token = mx.argmax(logits[:, -1, :], axis=-1)
            else:
                probs = mx.softmax(logits[:, -1, :] / temperature, axis=-1)
                next_token = mx.random.categorical(mx.log(probs), num_samples=1).squeeze()
            
            draft_tokens.append(int(next_token))
            
            # Append token for next iteration
            current_tokens = mx.concat([current_tokens, next_token.reshape(1, 1)], axis=1)
        
        self.total_draft_tokens += len(draft_tokens)
        return draft_tokens
    
    def verify_speculation(self, base_tokens: mx.array, draft_tokens: List[int],
                          temperature: float = 1.0) -> Tuple[int, List[int]]:
        """
        Verify speculative tokens using the target model.
        
        Args:
            base_tokens: Base sequence before speculation
            draft_tokens: Proposed tokens from draft model
            temperature: Sampling temperature
            
        Returns:
            Tuple of (num_accepted, corrected_tokens)
        """
        if not draft_tokens:
            return 0, []
        
        # Create full sequence with draft tokens
        draft_tensor = mx.array(draft_tokens).reshape(1, -1)
        full_sequence = mx.concat([base_tokens, draft_tensor], axis=1)
        
        # Get target model logits for the full sequence
        target_logits = self.target_model(full_sequence)
        
        # Also get draft model logits for comparison
        draft_logits = self.draft_model(full_sequence)
        
        # Verify each draft token
        accepted_tokens = []
        corrected_tokens = []
        
        for i, draft_token in enumerate(draft_tokens):
            # Position in the sequence (after base tokens)
            pos = base_tokens.shape[1] + i
            
            if pos >= target_logits.shape[1]:
                break
            
            # Get probabilities from both models
            target_probs = mx.softmax(target_logits[:, pos - 1, :] / temperature, axis=-1)
            draft_probs = mx.softmax(draft_logits[:, pos - 1, :] / temperature, axis=-1)
            
            # Acceptance probability: min(1, target_prob / draft_prob)
            target_prob = target_probs[0, draft_token]
            draft_prob = draft_probs[0, draft_token]
            
            acceptance_prob = min(1.0, float(target_prob / max(draft_prob, 1e-10)))
            
            # Accept or reject the token
            if mx.random.uniform() < acceptance_prob:
                accepted_tokens.append(draft_token)
            else:
                # Rejection sampling for correction
                # Adjusted probability: max(0, target_prob - draft_prob) / (1 - draft_prob)
                adjusted_probs = mx.maximum(target_probs - draft_probs, 0)
                adjusted_probs = adjusted_probs / mx.sum(adjusted_probs)
                
                # Sample corrected token
                corrected_token = mx.random.categorical(mx.log(adjusted_probs), num_samples=1).squeeze()
                corrected_tokens.append(int(corrected_token))
                break  # Stop after first rejection
        
        self.total_accepted_tokens += len(accepted_tokens)
        self.verification_calls += 1
        
        return len(accepted_tokens), corrected_tokens
    
    def generate(self, prompt_tokens: mx.array, max_tokens: int = 100,
                temperature: float = 1.0) -> List[int]:
        """
        Generate tokens using speculative decoding.
        
        Args:
            prompt_tokens: Initial prompt tokens [1, seq_len]
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            List of generated token IDs
        """
        current_tokens = prompt_tokens
        generated_tokens = []
        
        while len(generated_tokens) < max_tokens:
            # Generate draft tokens
            remaining_tokens = max_tokens - len(generated_tokens)
            draft_count = min(self.lookahead, remaining_tokens)
            
            draft_tokens = self.draft_generation(current_tokens, draft_count, temperature)
            
            if not draft_tokens:
                break
            
            # Verify draft tokens
            num_accepted, corrected_tokens = self.verify_speculation(
                current_tokens, draft_tokens, temperature
            )
            
            # Add accepted tokens
            if num_accepted > 0:
                accepted = draft_tokens[:num_accepted]
                generated_tokens.extend(accepted)
                
                # Update current tokens with accepted ones
                accepted_tensor = mx.array(accepted).reshape(1, -1)
                current_tokens = mx.concat([current_tokens, accepted_tensor], axis=1)
            
            # Add corrected token if there was a rejection
            if corrected_tokens:
                generated_tokens.extend(corrected_tokens)
                corrected_tensor = mx.array(corrected_tokens).reshape(1, -1)
                current_tokens = mx.concat([current_tokens, corrected_tensor], axis=1)
        
        return generated_tokens[:max_tokens]
    
    def get_acceptance_rate(self) -> float:
        """Get the current acceptance rate of speculative tokens."""
        if self.total_draft_tokens == 0:
            return 0.0
        return self.total_accepted_tokens / self.total_draft_tokens
    
    def get_speedup_estimate(self) -> float:
        """
        Estimate speedup compared to standard decoding.
        
        Returns:
            Estimated speedup factor
        """
        acceptance_rate = self.get_acceptance_rate()
        if acceptance_rate == 0:
            return 1.0
        
        # Speedup = (tokens_per_verification_call) / (1 + draft_overhead)
        # Assuming draft model is ~5x faster than target model
        draft_overhead = 0.2  # Relative cost of draft generation
        avg_tokens_per_call = 1 + acceptance_rate * self.lookahead
        
        return avg_tokens_per_call / (1 + draft_overhead)
    
    def reset_statistics(self):
        """Reset performance tracking statistics."""
        self.total_draft_tokens = 0
        self.total_accepted_tokens = 0
        self.verification_calls = 0


class AdaptiveSpeculator:
    """Adaptive speculation that adjusts lookahead based on acceptance rates."""
    
    def __init__(self, initial_lookahead: int = 4, min_lookahead: int = 1, max_lookahead: int = 8):
        """
        Initialize adaptive speculator.
        
        Args:
            initial_lookahead: Starting lookahead distance
            min_lookahead: Minimum lookahead to use
            max_lookahead: Maximum lookahead to use
        """
        self.current_lookahead = initial_lookahead
        self.min_lookahead = min_lookahead
        self.max_lookahead = max_lookahead
        
        # Track recent acceptance rates
        self.acceptance_history = []
        self.window_size = 10
        
    def update_lookahead(self, acceptance_rate: float):
        """
        Adjust lookahead based on recent acceptance rate.
        
        Args:
            acceptance_rate: Acceptance rate from recent speculation
        """
        self.acceptance_history.append(acceptance_rate)
        
        # Keep only recent history
        if len(self.acceptance_history) > self.window_size:
            self.acceptance_history.pop(0)
        
        # Compute average acceptance rate
        avg_acceptance = sum(self.acceptance_history) / len(self.acceptance_history)
        
        # Adjust lookahead based on acceptance rate
        if avg_acceptance > 0.8:  # High acceptance - can look further ahead
            self.current_lookahead = min(self.current_lookahead + 1, self.max_lookahead)
        elif avg_acceptance < 0.3:  # Low acceptance - reduce lookahead
            self.current_lookahead = max(self.current_lookahead - 1, self.min_lookahead)
        
    def get_lookahead(self) -> int:
        """Get current adaptive lookahead distance."""
        return self.current_lookahead


def batched_speculative_decode(batch_prompts: List[List[int]],
                              draft_model: Any, target_model: Any,
                              max_tokens: int = 100,
                              lookahead: int = 4) -> List[List[int]]:
    """
    Implement batched speculative decoding for multiple sequences.
    
    Args:
        batch_prompts: List of prompt token sequences
        draft_model: Fast draft model
        target_model: Accurate target model
        max_tokens: Maximum tokens to generate per sequence
        lookahead: Speculation distance
        
    Returns:
        List of generated token sequences
    """
    batch_size = len(batch_prompts)
    decoders = [SpeculativeDecoder(draft_model, target_model, lookahead) 
               for _ in range(batch_size)]
    
    results = []
    
    # Process each sequence (could be parallelized further)
    for i, prompt in enumerate(batch_prompts):
        prompt_tensor = mx.array(prompt).reshape(1, -1)
        generated = decoders[i].generate(prompt_tensor, max_tokens)
        results.append(generated)
    
    return results


def benchmark_speculative_decoding(draft_model: Any, target_model: Any,
                                  test_prompts: List[List[int]],
                                  max_tokens: int = 50) -> dict:
    """
    Benchmark speculative decoding performance.
    
    Args:
        draft_model: Draft model for speculation
        target_model: Target model for verification  
        test_prompts: Test prompts for benchmarking
        max_tokens: Tokens to generate per prompt
        
    Returns:
        Performance statistics dictionary
    """
    # Benchmark standard decoding
    standard_times = []
    for prompt in test_prompts:
        prompt_tensor = mx.array(prompt).reshape(1, -1)
        
        start_time = time.time()
        current_tokens = prompt_tensor
        for _ in range(max_tokens):
            logits = target_model(current_tokens)
            next_token = mx.argmax(logits[:, -1, :], axis=-1)
            current_tokens = mx.concat([current_tokens, next_token.reshape(1, 1)], axis=1)
        end_time = time.time()
        
        standard_times.append(end_time - start_time)
    
    # Benchmark speculative decoding
    speculative_times = []
    total_acceptance_rate = 0
    
    for prompt in test_prompts:
        prompt_tensor = mx.array(prompt).reshape(1, -1)
        decoder = SpeculativeDecoder(draft_model, target_model, lookahead=4)
        
        start_time = time.time()
        generated = decoder.generate(prompt_tensor, max_tokens)
        end_time = time.time()
        
        speculative_times.append(end_time - start_time)
        total_acceptance_rate += decoder.get_acceptance_rate()
    
    avg_standard_time = sum(standard_times) / len(standard_times)
    avg_speculative_time = sum(speculative_times) / len(speculative_times)
    avg_acceptance_rate = total_acceptance_rate / len(test_prompts)
    
    speedup = avg_standard_time / avg_speculative_time
    
    return {
        "standard_time_per_prompt": avg_standard_time,
        "speculative_time_per_prompt": avg_speculative_time,
        "speedup": speedup,
        "average_acceptance_rate": avg_acceptance_rate,
        "tokens_per_second_standard": max_tokens / avg_standard_time,
        "tokens_per_second_speculative": max_tokens / avg_speculative_time,
        "efficiency_gain": (speedup - 1) * 100  # Percentage improvement
    }
