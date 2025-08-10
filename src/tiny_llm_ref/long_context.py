"""
Long Context Handling techniques for processing very long sequences efficiently.

Implements various strategies for handling 100K+ token sequences while maintaining
quality and performance through efficient attention patterns and context management.
"""

import mlx.core as mx
from typing import List, Optional, Dict, Any, Tuple, Generator
import math
from .attention import scaled_dot_product_attention_grouped, causal_mask


class SlidingWindowAttention:
    """Attention with local sliding window for long sequences."""
    
    def __init__(self, window_size: int = 512, global_tokens: int = 0):
        """
        Initialize sliding window attention.
        
        Args:
            window_size: Size of local attention window
            global_tokens: Number of global tokens that attend to everything
        """
        self.window_size = window_size
        self.global_tokens = global_tokens
    
    def create_sliding_mask(self, seq_len: int, dtype: mx.Dtype = mx.float32) -> mx.array:
        """
        Create sliding window attention mask.
        
        Args:
            seq_len: Sequence length
            dtype: Data type for mask
            
        Returns:
            Attention mask with sliding window pattern
        """
        mask = mx.full((seq_len, seq_len), -mx.inf, dtype=dtype)
        
        # Create sliding window
        for i in range(seq_len):
            # Each token can attend to tokens within window
            start = max(0, i - self.window_size + 1)
            end = i + 1  # Causal: can't see future tokens
            mask = mask.at[i, start:end].set(0.0)
            
            # Global tokens (if any) can attend to everything up to their position
            if i < self.global_tokens:
                mask = mask.at[i, :i+1].set(0.0)
            
            # All tokens can attend to global tokens
            if self.global_tokens > 0:
                mask = mask.at[i, :self.global_tokens].set(0.0)
        
        return mask
    
    def sliding_attention(self, q: mx.array, k: mx.array, v: mx.array, 
                         scale: Optional[float] = None) -> mx.array:
        """
        Apply sliding window attention efficiently.
        
        Args:
            q: Query tensor [batch, heads, seq_len, head_dim]
            k: Key tensor [batch, heads, seq_len, head_dim]
            v: Value tensor [batch, heads, seq_len, head_dim]
            scale: Attention scale factor
            
        Returns:
            Attention output
        """
        batch, heads, seq_len, head_dim = q.shape
        
        if scale is None:
            scale = 1.0 / math.sqrt(head_dim)
        
        # Create sliding window mask
        mask = self.create_sliding_mask(seq_len, q.dtype)
        
        # Apply standard attention with mask
        return scaled_dot_product_attention_grouped(q, k, v, scale=scale, mask=mask)


class HierarchicalAttention:
    """Multi-scale hierarchical attention for long sequences."""
    
    def __init__(self, levels: List[int] = [64, 512, 4096]):
        """
        Initialize hierarchical attention.
        
        Args:
            levels: Attention scales from fine to coarse
        """
        self.levels = levels
        self.num_levels = len(levels)
    
    def create_hierarchical_masks(self, seq_len: int) -> List[mx.array]:
        """
        Create attention masks for each hierarchical level.
        
        Args:
            seq_len: Sequence length
            
        Returns:
            List of masks for each level
        """
        masks = []
        
        for level_size in self.levels:
            mask = mx.full((seq_len, seq_len), -mx.inf, dtype=mx.float32)
            
            # Create attention pattern for this level
            for i in range(seq_len):
                # Attend to tokens within this level's window
                start = max(0, i - level_size + 1)
                end = i + 1
                mask = mask.at[i, start:end].set(0.0)
            
            masks.append(mask)
        
        return masks
    
    def hierarchical_forward(self, q: mx.array, k: mx.array, v: mx.array,
                           layer_weights: Optional[List[float]] = None) -> mx.array:
        """
        Process through attention hierarchy.
        
        Args:
            q: Query tensor
            k: Key tensor
            v: Value tensor
            layer_weights: Weights for combining different levels
            
        Returns:
            Combined hierarchical attention output
        """
        seq_len = q.shape[2]
        
        if layer_weights is None:
            layer_weights = [1.0 / self.num_levels] * self.num_levels
        
        # Create masks for each level
        masks = self.create_hierarchical_masks(seq_len)
        
        # Compute attention at each level
        level_outputs = []
        for i, mask in enumerate(masks):
            level_output = scaled_dot_product_attention_grouped(
                q, k, v, mask=mask, scale=1.0 / math.sqrt(q.shape[-1])
            )
            level_outputs.append(level_output * layer_weights[i])
        
        # Combine outputs from all levels
        return sum(level_outputs)


class RingAttention:
    """Ring attention for distributed long sequence processing."""
    
    def __init__(self, ring_size: int, block_size: int = 1024):
        """
        Initialize ring attention.
        
        Args:
            ring_size: Number of devices/segments in ring
            block_size: Size of each processing block
        """
        self.ring_size = ring_size
        self.block_size = block_size
    
    def partition_sequence(self, seq_len: int) -> List[Tuple[int, int]]:
        """
        Partition sequence for ring processing.
        
        Args:
            seq_len: Total sequence length
            
        Returns:
            List of (start, end) indices for each partition
        """
        partition_size = (seq_len + self.ring_size - 1) // self.ring_size
        partitions = []
        
        for i in range(self.ring_size):
            start = i * partition_size
            end = min((i + 1) * partition_size, seq_len)
            if start < seq_len:
                partitions.append((start, end))
        
        return partitions
    
    def ring_attention_step(self, q_block: mx.array, k_block: mx.array, v_block: mx.array,
                          ring_position: int, causal_offset: int = 0) -> mx.array:
        """
        Single ring attention step.
        
        Args:
            q_block: Query block for this step
            k_block: Key block being processed
            v_block: Value block being processed
            ring_position: Position in the ring
            causal_offset: Offset for causal masking
            
        Returns:
            Partial attention output for this step
        """
        # Compute attention scores
        scores = mx.matmul(q_block, k_block.transpose(0, 1, 3, 2))
        scores = scores / math.sqrt(q_block.shape[-1])
        
        # Apply causal mask if needed
        if causal_offset >= 0:
            q_len = q_block.shape[2]
            k_len = k_block.shape[2]
            
            # Create causal mask considering the offset
            mask = mx.triu(mx.full((q_len, k_len), -mx.inf), k=k_len - q_len + causal_offset + 1)
            scores = scores + mask[None, None, :, :]
        
        # Apply softmax and compute output
        attn_weights = mx.softmax(scores, axis=-1)
        output = mx.matmul(attn_weights, v_block)
        
        return output, attn_weights
    
    def ring_attention(self, q: mx.array, k: mx.array, v: mx.array) -> mx.array:
        """
        Simulate ring attention processing.
        
        Args:
            q: Query tensor
            k: Key tensor
            v: Value tensor
            
        Returns:
            Ring attention output
        """
        batch, heads, seq_len, head_dim = q.shape
        
        # Partition sequence
        partitions = self.partition_sequence(seq_len)
        
        # Initialize output
        output = mx.zeros_like(q)
        
        # Process each query partition
        for q_idx, (q_start, q_end) in enumerate(partitions):
            q_block = q[:, :, q_start:q_end, :]
            
            # Accumulate attention from all key-value partitions
            q_output = mx.zeros_like(q_block)
            total_weights = mx.zeros((batch, heads, q_end - q_start, seq_len))
            
            for kv_idx, (kv_start, kv_end) in enumerate(partitions):
                k_block = k[:, :, kv_start:kv_end, :]
                v_block = v[:, :, kv_start:kv_end, :]
                
                # Compute causal offset
                causal_offset = kv_start - q_start
                
                # Ring attention step
                step_output, step_weights = self.ring_attention_step(
                    q_block, k_block, v_block, kv_idx, causal_offset
                )
                
                # Accumulate output and weights
                q_output += step_output
                total_weights[:, :, :, kv_start:kv_end] = step_weights
            
            # Normalize by total attention weights
            output[:, :, q_start:q_end, :] = q_output
        
        return output


class ContextCompressor:
    """Compress long contexts by selecting important tokens."""
    
    def __init__(self, compression_ratio: float = 0.5, importance_method: str = "attention"):
        """
        Initialize context compressor.
        
        Args:
            compression_ratio: Fraction of tokens to keep (0.5 = keep 50%)
            importance_method: Method for computing token importance
        """
        self.compression_ratio = compression_ratio
        self.importance_method = importance_method
    
    def compute_attention_importance(self, tokens: mx.array, attention_weights: mx.array) -> mx.array:
        """
        Compute token importance based on attention patterns.
        
        Args:
            tokens: Input tokens
            attention_weights: Attention weight matrices
            
        Returns:
            Importance scores for each token
        """
        # Average attention weights across heads and layers
        avg_attention = mx.mean(attention_weights, axis=(0, 1))  # [seq_len, seq_len]
        
        # Sum incoming attention (how much other tokens attend to this one)
        incoming_attention = mx.sum(avg_attention, axis=0)
        
        # Sum outgoing attention (how much this token attends to others)
        outgoing_attention = mx.sum(avg_attention, axis=1)
        
        # Combine incoming and outgoing attention
        importance = incoming_attention + outgoing_attention
        
        return importance
    
    def compute_gradient_importance(self, tokens: mx.array, gradients: mx.array) -> mx.array:
        """
        Compute token importance based on gradients.
        
        Args:
            tokens: Input tokens
            gradients: Gradient information
            
        Returns:
            Importance scores for each token
        """
        # Use gradient magnitude as importance
        importance = mx.linalg.norm(gradients, axis=-1)
        return importance
    
    def compress_context(self, tokens: mx.array, importance_scores: mx.array,
                        preserve_positions: Optional[List[int]] = None) -> Tuple[mx.array, mx.array]:
        """
        Compress context by selecting important tokens.
        
        Args:
            tokens: Input token sequence
            importance_scores: Importance score for each token
            preserve_positions: Token positions to always preserve
            
        Returns:
            Tuple of (compressed_tokens, position_mapping)
        """
        seq_len = tokens.shape[0] if tokens.ndim == 1 else tokens.shape[1]
        target_length = int(seq_len * self.compression_ratio)
        
        # Always preserve certain positions (e.g., start/end tokens)
        if preserve_positions is None:
            preserve_positions = [0, seq_len - 1]  # Preserve first and last tokens
        
        # Mark positions to preserve
        preserved_mask = mx.zeros(seq_len, dtype=bool)
        for pos in preserve_positions:
            if 0 <= pos < seq_len:
                preserved_mask = preserved_mask.at[pos].set(True)
        
        # Number of additional tokens to select
        additional_tokens = target_length - sum(preserve_positions)
        
        if additional_tokens <= 0:
            # Just keep preserved positions
            selected_indices = mx.array(preserve_positions)
        else:
            # Select top additional tokens by importance
            available_scores = mx.where(preserved_mask, -mx.inf, importance_scores)
            top_indices = mx.argsort(available_scores)[-additional_tokens:]
            
            # Combine preserved and selected indices
            all_indices = preserve_positions + top_indices.tolist()
            selected_indices = mx.array(sorted(all_indices))
        
        # Extract compressed tokens
        if tokens.ndim == 1:
            compressed_tokens = tokens[selected_indices]
        else:
            compressed_tokens = tokens[:, selected_indices]
        
        # Create position mapping (original_pos -> compressed_pos)
        position_mapping = mx.full((seq_len,), -1, dtype=mx.int32)
        for new_pos, old_pos in enumerate(selected_indices):
            position_mapping = position_mapping.at[old_pos].set(new_pos)
        
        return compressed_tokens, position_mapping


class StreamingProcessor:
    """Process very long sequences in streaming fashion."""
    
    def __init__(self, chunk_size: int = 2048, overlap: int = 256):
        """
        Initialize streaming processor.
        
        Args:
            chunk_size: Size of each processing chunk
            overlap: Overlap between consecutive chunks
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        
    def chunk_sequence(self, sequence: mx.array) -> List[Tuple[mx.array, int, int]]:
        """
        Split sequence into overlapping chunks.
        
        Args:
            sequence: Input sequence to chunk
            
        Returns:
            List of (chunk, start_pos, end_pos) tuples
        """
        seq_len = sequence.shape[-1] if sequence.ndim > 1 else sequence.shape[0]
        chunks = []
        
        start = 0
        while start < seq_len:
            end = min(start + self.chunk_size, seq_len)
            
            if sequence.ndim == 1:
                chunk = sequence[start:end]
            else:
                chunk = sequence[:, start:end]
            
            chunks.append((chunk, start, end))
            
            if end >= seq_len:
                break
                
            start = end - self.overlap
        
        return chunks
    
    def process_stream(self, token_stream: Generator[mx.array, None, None], 
                      model: Any) -> Generator[mx.array, None, None]:
        """
        Process infinite token stream in chunks.
        
        Args:
            token_stream: Generator yielding tokens
            model: Model to process chunks
            
        Yields:
            Processing results as they become available
        """
        buffer = []
        
        for tokens in token_stream:
            buffer.extend(tokens.tolist() if hasattr(tokens, 'tolist') else [tokens])
            
            # Process when we have enough tokens
            while len(buffer) >= self.chunk_size:
                # Extract chunk with overlap consideration
                chunk_tokens = mx.array(buffer[:self.chunk_size])
                
                # Process chunk
                result = model(chunk_tokens)
                yield result
                
                # Move buffer forward (keeping overlap)
                buffer = buffer[self.chunk_size - self.overlap:]
        
        # Process remaining tokens
        if buffer:
            chunk_tokens = mx.array(buffer)
            result = model(chunk_tokens)
            yield result
    
    def merge_chunk_outputs(self, chunk_outputs: List[mx.array], 
                          chunk_positions: List[Tuple[int, int]]) -> mx.array:
        """
        Merge overlapping chunk outputs.
        
        Args:
            chunk_outputs: List of chunk processing results
            chunk_positions: List of (start, end) positions for each chunk
            
        Returns:
            Merged output sequence
        """
        if not chunk_outputs:
            return mx.array([])
        
        # Determine total output length
        total_length = max(end for _, end in chunk_positions)
        output_dim = chunk_outputs[0].shape[-1]
        
        # Initialize output
        merged_output = mx.zeros((total_length, output_dim))
        weight_sum = mx.zeros((total_length, 1))
        
        # Merge chunks with overlap handling
        for output, (start, end) in zip(chunk_outputs, chunk_positions):
            chunk_len = end - start
            
            # Weight function for smooth blending in overlap regions
            weights = mx.ones((chunk_len, 1))
            
            # Reduce weights at boundaries for smooth blending
            if start > 0:  # Not first chunk
                fade_in = min(self.overlap, chunk_len) // 2
                for i in range(fade_in):
                    weights = weights.at[i].set(i / fade_in)
            
            if end < total_length:  # Not last chunk
                fade_out = min(self.overlap, chunk_len) // 2
                for i in range(fade_out):
                    pos = chunk_len - 1 - i
                    weights = weights.at[pos].set(i / fade_out)
            
            # Add weighted contribution
            merged_output = merged_output.at[start:end].add(output * weights)
            weight_sum = weight_sum.at[start:end].add(weights)
        
        # Normalize by weights
        merged_output = merged_output / mx.maximum(weight_sum, 1e-8)
        
        return merged_output


class AdaptiveContextManager:
    """Intelligently manage context for very long conversations."""
    
    def __init__(self, max_context: int = 32768, summary_ratio: float = 0.1):
        """
        Initialize adaptive context manager.
        
        Args:
            max_context: Maximum context length to maintain
            summary_ratio: Fraction of old context to summarize
        """
        self.max_context = max_context
        self.summary_ratio = summary_ratio
        self.conversation_segments = []
        
    def add_message(self, message: str, role: str = "user"):
        """Add new message to conversation."""
        self.conversation_segments.append({
            "content": message,
            "role": role,
            "timestamp": time.time(),
            "importance": 1.0  # Default importance
        })
    
    def estimate_token_count(self, text: str) -> int:
        """Estimate token count for text (simple approximation)."""
        return len(text.split()) * 1.3  # Rough approximation
    
    def extract_key_points(self, text: str) -> List[str]:
        """
        Extract key information from text.
        
        Args:
            text: Text to analyze
            
        Returns:
            List of key points
        """
        # Simple key point extraction (in practice would use more sophisticated NLP)
        sentences = text.split('.')
        
        # Score sentences by length and certain keywords
        scored_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 10:  # Skip very short sentences
                continue
                
            score = len(sentence)
            
            # Boost score for important keywords
            important_words = ['important', 'key', 'main', 'critical', 'essential', 'remember']
            for word in important_words:
                if word in sentence.lower():
                    score *= 1.5
            
            scored_sentences.append((sentence, score))
        
        # Return top-scoring sentences
        scored_sentences.sort(key=lambda x: x[1], reverse=True)
        top_sentences = [sent for sent, score in scored_sentences[:3]]
        
        return top_sentences
    
    def summarize_old_context(self, segments: List[Dict]) -> str:
        """
        Summarize old conversation segments.
        
        Args:
            segments: List of conversation segments to summarize
            
        Returns:
            Summary text
        """
        # Extract key points from each segment
        all_key_points = []
        for segment in segments:
            key_points = self.extract_key_points(segment["content"])
            all_key_points.extend(key_points)
        
        # Create summary
        if all_key_points:
            summary = "Previous conversation summary:\n" + "\n".join(f"- {point}" for point in all_key_points[:5])
        else:
            summary = "Previous conversation occurred but no key points identified."
        
        return summary
    
    def manage_context(self, new_input: str) -> str:
        """
        Manage conversation context, summarizing old content as needed.
        
        Args:
            new_input: New input to add to context
            
        Returns:
            Managed context string
        """
        self.add_message(new_input, "user")
        
        # Calculate total context length
        total_tokens = sum(self.estimate_token_count(seg["content"]) 
                          for seg in self.conversation_segments)
        
        if total_tokens <= self.max_context:
            # Context fits, return full conversation
            return "\n".join(seg["content"] for seg in self.conversation_segments)
        
        # Need to compress context
        tokens_to_summarize = int(total_tokens * self.summary_ratio)
        
        # Find segments to summarize (oldest first)
        segments_to_summarize = []
        summarized_tokens = 0
        
        for segment in self.conversation_segments:
            if summarized_tokens >= tokens_to_summarize:
                break
            segments_to_summarize.append(segment)
            summarized_tokens += self.estimate_token_count(segment["content"])
        
        # Create summary
        summary = self.summarize_old_context(segments_to_summarize)
        
        # Keep remaining segments
        remaining_segments = self.conversation_segments[len(segments_to_summarize):]
        
        # Combine summary with remaining context
        context_parts = [summary] + [seg["content"] for seg in remaining_segments]
        return "\n".join(context_parts)


def benchmark_long_context_techniques(models: Dict[str, Any], test_sequences: List[mx.array]) -> Dict[str, Dict[str, float]]:
    """
    Benchmark different long context techniques.
    
    Args:
        models: Dictionary of model implementations to test
        test_sequences: List of test sequences of varying lengths
        
    Returns:
        Benchmark results for each technique
    """
    results = {}
    
    techniques = {
        "sliding_window": SlidingWindowAttention(),
        "hierarchical": HierarchicalAttention(),
        "ring_attention": RingAttention(ring_size=4),
        "compressed": ContextCompressor()
    }
    
    for technique_name, technique in techniques.items():
        technique_results = {
            "avg_time": 0.0,
            "max_memory": 0.0,
            "accuracy": 0.0,
            "sequences_processed": 0
        }
        
        for seq in test_sequences:
            if seq.shape[-1] > 1000:  # Only test on long sequences
                try:
                    start_time = time.time()
                    
                    # Process sequence with technique
                    if technique_name == "sliding_window":
                        q = k = v = mx.random.normal((*seq.shape, 64))
                        output = technique.sliding_attention(q, k, v)
                    elif technique_name == "hierarchical":
                        q = k = v = mx.random.normal((*seq.shape, 64))
                        output = technique.hierarchical_forward(q, k, v)
                    elif technique_name == "ring_attention":
                        q = k = v = mx.random.normal((*seq.shape, 64))
                        output = technique.ring_attention(q, k, v)
                    elif technique_name == "compressed":
                        importance = mx.random.uniform(shape=(seq.shape[-1],))
                        compressed, mapping = technique.compress_context(seq, importance)
                        output = compressed
                    
                    processing_time = time.time() - start_time
                    
                    technique_results["avg_time"] += processing_time
                    technique_results["sequences_processed"] += 1
                    
                except Exception as e:
                    print(f"Error processing {technique_name}: {e}")
        
        # Average results
        if technique_results["sequences_processed"] > 0:
            technique_results["avg_time"] /= technique_results["sequences_processed"]
        
        results[technique_name] = technique_results
    
    return results
