from typing import Optional

from .attention import causal_mask
import mlx.core as mx


class TinyKvCache:
    def update_and_fetch(
        self,
        key: mx.array,
        value: mx.array,
        mask_length: int | None = None,
        mask: mx.array | str | None = None,
    ) -> tuple[mx.array, mx.array, int, mx.array]:
        """
        Update the KV cache with new key/value pairs and return all cached data.
        
        Args:
            key: New key tensor [B, H, L, D]
            value: New value tensor [B, H, L, D]
            mask_length: Length for attention mask
            mask: Attention mask
            
        Returns:
            Tuple of (all_keys, all_values, sequence_length, updated_mask)
        """
        pass


class BatchingKvCache(TinyKvCache):
    def __init__(self, max_active_requests: int, max_seq_len: int):
        """
        Initialize batching KV cache for multiple concurrent requests.
        
        TODO: Implement initialization
        - Store max_active_requests and max_seq_len
        - Create list of kv_caches with None entries
        - Initialize HD (head dimensions) tracking
        """
        pass

    def update_and_fetch(
        self,
        keys: mx.array,
        values: mx.array,
        mask_length: int | None = None,
        mask: mx.array | str | None = None,
    ) -> tuple[mx.array, mx.array, int, Optional[mx.array]]:
        """
        Update all active caches and return batched tensors.
        
        TODO: Implement batched update
        - Process each request in the batch
        - Handle variable sequence lengths
        - Create proper attention masks
        - Return padded/aligned tensors for batch processing
        """
        pass

    def add_request(self, prefilled: TinyKvCache, id: int):
        """
        Add a new request to the batch.
        
        TODO: Implement request addition
        - Validate id is in valid range
        - Store the prefilled cache at the correct slot
        - Update HD dimensions if needed
        """
        pass

    def remove_request(self, id: int):
        """
        Remove a finished request from the batch.
        
        TODO: Implement request removal  
        - Clear the cache slot
        - Handle cleanup
        """
        pass


class TinyKvFullCache(TinyKvCache):
    def __init__(self):
        """Initialize an empty full KV cache."""
        self.key_values = None  # Will store (keys, values) tuple
        self.offset = 0  # Current sequence length

    def update_and_fetch(
        self,
        key: mx.array,
        value: mx.array,
        mask_length: int | None = None,
        mask: mx.array | str | None = None,
    ) -> tuple[mx.array, mx.array, int, mx.array]:
        """
        Update cache with new key/value pairs.
        
        TODO: Implement this method
        - If this is the first call (self.key_values is None), store key/value directly
        - Otherwise, concatenate with existing keys/values along sequence dimension (axis=2)
        - Update self.offset to track total sequence length
        - Return (all_keys, all_values, seq_len, mask)
        """
        pass
    
    def get_offset(self):
        """Get current sequence offset."""
        return self.offset


class TinyKvRotatingCache(TinyKvCache):
    def __init__(self, max_seq_len: int):
        pass

    def update_and_fetch(
        self, key: mx.array, value: mx.array, offset: int
    ) -> tuple[mx.array, mx.array]:
        pass
