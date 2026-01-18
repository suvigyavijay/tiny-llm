#include <metal_stdlib>
#include "mlx/backend/metal/kernels/utils.h"

using namespace metal;

// Paged Attention Kernel
// Supported Types: float32 (for simplicity in teaching)
// Query: [batch, num_heads, head_dim]
// Key Cache: [num_blocks, block_size, num_kv_heads, head_dim]
// Value Cache: [num_blocks, block_size, num_kv_heads, head_dim]
// Block Tables: [batch, max_blocks_per_seq]
// Context Lens: [batch]
// Output: [batch, num_heads, head_dim]

// Note: For multi-query attention (num_heads > num_kv_heads), we repeat KV.

[[kernel]] void paged_attention_kernel(
    device const float* query [[buffer(0)]],
    device const float* key_cache [[buffer(1)]],
    device const float* value_cache [[buffer(2)]],
    device const int* block_tables [[buffer(3)]],
    device const int* context_lens [[buffer(4)]],
    device float* output [[buffer(5)]],
    constant int& num_heads [[buffer(6)]],
    constant int& num_kv_heads [[buffer(7)]],
    constant int& head_dim [[buffer(8)]],
    constant int& block_size [[buffer(9)]],
    constant int& max_blocks_per_seq [[buffer(10)]],
    constant float& scale [[buffer(11)]],
    constant int& num_blocks [[buffer(12)]],
    uint3 group_id [[threadgroup_position_in_grid]],
    uint3 thread_id [[thread_position_in_threadgroup]]) {
    
    // Each threadgroup handles one head of one sequence
    const int batch_idx = group_id.x;
    const int head_idx = group_id.y;
    
    if (batch_idx >= context_lens[batch_idx]) return; // Should check bounds properly

    const int q_kv_ratio = num_heads / num_kv_heads;
    const int kv_head_idx = head_idx / q_kv_ratio;
    
    const int context_len = context_lens[batch_idx];
    const int num_seq_blocks = (context_len + block_size - 1) / block_size;
    
    // Pointer to this sequence's block table
    device const int* my_block_table = block_tables + batch_idx * max_blocks_per_seq;
    
    // Pointer to this head's query
    // Query shape: [batch, num_heads, head_dim]
    device const float* my_query = query + (batch_idx * num_heads + head_idx) * head_dim;
    
    // Thread Local Accumulators
    float max_score = -1e9f;
    float sum_exp = 0.0f;
    float acc[128]; // Max head_dim 128
    for (int i = 0; i < head_dim; i++) acc[i] = 0.0f;
    
    // 1. Compute Scores (Q * K^T)
    // We iterate over blocks.
    // For a real efficient kernel, we'd load Q into registers/shared memory.
    // Here we do a simple loop for educational clarity.
    
    // We need to store scores for Softmax. 
    // Since context_len can be large, we can't keep all scores in registers.
    // We use the Online Softmax trick (FlashAttention style).
    
    // Iterate over all logical blocks for this sequence
    for (int b = 0; b < num_seq_blocks; b++) {
        int physical_block_idx = my_block_table[b];
        
        // How many tokens in this block? 
        // usually block_size, except the last one.
        int tokens_in_block = block_size;
        if (b == num_seq_blocks - 1) {
            tokens_in_block = context_len % block_size;
            if (tokens_in_block == 0) tokens_in_block = block_size;
        }
        
        // Key Cache Shape: [num_blocks, block_size, num_kv_heads, head_dim]
        // Offset for this block
        long block_offset = (long)physical_block_idx * block_size * num_kv_heads * head_dim;
        
        for (int t = 0; t < tokens_in_block; t++) {
            // Calculate score for token t in block b
            float score = 0.0f;
            
            // Pointer to K vector for this token and kv_head
            // K[block, t, kv_head, :]
            long k_offset = block_offset + (long)t * num_kv_heads * head_dim + (long)kv_head_idx * head_dim;
            
            for (int d = 0; d < head_dim; d++) {
                score += my_query[d] * key_cache[k_offset + d];
            }
            score *= scale;
            
            // Online Softmax Update
            float old_max = max_score;
            max_score = max(max_score, score);
            float exp_score = exp(score - max_score);
            float exp_old_max = exp(old_max - max_score);
            
            sum_exp = sum_exp * exp_old_max + exp_score;
            
            // Pointer to V vector
            // V[block, t, kv_head, :]
            long v_offset = block_offset + (long)t * num_kv_heads * head_dim + (long)kv_head_idx * head_dim;
            
            for (int d = 0; d < head_dim; d++) {
                acc[d] = acc[d] * exp_old_max + exp_score * value_cache[v_offset + d];
            }
        }
    }
    
    // 2. Write Output
    // Output shape: [batch, num_heads, head_dim]
    device float* my_output = output + (batch_idx * num_heads + head_idx) * head_dim;
    
    for (int d = 0; d < head_dim; d++) {
        my_output[d] = acc[d] / sum_exp;
    }
}
