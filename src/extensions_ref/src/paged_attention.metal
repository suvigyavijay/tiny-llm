#include <metal_stdlib>
#include "mlx/backend/metal/kernels/utils.h"

using namespace metal;

#define MAX_HEAD_DIM 256

[[kernel]] void paged_attention_kernel(
    device const float* q [[buffer(0)]],
    device const float* k_cache [[buffer(1)]],
    device const float* v_cache [[buffer(2)]],
    device const int* page_table [[buffer(3)]],
    device float* out [[buffer(4)]],
    constant const int& n_heads [[buffer(5)]],
    constant const int& head_dim [[buffer(6)]],
    constant const int& page_size [[buffer(7)]],
    constant const int& seq_len [[buffer(8)]],
    uint3 gid [[thread_position_in_grid]]) {
    
    static_assert(MAX_HEAD_DIM >= 64, "MAX_HEAD_DIM must be >= 64");

    // Deconstruct the gid
    int b = gid.x;
    int h = gid.y;
    int i = gid.z;

    if (i >= seq_len) {
        return;
    }

    // Get query vector
    int q_offset = b * n_heads * seq_len * head_dim + h * seq_len * head_dim + i * head_dim;
    device const float* q_vec = q + q_offset;
    
    // Accumulator for the output
    float out_vec[MAX_HEAD_DIM];
    for(int d=0; d<head_dim; ++d) {
        out_vec[d] = 0;
    }

    float max_score = -FLT_MAX;
    float exp_sum = 0.0;

    for (int j = 0; j <= i; ++j) {
        int page_idx = page_table[j / page_size];
        int page_offset = j % page_size;
        
        int k_offset = page_idx * n_heads * page_size * head_dim + h * page_size * head_dim + page_offset * head_dim;
        device const float* k_vec = k_cache + k_offset;

        float score = 0.0;
        for (int d = 0; d < head_dim; ++d) {
            score += q_vec[d] * k_vec[d];
        }
        score /= sqrt(float(head_dim));
        
        if (score > max_score) {
            float old_max_score = max_score;
            max_score = score;
            float scale = exp(old_max_score - max_score);
            exp_sum *= scale;
            for (int d = 0; d < head_dim; ++d) {
                out_vec[d] *= scale;
            }
        }

        float attention_weight = exp(score - max_score);
        exp_sum += attention_weight;

        int v_offset = page_idx * n_heads * page_size * head_dim + h * page_size * head_dim + page_offset * head_dim;
        device const float* v_vec = v_cache + v_offset;

        for (int d = 0; d < head_dim; ++d) {
            out_vec[d] += attention_weight * v_vec[d];
        }
    }
    
    int out_offset = b * n_heads * seq_len * head_dim + h * seq_len * head_dim + i * head_dim;
    for (int d = 0; d < head_dim; ++d) {
        out[out_offset + d] = out_vec[d] / exp_sum;
    }
}
