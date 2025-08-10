#include <metal_stdlib>
#include "mlx/backend/metal/kernels/utils.h"

using namespace metal;

template <typename T, int bits, int group_size>
[[kernel]] void quantized_matmul(
    device const T* scales [[buffer(0)]],
    device const T* biases [[buffer(1)]],
    device const T* a [[buffer(2)]],
    device const uint32_t* b [[buffer(3)]],
    device T* out [[buffer(4)]],
    device const int &M [[buffer(5)]],
    device const int &N [[buffer(6)]],
    device const int &K [[buffer(7)]],
    uint3 group_id [[threadgroup_position_in_grid]],
    uint3 thread_id [[thread_position_in_threadgroup]],
    uint3 threads_per_threadgroup [[threads_per_threadgroup]],
    [[maybe_unused]] threadgroup char * shmem [[threadgroup(0)]]) {
    constexpr const int packs_per_item = 32 / bits;
    const int groups_per_row = N / group_size;
    // Each thread processes an element in the output matrix
    const int i = group_id.x * threads_per_threadgroup.x + thread_id.x;
    const int k = group_id.y * threads_per_threadgroup.y + thread_id.y;
    float sum = 0;
    int scales_biases_loc = k * groups_per_row;
    constexpr const int mask = (1 << bits) - 1;
    // A: M * N, B: K * N where N gets quantized
    if (i < M && k < K) {
        int b_loc_base = k * N;
        int a_loc = i * N;
        for (int group_idx = 0; group_idx < groups_per_row; group_idx++) {
            const float scale = scales[scales_biases_loc];
            const float bias = biases[scales_biases_loc];
            for (int item_idx = 0; item_idx < group_size; item_idx += packs_per_item) {
                int b_loc = (b_loc_base + item_idx) * bits / 32;
                uint32_t b_val_packed = b[b_loc];
                for (int pack_idx = 0; pack_idx < packs_per_item; pack_idx++) {
                    sum += (static_cast<float>((b_val_packed >> (pack_idx * bits)) & mask) * scale + bias) * static_cast<float>(a[a_loc]);
                    a_loc++;
                }
            }
            scales_biases_loc++;
            b_loc_base += group_size;
        }
        out[i * K + k] = static_cast<T>(sum);
    }
}

#define instantiate_quantized_matmul(bits, group_size, p_type, m_type) \
    instantiate_kernel( \
        "quantized_matmul_w" #bits "a16_g" #group_size "_" #p_type, \
        quantized_matmul, \
        m_type, \
        bits, \
        group_size \
    )


instantiate_quantized_matmul(2, 32, f16, float16_t);
instantiate_quantized_matmul(2, 64, f16, float16_t);
instantiate_quantized_matmul(4, 32, f16, float16_t);
instantiate_quantized_matmul(4, 64, f16, float16_t);
instantiate_quantized_matmul(8, 32, f16, float16_t);
instantiate_quantized_matmul(8, 64, f16, float16_t);

instantiate_quantized_matmul(2, 32, bf16, bfloat16_t);
instantiate_quantized_matmul(2, 64, bf16, bfloat16_t);
instantiate_quantized_matmul(4, 32, bf16, bfloat16_t);
instantiate_quantized_matmul(4, 64, bf16, bfloat16_t);
instantiate_quantized_matmul(8, 32, bf16, bfloat16_t);
instantiate_quantized_matmul(8, 64, bf16, bfloat16_t);