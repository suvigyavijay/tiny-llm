#include <iostream>
#include <vector>

#include "mlx/backend/common/utils.h"
#include "mlx/backend/cpu/encoder.h"
#include "mlx/utils.h"
#include "tiny_llm_ext.h"

#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#endif

namespace tiny_llm_ext_ref {

mx::array paged_attention(const mx::array &query, const mx::array &key_cache, const mx::array &value_cache,
                          const mx::array &block_tables, const mx::array &context_lens, const int block_size,
                          const float scale, const int num_kv_heads, const int num_heads, mx::StreamOrDevice s) {
    auto stream = mx::to_stream(s);
    
    return mx::array(
        {query.shape(0), num_heads, query.shape(2)}, // Output shape: [Batch, Num_Heads, Head_Dim]
        mx::float32,
        std::make_shared<PagedAttention>(stream, block_size, scale, num_kv_heads, num_heads),
        {query, key_cache, value_cache, block_tables, context_lens}
    );
}

void PagedAttention::eval_cpu(const std::vector<mx::array> &inputs, std::vector<mx::array> &outputs) {
    throw std::runtime_error("PagedAttention: CPU implementation not available (use GPU)");
}

void PagedAttention::eval_gpu(const std::vector<mx::array> &inputs, std::vector<mx::array> &outputs) {
#ifdef _METAL_
    auto &query = inputs[0];
    auto &key_cache = inputs[1];
    auto &value_cache = inputs[2];
    auto &block_tables = inputs[3];
    auto &context_lens = inputs[4];
    auto &output = outputs[0];

    auto &s = stream();
    auto &d = mx::metal::device(s.device);

    // Ensure outputs are allocated
    output.set_data(mx::allocator::malloc(output.nbytes()));

    // Get kernel
    auto library = d.get_library("tiny_llm_ext_ref");
    auto kernel = d.get_kernel("paged_attention_kernel", library);

    auto &compute_encoder = d.get_command_encoder(s.index);
    compute_encoder.set_compute_pipeline_state(kernel);

    // Set inputs
    compute_encoder.set_input_array(query, 0);
    compute_encoder.set_input_array(key_cache, 1);
    compute_encoder.set_input_array(value_cache, 2);
    compute_encoder.set_input_array(block_tables, 3);
    compute_encoder.set_input_array(context_lens, 4);
    compute_encoder.set_output_array(output, 5);

    // Set constants
    compute_encoder.set_bytes(num_heads_, 6);
    compute_encoder.set_bytes(num_kv_heads_, 7);
    int head_dim = query.shape(2);
    compute_encoder.set_bytes(head_dim, 8);
    compute_encoder.set_bytes(block_size_, 9);
    int max_blocks = block_tables.shape(1);
    compute_encoder.set_bytes(max_blocks, 10);
    compute_encoder.set_bytes(scale_, 11);
    int num_blocks = key_cache.shape(0);
    compute_encoder.set_bytes(num_blocks, 12);

    // Dispatch: One threadgroup per head per sequence
    // Grid: [Batch, Num_Heads, 1]
    // Threadgroup: [1, 1, 1] (Simplest - 1 thread handles loop; optimization: parallelize over heads or use SIMD)
    // Wait, in my metal kernel I used `group_id.x` as batch and `group_id.y` as head.
    // Metal dispatch is (Width, Height, Depth).
    // So Grid Size = (Batch, Num_Heads, 1).
    // Threadgroup size = (1, 1, 1). (This is very slow but correct for "v1").
    
    // Optimization: If we want to use threads, we should parallelize the loop inside the kernel.
    // For now, let's stick to 1 thread per head to keep the kernel logic simple as implemented in .metal.
    
    MTL::Size grid_size = MTL::Size(query.shape(0), num_heads_, 1);
    MTL::Size group_size = MTL::Size(1, 1, 1);
    
    compute_encoder.dispatch_threadgroups(grid_size, group_size);
#else
    throw std::runtime_error("PagedAttention: Metal not available");
#endif
}

} // namespace tiny_llm_ext_ref
