#include "tiny_llm_ext.h"
#include "mlx/backend/metal/device.h"
#include "mlx/stream.h"
#include "mlx/device.h"
#include "mlx/utils.h"
#include <iostream>

namespace tiny_llm_ext_ref {
    mx::array paged_attention(const mx::array& q, mx::array& k_cache, mx::array& v_cache, const mx::array& page_table, mx::StreamOrDevice s) {
        return mx::array(q.shape(), q.dtype(), std::make_shared<PagedAttention>(mx::to_stream(s)), {q, k_cache, v_cache, page_table});
    }

    void PagedAttention::eval_cpu(const std::vector<mx::array> &inputs, std::vector<mx::array> &outputs) {
        throw std::runtime_error("PagedAttention has no CPU implementation.");
    }

    void PagedAttention::eval_gpu(const std::vector<mx::array> &inputs, std::vector<mx::array> &outputs) {
        auto& q = inputs[0];
        auto& k_cache = inputs[1];
        auto& v_cache = inputs[2];
        auto& page_table = inputs[3];
        auto& out = outputs[0];

        auto& s = stream();
        auto& d = mx::metal::device(s.device);
        out.set_data(mx::allocator::malloc(out.nbytes()));

        auto library = d.get_library("tiny_llm_ext_ref");
        auto kernel = d.get_kernel("paged_attention_kernel", library);

        auto& compute_encoder = d.get_command_encoder(s.index);
        compute_encoder.set_compute_pipeline_state(kernel);

        compute_encoder.set_input_array(q, 0);
        compute_encoder.set_input_array(k_cache, 1);
        compute_encoder.set_input_array(v_cache, 2);
        compute_encoder.set_input_array(page_table, 3);
        compute_encoder.set_output_array(out, 4);

        MTL::Size grid_dims = MTL::Size(q.size(), 1, 1);
        MTL::Size group_dims = MTL::Size(kernel->maxTotalThreadsPerThreadgroup(), 1, 1);
        compute_encoder.dispatch_threads(grid_dims, group_dims);
    }
}
