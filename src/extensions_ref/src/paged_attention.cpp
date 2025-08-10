#include "tiny_llm_ext.h"
#include "mlx/backend/metal/device.h"

namespace tiny_llm_ext_ref {
    mx::array paged_attention(const mx::array& q, mx::array& k_cache, mx::array& v_cache, const mx::array& page_table, mx::StreamOrDevice s) {
        auto& d = mx::metal::device(s.device);
        auto out = mx::empty_like(q);

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

        return out;
    }
}
