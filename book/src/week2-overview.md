# Week 2: Build Your Own vLLM

In this week, we will focus on optimizing the inference process. We will implement several key techniques that are used in state-of-the-art LLM serving systems like vLLM. By the end of this week, you will have a high-performance inference engine that can handle multiple requests concurrently.

## What We will Cover

* **Key-Value Cache**: Implement a KV cache to avoid re-computing attention for previous tokens.
* **Quantization**: Implement quantized matrix multiplication on both CPU and GPU to reduce memory usage and accelerate computation.
* **Flash Attention**: Implement Flash Attention, a memory-efficient attention algorithm, on both CPU and GPU.
* **Continuous Batching**: Implement continuous batching to improve throughput by processing multiple requests concurrently.
* **Chunked Prefill**: Implement chunked prefill to handle long prompts more efficiently.

## What We will Not Cover

* **Advanced Quantization Techniques**: We will focus on a basic implementation of quantization. More advanced techniques like AWQ or GPT-Q are out of the scope of this course.
* **Paged Attention**: While we will implement the data structures for paged attention, the full implementation of the paged attention kernel is a complex topic that we will not cover in detail.

## vLLM

vLLM is a high-throughput and memory-efficient LLM serving library. It uses a novel memory management technique called PagedAttention to achieve state-of-the-art performance. In this week, we will implement several of the key optimizations that are used in vLLM.

**📚 Readings**

- [vLLM: Easy, Fast, and Cheap LLM Serving with PagedAttention](https://vllm.ai/)
- [PagedAttention for Large Language Models](https://vllm.ai/posts/2023-06-20-pagedattention.html)

## Additional Resources

Here are some additional resources that you may find helpful for this week's topics:

### Quantization
- [MLX CPU Quantized Kernel](https://github.com/ml-explore/mlx/blob/main/mlx/backend/cpu/quantized.cpp)
- [vLLM Linear Layer](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/linear.py)
- [MLX Extensions](https://ml-explore.github.io/mlx/build/html/dev/extensions.html)
- [llama.cpp Metal Kernels](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-metal/ggml-metal.metal)

### Flash Attention
- [MLX Metal SDPA Kernel](https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/kernels/sdpa_vector.h)
- [Metal Flash Attention Implementation](https://github.com/philipturner/metal-flash-attention)
- [Triton Fused Attention Tutorial](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html)

### General
- [Apple Metal vs NVIDIA CUDA](https://www.shashankshekhar.com/blog/apple-metal-vs-nvidia-cuda)
- [Hugging Face Transformers - Padding and Truncation](https://huggingface.co/docs/transformers/pad_truncation)

{{#include copyright.md}}
