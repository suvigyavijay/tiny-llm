# Week 2: Optimizing LLM Inference

In Week 2, we dive into the optimizations that make LLM serving practical and efficient. Building on the foundational understanding from Week 1, we'll implement key optimizations used in production LLM serving systems like vLLM.

We will use the same Qwen2-7B-Instruct model, but now with quantized operations and custom kernels to dramatically improve memory usage and inference speed.

## What We Will Cover

* **Key-Value Cache**: Implement efficient caching to avoid recomputing attention for previous tokens
* **Quantized Operations**: Use INT4 quantization for linear layers to reduce memory usage
* **Flash Attention**: Implement memory-efficient attention computation for both CPU and GPU
* **Continuous Batching**: Serve multiple requests simultaneously with dynamic batching
* **Chunked Prefill**: Break down long prefill sequences into manageable chunks

## What We Will Learn

### Memory Optimizations
- How KV caching reduces computation from O(n²) to O(n) for autoregressive generation
- INT4 quantization reduces model memory usage by ~75% with minimal quality loss
- Flash Attention reduces memory complexity from O(n²) to O(n) for attention

### Performance Optimizations
- Custom C++/Metal kernels for quantized matrix multiplication
- GPU-optimized Flash Attention using Metal Shading Language
- Batching strategies to maximize throughput

### System Design
- How to handle variable-length sequences in batched inference
- Request scheduling and memory management
- Prefill vs decode phase optimizations

## Architecture Overview

Week 2 introduces three key components:

1. **Quantized Model (`Qwen2ModelWeek2`)**: Uses INT4 weights with optimized kernels
2. **KV Cache System**: Manages cached key-value pairs for efficient generation  
3. **Batching Engine**: Handles multiple concurrent requests

The system maintains the same API as Week 1 but with dramatically improved performance characteristics.

## Performance Improvements

Compared to Week 1, you should expect:
- **Memory**: 75% reduction in model memory usage (via quantization)
- **Speed**: 3-5x faster attention computation (via Flash Attention)
- **Throughput**: 10-20x higher request throughput (via batching)

## Technical Deep Dives

Each day focuses on a specific optimization:

- **Day 1 (2.1)**: KV Cache data structures and algorithms
- **Day 2-3 (2.2-2.3)**: Quantized linear operations on CPU and GPU
- **Day 4-5 (2.4-2.5)**: Flash Attention implementation and optimization
- **Day 6 (2.6)**: Continuous batching for dynamic request handling
- **Day 7 (2.7)**: Chunked prefill for long context processing

## Prerequisites

- Completion of Week 1
- Basic understanding of CUDA/Metal programming concepts (helpful but not required)
- Familiarity with matrix operations and memory layouts

## Environment Setup

Week 2 introduces C++/Metal extensions. Make sure you can build the extensions:

```bash
# Build the extensions (required for Week 2)
pdm run python src/extensions_ref/build.py
```

The extensions provide optimized kernels for quantized operations and Flash Attention.

**📚 Key Readings**

- [Flash Attention Paper](https://arxiv.org/abs/2205.14135)
- [Continuous Batching in vLLM](https://blog.vllm.ai/2023/06/20/vllm.html)
- [Quantization Techniques](https://huggingface.co/blog/4bit-transformers-bitsandbytes)
- [vLLM Architecture Deep Dive](https://github.com/vllm-project/vllm)

{{#include copyright.md}}