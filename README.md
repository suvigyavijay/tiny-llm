# tiny-llm - LLM Serving in a Week

[![CI (main)](https://github.com/skyzh/tiny-llm/actions/workflows/main.yml/badge.svg)](https://github.com/skyzh/tiny-llm/actions/workflows/main.yml)

A hands-on course on LLM serving using MLX for system engineers. The codebase
is solely (almost!) based on MLX array/matrix APIs without any high-level neural network APIs, so that we
can build the model serving infrastructure from scratch and dig into the optimizations.

The goal is to learn the techniques behind efficiently serving a large language model (e.g., Qwen2 models).

In this course, you will learn:
- **Week 1: Build Your Own Qwen2 From Scratch**: You will implement the necessary components in Python to use the Qwen2 model to generate responses (e.g., attention, RoPE, etc).
- **Week 2: Build Your Own vLLM**: You will implement the inference system which is similar to but a much simpler version of vLLM (e.g., KV cache, continuous batching, flash attention, etc).
- **Week 3: Advanced Topics**: We will cover more advanced topics and how the model interacts with the outside world (e.g., paged attention, MoE, RAG, tool calling, etc).

Why MLX: nowadays it's easier to get a macOS-based local development environment than setting up an NVIDIA GPU.

Why Qwen2: this was the first LLM I've interacted with -- it's the go-to example in the vllm documentation. I spent some time looking at the vllm source code and built some knowledge around it.

## Book

The tiny-llm book is available at [https://skyzh.github.io/tiny-llm/](https://skyzh.github.io/tiny-llm/). You can follow the guide and start building.

## Running the Tests

To run the tests for a specific day, you can use the following command:

```
pdm run test-refsol tests_refsol/test_week_1_day_1.py
```

Replace `week_1_day_1` with the week and day you want to test.

## Community

You may join skyzh's Discord server and study with the tiny-llm community.

[![Join skyzh's Discord Server](book/src/discord-badge.svg)](https://skyzh.dev/join/discord)

## Roadmap

| Week + Chapter | Topic                                                       | Code | Test | Doc |
| -------------- | ----------------------------------------------------------- | ---- | ---- | --- |
| 1.1            | Attention                                                   | ✅    | ✅   | ✅  |
| 1.2            | RoPE                                                        | ✅    | ✅   | ✅  |
| 1.3            | Grouped Query Attention                                     | ✅    | ✅   | ✅  |
| 1.4            | RMSNorm and MLP                                             | ✅    | ✅   | ✅  |
| 1.5            | Load the Model                                              | ✅    | ✅   | ✅  |
| 1.6            | Generate Responses (aka Decoding)                           | ✅    | ✅   | ✅  |
| 1.7            | Sampling                                                    | ✅    | ✅   | ✅  |
| 2.1            | Key-Value Cache                                             | ✅    | ✅   | ✅  |
| 2.2            | Quantized Matmul and Linear - CPU                           | ✅    | ✅   | ✅  |
| 2.3            | Quantized Matmul and Linear - GPU                           | ✅    | ✅   | ✅  |
| 2.4            | Flash Attention 2 - CPU                                     | ✅    | ✅   | ✅  |
| 2.5            | Flash Attention 2 - GPU                                     | ✅    | ✅   | ✅  |
| 2.6            | Continuous Batching                                         | ✅    | ✅   | ✅  |
| 2.7            | Chunked Prefill                                             | ✅    | ✅   | ✅  |
| 3.1            | Paged Attention - Data Structures                           | ✅    | ✅   | ✅  |
| 3.2            | Paged Attention - Function                                  | ✅    | ✅   | ✅  |
| 3.3            | MoE (Mixture of Experts)                                    | ✅    | ✅   | ✅  |
| 3.4            | Speculative Decoding                                        | ✅    | ✅   | ✅  |
| 3.5            | RAG Pipeline                                                | ✅    | ✅   | ✅  |
| 3.6            | AI Agent / Tool Calling                                     | ✅    | ✅   | ✅  |
| 3.7            | Long Context                                                | ✅    | ✅   | ✅  |

Other topics not covered: quantized/compressed kv cache, prefix/prompt cache; sampling, fine tuning; smaller kernels (softmax, silu, etc)
