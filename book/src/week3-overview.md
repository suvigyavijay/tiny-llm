# Week 3: Serving at Scale

In this week, we will shift focus from single-batch inference (Week 1) and core optimization techniques (Week 2) to the advanced topics required for building a production-grade **LLM Serving System**.

This week covers the "secret sauce" behind high-throughput engines like vLLM and TGI, as well as how models interact with the outside world.

## What We will Cover

* PagedAttention and continuous batching
* Mixture of Experts (MoE) routing and execution
* Speculative Decoding for faster inference
* RAG Pipeline implementation
* Agents & Tool Use with ReAct loop
* Long Context scaling (RoPE interpolation/YaRN)

## Goals

By the end of this week, you will have a comprehensive understanding of the modern LLM serving stack, from the lowest-level kernel optimizations (PagedAttention) to the highest-level application logic (Agents).

{{#include copyright.md}}
