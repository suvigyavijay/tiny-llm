# Week 3 Day 3: Mixture of Experts (MoE)

Mixture of Experts (MoE) is a powerful architecture that allows for scaling up the number of parameters in a model without a proportional increase in computational cost. This is achieved by having a set of "expert" sub-networks and a "gating" network that learns to route each token to the most appropriate experts.

In this chapter, you will implement a simplified MoE layer. This will give you a foundational understanding of how MoE models work and how they can be used to build more powerful and efficient language models.

[📚 Reading: Mixture of Experts Explained](https://huggingface.co/blog/moe)

## Task 1: Implement the MoE Layer

Your task is to implement the `MoE` layer, along with the `Expert` and `Gating` classes.

```
src/tiny_llm/moe.py
```

The implementation will consist of three main components:
- **`Expert`**: A standard feed-forward network that will act as one of the experts.
- **`Gating`**: A network that takes the input token and outputs a probability distribution over the experts.
- **`MoE`**: The main layer that combines the gating network and the experts. It will route each token to the top-k experts based on the gating weights and then combine their outputs.

You can run the following tests to verify your implementation:

```
pdm run test --week 3 --day 3
```

This simplified implementation will give you a solid understanding of the core concepts behind MoE. In a production system, you would use more advanced techniques for routing and load balancing to ensure efficient use of hardware.

{{#include copyright.md}}
