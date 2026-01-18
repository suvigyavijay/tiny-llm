# Week 2 Day 2: Quantized Matrix Multiplication (CPU)

In this chapter, we will optimize our model by implementing **quantized matrix multiplication**. This will allow us to run the model with significantly lower memory usage.

**📚 Readings**

- [A Visual Guide to Quantization - Maarten Grootendorst](https://www.maartengrootendorst.com/blog/quantization/)
- [LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale - Tim Dettmers](https://timdettmers.com/2022/08/17/llm-int8-and-emergent-features/)
- [Making LLMs even more accessible with bitsandbytes, 4-bit quantization and QLoRA - Hugging Face](https://huggingface.co/blog/4bit-transformers-bitsandbytes)

## Motivation

Large Language Models have billions of parameters. A 7B model in `float16` (2 bytes per parameter) requires about 14GB of RAM just to store the weights. By quantizing weights to 4-bit integers (0.5 bytes per parameter), we can reduce this to ~3.5GB, making it possible to run on consumer hardware with much less memory bandwidth pressure.

As Maarten Grootendorst explains visually, quantization maps a large range of continuous values (floats) to a smaller, discrete set of values (integers). This is like reducing the number of colors in an image: you lose some fidelity, but the file size drops dramatically.

## Quantization Scheme: Group-wise Quantization

We will use a **block-wise** (or group-wise) quantization scheme. A naive approach would quantize the entire matrix with a single scale/bias, but outliers in one part of the matrix could skew the scale for everyone else (as detailed in Tim Dettmers' blog).

Instead, we divide the weight matrix $W$ into groups of size $G$ (e.g., 64). Each group has its own local statistics.
For each group, we store:
- A **scale** (float16): Determines the step size.
- A **bias** (float16): Shifts the zero-point.
- The **quantized weights** (4-bit integers): The indices.

The reconstruction formula for a weight $w$ is:
$$ w \approx w_{quant} \times \text{scale} + \text{bias} $$

Where $w_{quant}$ is the 4-bit integer value.

Consider a small example (group size = 4):

```plain
Original weights (float): [0.1, 0.5, -0.2, 0.9]
Quantized (4-bit):        [2,   10,   0,    14]
Scale: 0.05
Bias:  0.0

Reconstruction:
2 * 0.05 = 0.1
10 * 0.05 = 0.5
...
```

## Data Layout and Packing

To store 4-bit integers efficiently, we pack them into `uint32` values. A standard `uint32` has 32 bits, so it can hold exactly 8 sets of 4-bit weights ($8 \times 4 = 32$).

In our implementation:
- **Group Size ($G$)**: 64
- **Bits**: 4
- **Packing**: 8 weights per `uint32`

The logical weight matrix $W$ (quantized) effectively uses the `uint32` buffer `B` with shape `[K, N // 8]`.

## Task 1: Implement CPU Kernel

```
src/extensions/src/quantized_matmul.cpp
```

You need to implement the `quantized_matmul_impl` function. This function performs the matrix multiplication $Y = X W^T$.

### Algorithm

The algorithm iterates through the input matrix $X$ and the quantized weight matrix $W^T$, unpacking weights on the fly.

```python
# Pseudo-code for quantized matmul
for i in range(M):           # Iterate rows of X
  for j in range(K):         # Iterate rows of B (columns of W)
    sum = 0
    # Iterate through groups (chunks of 64)
    for group_idx in range(groups_per_row):
       scale = scales[j, group_idx]
       bias = biases[j, group_idx]
       
       # Iterate through packed integers in the group
       for item_idx in range(items_per_group):
         packed = b[j, group_start + item_idx]
         
         # Unpack 8 weights from one uint32
         for pack_idx in range(8):
           w_quant = (packed >> (pack_idx * 4)) & 0xF
           w = w_quant * scale + bias
           
           x_val = x[i, input_col_idx]
           sum += x_val * w
           input_col_idx += 1
     
    out[i, j] = sum
```

1.  **Iterate** over rows of $X$ ($i$ from $0$ to $M$).
2.  **Iterate** over rows of $W^T$ (columns of $W$, $j$ from $0$ to $N$).
3.  **Inner Loop**: Compute the dot product row $i$ of $X$ and column $j$ of $W$.
    - The column $j$ of $W$ is quantized.
    - We iterate through groups (chunks of 64).
    - For each group:
        - Load `scale` and `bias`.
        - Load 8 weights at a time from the packed `uint32` array.
        - **Unpack**: Use bitwise shifts (`>>`) and masks (`& 0xF`) to extract the 4-bit values.
        - **Dequantize**: `w = val * scale + bias`.
        - **Multiply-Add**: `sum += x_val * w`.

This requires careful indexing because `b` is packed. Drawing a diagram of the memory layout (rows vs columns vs groups) helps significantly.

## Task 2: Handle Types

We need to support `float16`, `bfloat16`, and `float32`.
Use C++ templates to write a generic implementation `quantized_matmul_impl_typed<T>`.

## Testing

Run the tests to verify your implementation:

```bash
pdm run test-refsol tests_refsol/test_week_2_day_2.py
```

(Note: This test file covers both CPU and GPU. For today, focus on the CPU tests failing or passing).

{{#include copyright.md}}
