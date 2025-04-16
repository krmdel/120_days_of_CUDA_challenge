Day 45: Implementation of cross model attention mechanism in CUDA

1) Summary of the daily tutorial

This tutorial presents the integration and optimization of a cross‑modal attention mechanism implemented in CUDA. In this implementation, visual features (such as image patch embeddings) and text features (from a language encoder) are fused using a multi‑head scaled dot‑product attention mechanism. The approach first projects visual and text features into a common embedding space using learned weight matrices. It then splits the projected tokens into multiple heads and computes attention scores that measure the correspondence between visual queries and textual keys. The normalized attention scores are used to obtain weighted sums from text value representations, which are finally merged through a linear projection to generate the output.

The computation is based on:

  ```math
  \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_{head}}}\right)V
  ```

- CUDA kernels:
  
  - matmul_kernel: Performs tiled matrix multiplication using shared memory. Multiplies two matrices (e.g., for the linear projections (Q = vis x W_q, K = text x W_k), and (V = text x W_v)) by loading sub-tiles into shared memory for efficient reuse.
  - compute_scores_kernel: Computes the scaled dot-product scores between visual queries and text keys. For each attention head, h, it calculates the dot product between the corresponding slices of Q (from visual features) and K (from text features) for each visual-text token pair, and then scales the result by 1 / d_head.
  - softmax_kernel: Normalizes the computed attention scores via a row‑wise softmax. For each pair of visual token and attention head, the kernel uses shared memory for a reduction to compute the maximum value and the sum of exponentials, then applies the softmax operation across all text tokens.
  - weighted_sum_kernel: Computes the weighted sum over the text value vectors. Uses the normalized attention scores and the value matrix, V, to produce the attention output for each visual token and head. For every token, it sums the product of the score and the corresponding text feature value over all text tokens.
  - add_kernel: Performs an element‑wise addition of two input tensors. Though defined in the code for completeness, this kernel can be used for combining intermediate outputs (e.g., adding bias terms or residuals).
  - relu_kernel: Applies the ReLU (Rectified Linear Unit) activation function element‑wise. Ensures non-linearity by setting negative values in the feature map to zero.
  - layer_norm_kernel: Performs layer normalization over each token (i.e., row) of the output matrix. Computes the mean and variance for each token, and normalizes the token’s features to stabilize the training and improve convergence.

2) Implementation

Compiling the code:

<pre>nvcc .\cross_modal_attention.cu -o cross</pre>

Running the code after compiling:

<pre>cross</pre>

<pre>Total CPU inference time (ms): 657
GPU Timings (ms):
  Host-to-Device copy time: 0.429408
  Kernel execution time:    1.83171
  Device-to-Host copy time: 0.278272
  Total GPU time:           2.57667</pre>