Day 36: Implementation of transformer encoder in CUDA

1) Summary of the daily tutorial:

This tutorial demonstrates a complete transformer encoder block implemented in CUDA. The encoder consists of two main sub-layers: the multi‑head attention layer and the position‑wise feed‑forward network, each followed by a residual connection and layer normalization.

- Multi‑Head Attention:
  - Linear Projections: The input matrix \(X\) (of shape \([M, d_{model}]\)) is projected into queries \(Q\), keys \(K\), and values \(V\) using learned weight matrices \(W_q\), \(W_k\), and \(W_v\). This is computed as:
  
  ```math
  Q = X \times W_q,\quad K = X \times W_k,\quad V = X \times W_v
  ```
  - Splitting into Heads: The projected matrices are divided into \(H\) heads, where each head has a dimension:
  
  ```math
  d_{head} = \frac{d_{model}}{H}
  ```

  - Scaled Dot‑Product Attention: For each head, the attention scores are computed by taking the dot product between queries and keys, and then scaling by \( \frac{1}{\sqrt{d_{head}}} \):
  
  ```math
  \text{score}[i,j] = \frac{Q[i,:] \cdot K[j,:]^T}{\sqrt{d_{head}}}
  ```
  
A softmax is applied over these scores to obtain the attention weights, which are then used to compute a weighted sum of the values \(V\). Finally, the outputs from all heads are concatenated and projected with a final weight matrix \(W_o\) to yield the attention output.

- Feed‑Forward Network:

  - Residual Connection and Normalization: The attention output is added to the original input \(X\) (residual connection) and then normalized via layer normalization.
  - Two‑Layer Feed‑Forward Network: The normalized output is then passed through a two‑layer feed‑forward network:
    - The first layer projects the data from \(d_{model}\) to a higher dimension \(d_{ff}\) (commonly \(d_{ff} = 4 \times d_{model}\)) and applies a ReLU activation.
    - The second layer projects the result back to \(d_{model}\).
    Another residual connection is applied, followed by a second layer normalization, to obtain the final encoder output.

- CUDA Kernels:
  - matmul_kernel: Implements tiled matrix multiplication. It is used both for computing the linear projections (i.e., \(Q\), \(K\), \(V\)) and for the feed‑forward layers. Tiling helps to maximize data reuse in shared memory for increased performance.
  - compute_scores_kernel: Computes the scaled dot‑product between the query and key matrices for each head. The result is scaled by \(1/\sqrt{d_{head}}\) to stabilize gradients.
  - softmax_kernel: Applies the softmax function to the computed scores across the key dimension. It uses a parallel reduction to find the maximum value and to sum the exponentials, ensuring numerical stability.
  - weighted_sum_kernel: Uses the softmax scores to compute a weighted sum over the value vectors \(V\) for each head, producing the attention output for that head.
  - add_kernel: Performs element‑wise addition (used for residual connections).
  - relu_kernel: Applies the ReLU activation function in a parallel, in‑place fashion.
  - layer_norm_kernel: Computes the layer normalization per token (row-wise normalization). It calculates the mean and variance and then normalizes the input features accordingly.

2) Implementation:

Compiling the code:  

<pre>nvcc .\encoder.cu -o encoder</pre>

Running the code after compiling: 

<pre> encoder </pre>

<pre>CPU Total Time: 1108 ms
GPU Host-to-Device Copy Time: 0.727488 ms
GPU Kernel Execution Time: 2.28893 ms
GPU Device-to-Host Copy Time: 0.166688 ms
Total GPU Time: 3.22816 ms</pre>