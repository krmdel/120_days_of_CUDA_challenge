Day 44: Implementation of text encoder in CUDA

1) Summary of the daily tutorial

The code implements text encoder in CUDA. The text encoder takes a sequence of token embeddings and processes them through a multi-head attention mechanism followed by a feed-forward network. Throughout the pipeline, residual connections and layer normalization are applied to improve training stability and performance. The code includes CUDA kernels for operations such as tiled matrix multiplication, scaled dot‑product attention, softmax normalization, and activation functions

- Multi-head attention:
  - Linear projections: The input token embeddings are transformed into query (Q), key (K), and value (V) matrices using tiled matrix multiplication kernels.  
  - Scaled dot-product attention: For each head, the attention scores are computed using the dot product of Q and the transpose of K. The scores are then scaled by the inverse square root of the head dimension and normalized via a softmax function. This is expressed as:  

    ```math
    \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^{T}}{\sqrt{d_{head}}}\right)V
    ```  

  - Weighted Aggregation: The normalized scores are used to compute a weighted sum of the value vectors, producing the attention output for each head.
  
  - Feed-Forward Network (FFN):
    - A two-layer feed-forward network is applied to the token representations after the multi-head attention branch. A ReLU activation function introduces non-linearity between the layers.  
    - In typical transformer fashion, the FFN is defined as:  

      ```math
      \text{FFN}(x) = \text{ReLU}(xW_{1})W_{2}
      ```

  - Residual Connections and Layer Normalization:
    - Residual Connections: The results of the multi-head attention and the FFN branches are added back to their corresponding inputs.  
    - Layer Normalization: Each token's feature vector is normalized by computing its mean and variance, then scaling appropriately. This operation can be summarized as:  
      ```math
      \text{output}[i] = \text{LayerNorm}\big(x[i] + \text{FFN}(x[i])\big)
      ```

- CUDA Kernels:

  - matmul_kernel: Performs tiled matrix multiplication to efficiently compute the linear projections (for Q, K, and V) as well as the feed-forward layers. Shared memory is used to load tiles of the input matrices to reduce global memory access

  - compute_scores_kernel: Calculates the scaled dot‑product attention scores for each attention head by multiplying segments of the query and key matrices. The kernel implements tiling to efficiently compute the dot product over the head dimension

  - softmax_kernel: Applies the softmax operation along the keys’ dimension. Each block of this kernel is responsible for one (query, head) pair and uses shared memory to perform parallel reduction for both the maximum value and the sum of exponentials

  - weighted_sum_kernel: Aggregates the value vectors with the previously computed attention scores to produce the final attention output for each head.

  - add_kernel: Executes element-wise addition to implement residual connections across different layers of the encoder

  - relu_kernel: Applies the ReLU activation function to introduce non-linearity in the feed-forward network

  - layer_norm_kernel: Performs layer normalization by computing the mean and variance for each token’s feature vector and then normalizing the output. Each block processes one token (i.e., one row) to parallelize the operation.

2) Implementation

Compiling the code:

<pre>nvcc .\text_encoder.cu -o text_encoder</pre>

Running the code after compiling:

<pre>text_encoder</pre>

<pre>Total CPU inference time (ms): 490 ms
GPU Timings (ms):
  Host-to-Device copy time: 0.584416
  Kernel execution time:    1.7967
  Device-to-Host copy time: 0.103104
  Total GPU inference time: 2.5311</pre>