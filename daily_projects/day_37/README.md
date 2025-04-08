Day 37: Implementation of transformer decoder in CUDA

The code performs transformer decoder block in CUDA. The decoder consists of three main sub-layers:
  - Self‑Attention Block:
    - Linear Projections: The decoder input \( Y \) of shape \((M \times d_{model})\) is first linearly projected into queries, keys, and values using learned weights:
  
      ```math
      Q = Y \times W_{q}^{(\text{self})},\quad K = Y \times W_{k}^{(\text{self})},\quad V = Y \times W_{v}^{(\text{self})}
      ```
  
    - Splitting into Heads: The projected matrices are split into \( H \) heads, where each head has a dimension:
  
      $$
      d_{\text{head}} = \frac{d_{model}}{H}
      $$
  
    - Scaled Dot‑Product Attention: For each head, the attention scores are computed as:
  
      $$
      \text{score}[i,j] = \frac{Q[i, :] \cdot K[j, :]^T}{\sqrt{d_{\text{head}}}}
      $$
    
      A softmax is then applied over the scores to obtain the attention weights:
    
      $$
      \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_{\text{head}}}}\right) \times V
      $$
  
    - Final Projection: The outputs from all heads are concatenated and projected using \( W_{o}^{(\text{self})} \) to yield the self-attention output.

  - Encoder‑Decoder Attention Block: The queries come from the output of self‑attention after layer normalization, while the keys and values come from the encoder output \( \text{Enc} \).

    - Linear Projections: 
    
      $$
      Q = \text{LN}(\text{SelfAttentionOutput}) \times W_{q}^{(\text{encdec})}
      $$
      
      $$
      K = \text{Enc} \times W_{k}^{(\text{encdec})},\quad V = \text{Enc} \times W_{v}^{(\text{encdec})}
      $$
  
    - Scaled Dot‑Product Attention: The attention scores are computed as:
  
      $$
      \text{score}[i,j] = \frac{Q[i, :] \cdot K[j, :]^T}{\sqrt{d_{\text{head}}}}
      $$
  
  After applying softmax and computing the weighted sum of values, the result is projected with \( W_{o}^{(\text{encdec})} \).

- Feed‑Forward Network Block: The feed‑forward network performs a point‑wise non-linear transformation using two layers

  - First Layer (Expansion):

    $$
    \text{FFN}_1(x) = \text{ReLU}(x \times W_1)
    $$
    
    where the intermediate dimension is typically:
    
    $$
    d_{ff} = 4 \times d_{model}
    $$
  
- Second Layer (Projection):

    $$
    \text{FFN}(x) = \text{FFN}_1(x) \times W_2
    $$

- Residual and Normalization: Each block (attention and feed‑forward) is wrapped with a residual connection and layer normalization
  
  $$
  \text{Output} = \text{LayerNorm}(x + \text{SubLayer}(x))
  $$


- CUDA Kernels:
  - matmul_kernel: Implements tiled matrix multiplication. It is used for all linear projections and feed‑forward layers
  - compute_scores_kernel: Computes scaled dot‑product scores between queries and keys for a given head
  - softmax_kernel: Applies softmax over the key dimension using parallel reduction for numerical stability
  - weighted_sum_kernel: Computes the weighted sum over value vectors \( V \) using the softmax scores
  - add_kernel: Performs element‑wise addition for the residual connection
  - relu_kernel: Applies the ReLU activation function in-place
  - layer_norm_kernel: Computes the layer normalization (calculates mean and variance per token and normalizes features).

2) Implementation:

Compiling the code:  

<pre>nvcc .\decoder.cu -o decoder</pre>

Running the code after compiling: 

<pre> decoder </pre>

<pre>CPU Decoder Total Time: 1697 ms
GPU Decoder Host-to-Device Copy Time: 1.02646 ms
GPU Decoder Kernel Execution Time: 2.95792 ms
GPU Decoder Device-to-Host Copy Time: 0.182752 ms
Total GPU Decoder Time: 4.22301 ms</pre>