Day 35: Implementation of multihead attention in CUDA

1) Summary of the daily tutorial:

The code performs multi‑head attention implementation in CUDA that computes scaled dot‑product attention using a tiled approach. The input matrix X (of shape [M, d_model]) is projected into queries (Q), keys (K), and values (V) using learned weight matrices Wq, Wk, and Wv. This is done via matrix multiplication. The projected Q, K, and V are split into H heads (with each head having dimension d_head = d_model/H). For each head, scaled dot‑product is computed and scaled by 1/sqrt(d_head):

```math
\text{score}[i,j] = \frac{Q[i,:] \cdot K[j,:]^T}{\sqrt{d_{head}}}
```

A softmax is applied over the key dimension to produce normalized attention weights. The softmax scores are then used to weight the corresponding V vectors to produce each head’s output. Finally, the outputs from all heads (concatenated along the feature dimension) are further projected by multiplying with a final weight matrix Wo to yield the final attention output.

- CUDA kernels:  
  - matmul_kernel: This kernel performs tiled matrix multiplication and is used for both the linear projections (to compute Q, K, and V) and the final projection (to compute the output after attention)
  - compute_scores_kernel: For each head, it computes the dot product between the query and key tiles, scales the result by 1/√(d_head), and stores the score
  - softmax_kernel: For each (query, head) pair, it performs a parallel reduction to compute the softmax over the keys
  - weighted_sum_kernel: Using the softmax scores, it computes a weighted sum over V to produce the output of each head

2) Implementation:

Compiling the code:  

<pre>nvcc .\multihead_attention.cu -o multihead</pre>

Running the code after compiling: 

<pre> multihead </pre>

<pre>CPU Total Time: 582 ms
GPU Host-to-Device Copy Time: 0.3128 ms
GPU Linear Projection Time (Q, K, V): 0.960864 ms
GPU Multi-Head Attention Time: 0.54272 ms
GPU Final Projection Time: 0.065472 ms
GPU Device-to-Host Copy Time: 0.111648 ms
Total GPU Time: 2.05101 ms</pre>