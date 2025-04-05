Day 34: Implementation of scaled dot-product attention in CUDA

1) Summary of the daily tutorial:

The code performs a scaled dot-product attention mechanism on CUDA. The dot product between the query (Q) and key (K) matrices are calculated and scaled the result by 1/sqrt(d_k), where d_k is the dimension of the key vectors. 

- Attention kernel:
  scaled_dot_product_kernel: The CUDA kernel computes the dot product QK^T using a tiled approach. Sub-matrices (tiles) of Q and K are loaded into shared memory to improve performance. Each output element is calculated as:

```math
C[i,j] = \frac{1}{\sqrt{d}} \sum_{k=0}^{d-1} Q[i,k] \cdot K[j,k]
```  

Here, 𝑄 has dimensions [𝑀×𝑑] and 𝐾 has dimensions [𝑁×𝑑]; note that 𝐾 is transposed during the multiplication. The computed dot products are scaled by 1/sqrt(d) to maintain stability.

2) Implementation:

Compiling the code:  

<pre>nvcc .\attention.cu -o attention</pre>

Running the code after compiling: 

<pre> attention </pre>

<pre>CPU Inference Time: 973 ms
GPU Host-to-Device Copy Time: 0.51392 ms
GPU Kernel Execution Time: 1.67027 ms
GPU Device-to-Host Copy Time: 0.806624 ms
Total GPU Inference Time: 3.0335 ms</pre>