Day 56: Implementation of flash attention in CUDA

1) Summary of the daily tutorial

The code integrates and compares two ways of computing the standard scaled-dot-product attention in CUDA:  

- A "baseline" approach that splits the computation into three separate kernels: 'matmul_qk', 'softmax_inplace', and 'matmul_pv'
- A fused "FlashAttention" CTA-tiled kernel that loads queries, keys, and values into shared memory, performs the dot-product, softmax and value-projection in a single pass with online normalization for numerical stability, then writes out the output in one go.  

The overall operation implemented is:  
```math
\text{Attention}(Q,K,V) \;=\;\text{Softmax}\!\Bigl(\tfrac{Q\,K^\intercal}{\sqrt{d}}\Bigr)\,V
```

- CUDA kernels:  
  - matmul_qk: Computes the raw attention score matrix S = Q·Kᵀ via a straightforward double-loop matrix multiplication.  
  - softmax_inplace  
    Applies a row-wise softmax directly in shared memory using warp-shuffle reductions to find the max and normalize each row of S.  
  - matmul_pv: Multiplies the probability matrix (P = Softmax(S)) by the value matrix V to get the context vectors (O = PV).  
  - flash_attention_tiled:
    A CTA-tiled "FlashAttention" kernel that:  
    1. Loads a tile of Q once into registers/shared memory.  
    2. Streams through matching tiles of K and V, updating an online softmax (maintaining running max and sum) and accumulating the weighted sum into an output register buffer.  
    3. Writes the fused output in a single coalesced write.

2) Implementation

Compiling the code:

<pre>nvcc .\flash_attention.cu -o flash_attention</pre>

Running the code after compiling:

<pre>flash_attention</pre>

<pre>Baseline (L=512, d=64)
  H2D   : 0.188544 ms
  Kernels: 1.40992 ms
  D2H   : 0.094144 ms
  Total : 1.69261 ms

FlashAttention-CTA (L=512, d=64)
  H2D   : 0.121216 ms
  Kernel: 0.048672 ms
  D2H   : 0.047072 ms
  Total : 0.21696 ms</pre>