Day 54: Implementation of Q·Kᵀ via I/O-aware tiling in CUDA

1) Summary of the daily tutorial

The code implements two ways to compute the attention score matrix  

```math
S_{i,j} = \sum_{k=0}^{d-1} Q_{i,k} \cdot K_{j,k}
```

The naive kernel assigns one thread to each (i,j) entry of S, reading all d elements of Q and K from global memory for every dot product. The I/O-aware tiled kernel stages each query row once into shared memory and streams blocks of the key matrix into registers, dramatically reducing redundant global‐memory traffic and improving arithmetic intensity.

  - matmul_naive  
    Launches a 2D grid of threads where each thread computes one S_{i,j} by looping over k from 0 to (d-1).  
  - matmul_tiled  
    Divides the work into tiles of size BLOCK_M * BLOCK_N. Each block:
      1. Stages one query row of length d into shared memory once.
      2. Iterates over key blocks of width BLOCK_K, loading each small chunk into registers.
      3. Performs an inner unrolled dot-product over that chunk before moving to the next.

  - Baseline kernel (`matmul_naive`)  
    Computes each element of S independently—simple but poor data reuse.
  - Tiled kernel (`matmul_tiled`)  
    Implements an I/O-aware tiling strategy:
    1. Prefetch a full query row into shared memory.
    2. Stream key-vector chunks into registers.
    3. Accumulate partial dot-products in registers.
    4. Write back the final result to global memory.

- Dot-product attention:

  ```math
  S_{i,j} = \sum_{k=0}^{d-1} Q_{i,k} \cdot K_{j,k}
  ```

- CUDA kernels:
  - matmul_naive  
    A straightforward, one-thread-per-output approach. Good for clarity, but each thread redundantly reads the same query and key data.
  - matmul_tiled  
    An I/O-aware implementation that uses:
    - "extern __shared__" memory to cache one query row per block.
    - Thread‐local registers to hold blocks of the key row.
    - Loop unrolling over the inner dimension BLOCK_K for maximum throughput.

2) Implementation

Compiling the code:

<pre>nvcc .\tiled_qk.cu -o tiled_qk</pre>

Running the code after compiling:

<pre>tiled_qk</pre>

<pre>Naive kernel (L=512, d=64)
  Host-to-Device copy time:      : 0.115424 ms
  Kernel execution time:         : 1.05386 ms
  Device-to-Host copy time:      : 0.211872 ms
  Total GPU time:                : 1.38115 ms

Tiled kernel  (BLOCK_M=N=32, BLOCK_K=64)
  Host-to-Device copy time:      : 0.067136 ms
  Kernel execution time:         : 0.05328 ms
  Device-to-Host copy time:      : 0.193312 ms
  Total GPU time:                : 0.313728 ms</pre>