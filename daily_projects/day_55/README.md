Day 55: Implementation of online softmax in CUDA

1) Summary of the daily tutorial

The code implements and compares two versions of a row-wise softmax in CUDA:  
- softmax_naive: uses a shared‐memory buffer per row to compute max, exponentials, sum and normalization in three passes.
- softmax_online: streams through each row in three register‐based passes (max, sum, write) and uses atomic operations on a few shared scalars, avoiding a full shared‐memory buffer.  

The softmax implemented is:  
```math
P_{ij} = \frac{\exp\bigl(S_{ij} - \max_{k} S_{ik}\bigr)}{\sum_{k=1}^{L} \exp\bigl(S_{ik} - \max_{k} S_{ik}\bigr)}
```

- CUDA kernels:  
  - softmax_naive: Launches one block per row, loads the entire row into shared memory, uses warp‐shuffle reductions to find the maximum, computes exponentials into shared memory, reduces to get the sum, then writes normalized probabilities back to global memory.  
  - softmax_online: Launches one block per row, performs three streaming passes purely in registers: (1) row‐max reduction, (2) sum of exp(x–max), (3) write exp(x–max)/sum, using 'atomicMax' and 'atomicAdd' on two shared floats to coordinate across warps.

2) Implementation

Compiling the code:

<pre>nvcc .\online_softmax.cu -o online_softmax</pre>

Running the code after compiling:

<pre>online_softmax</pre>

<pre>Naive softmax (L=2048)
  Host-to-Device copy time:      : 1.49053 ms
  Kernel execution time:         : 0.888224 ms
  Device-to-Host copy time:      : 1.70493 ms
  Total GPU time:                : 4.08368 ms

Online softmax (L=2048)
  Host-to-Device copy time:      : 1.46096 ms
  Kernel execution time:         : 0.16496 ms
  Device-to-Host copy time:      : 1.47155 ms
  Total GPU time:                : 3.09747 ms</pre>