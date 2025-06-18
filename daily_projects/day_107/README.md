Day 107: Implementation of zero overhead multiscale processing with cudaMallocAsync and memory-pools in CUDA

1) Summary of the daily tutorial

The code implements benchmark for a pyramid (multi-resolution) workflow that repeatedly:
- Builds four down-scaled copies of an image (4096 × 4096 → … → 512 × 512) on the host.
- For 100 runs, loops over the four scales and, on a single CUDA stream:
    - allocates device memory with cudaMallocAsync (drawn from a pre-configured memory pool),
    - moves the current level to the GPU,
    - launches a tiny kernel that sets every pixel to a constant value (42),
    - copies the result back, then frees the buffer with cudaFreeAsync.
- Records per-stage timings (allocation, H2D, kernel, D2H, total), prints means/σ after a warm-up, and compares against a CPU baseline that uses std::fill.

CUDA kernels:
    - fillKernel: writes a constant value v (here 42) into every pixel of a 2-D buffer. The kernel is memory-bandwidth bound and embarrassingly parallel.

Operation performed by fillKernel

```math
\text{out}[y,\,x] \;\leftarrow\; v
```

for every pixel coordinate (x, y) inside the current level.

2) Implementation

Compiling the code:

<pre>nvcc .\multiscale.cu -o multiscale</pre>

Running the code after compiling:

<pre>multiscale</pre>

<pre>Runs (cold+warm)            : 100
Alloc  mean ± σ  (warm)     : 0.3241 ± 0.0153 ms  (<- expect <0.05)
H2D   mean                  : 6.923 ms
Kernel mean                 : 0.118 ms
D2H   mean                  : 5.050 ms
GPU total mean              : 12.995 ms
CPU reference               : 2.114 ms</pre>