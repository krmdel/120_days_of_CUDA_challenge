Day 110: Implementation of one-warp HOG cell histogram in CUDA

1) Summary of the daily tutorial

The code implements 9-bin Histogram-of-Oriented-Gradients (HOG) cell histograms for a large gray-scale image (default 4096 × 4096).
- Each 8 × 8 pixel cell is processed by a single warp (32 threads).
- Every thread reads up to two pixels, accumulates its own 9-element vector of gradient-magnitude votes, then a warp-wide reduction fuses these into the final cell histogram.


- CUDA kernels:
  - warpReduceSum: Reduces a scalar across the 32 threads of a warp using __shfl_down_sync. 
  - cellHistKernel: Builds one 9-bin HOG histogram for every 8 × 8 cell. One block = one warp = one cell, each thread handles ≤2 pixels (64 pixels / 32 threads). Accumulates local h[9], then reduces each bin with warpReduceSum. Thread 0 writes the 9-value result to global memory.

2) Implementation

Compiling the code:

<pre>nvcc .\hog_cell.cu -o hog_cell</pre>

Running the code after compiling:

<pre>hog_cell</pre>

<pre>Random 4096×4096 image
CPU total               : 45.5322 ms
GPU H2D copy            : 29.0096 ms
GPU kernel              : 0.542976 ms
GPU D2H copy            : 1.9553 ms
GPU total               : 35.8628 ms
RMSE  histograms        : 0</pre>