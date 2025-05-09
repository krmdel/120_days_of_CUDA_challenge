Day 68: Implementation of one dimensional (1D) discrete Fourier transform (DFT) in CUDA

1) Summary of the daily tutorial


2) Implementation

Compiling the code:

<pre>nvcc .\flash_sam.cu -o flash_sam</pre>

Running the code after compiling:

<pre>flash_sam</pre>

<pre>Image size: 128 x 128 (16384 pixels)

CPU DFT     : 4787.070 ms
GPU H2D     : 1.132 ms
GPU Kernel  : 8.352 ms
GPU D2H     : 0.054 ms
GPU Total   : 9.537 ms</pre>