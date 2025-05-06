Day 65: Implementation of one dimensional (1D) discrete Fourier transform (DFT) in CUDA

1) Summary of the daily tutorial

The code implements a baseline 1-D DFT that runs both on the CPU (single-thread reference) and the GPU, measuring and comparing their execution times. GPU kernel assigns one output frequency bin, k, to each CUDA thread. Every thread iterates over the full time-domain signal to accumulate the real and imaginary parts of its spectrum coefficient: X[k].

  ```math
  X[k] \;=\; \sum_{n=0}^{N-1} x[n]\;e^{-j\,\tfrac{2\pi}{N}kn},
  \qquad k = 0,\dotsc,N-1
  ```

- CUDA kernel:
  - dft1d_kernel:Launches configuration: 1-D grid, `THREADS` threads per block and each thread computes a single coefficient X[k]:

    ```math
    \begin{aligned}
    \Re\{X[k]\} &\!+= x[n]\cos\!\bigl(-\tfrac{2\pi}{N}kn\bigr) \\
    \Im\{X[k]\} &\!+= x[n]\sin\!\bigl(-\tfrac{2\pi}{N}kn\bigr)
    \end{aligned}
    ```

  - Uses `sincosf` to fetch `sin` and `cos` in one instruction and writes the result back as a `cuFloatComplex`.

2) Implementation

Compiling the code:

<pre>nvcc .\dft_1d.cu -o dft_1d</pre>

Running the code after compiling:

<pre>dft_1d</pre>

<pre>1-D signal length: 16384

CPU DFT     : 8993.941 ms
GPU H2D     : 0.033 ms
GPU Kernel  : 4.923 ms
GPU D2H     : 0.127 ms
GPU Total   : 5.083 ms</pre>