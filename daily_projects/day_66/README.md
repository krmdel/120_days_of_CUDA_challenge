Day 66: Implementation of one dimensional (1D) inverse discrete Fourier transform (DFT) in CUDA

1) Summary of the daily tutorial

The code implements one dimensional (1D) inverse discrete Fourier transform (DFT) in CUDA. It generates a random real-valued signal, x[n], (default length N = 16 384). It computes its forward DFT, then immediately performs the inverse DFT on both back-ends. 

  ```math
  X[k] \;=\; \sum_{n=0}^{N-1} x[n]\,e^{-j\,2\pi kn/N}
  ```

  ```math
  x[n] \;=\; \frac{1}{N}\sum_{k=0}^{N-1} X[k]\,e^{+j\,2\pi kn/N}
  ```
- CUDA kernels:
  - dft1d_kernel: Each thread computes one frequency bin, k. Accumulates the real input samples multiplied by cos() and sin() terms with a negative sign in the exponent.
  - dft1d_kernel: Each thread reconstructs one time-domain sample, k. Uses the positive sign in the exponent and divides the final result by N to finish the inverse transform.      |

2) Implementation

Compiling the code:

<pre>nvcc .\dft_1d.cu -o dft_1d</pre>

Running the code after compiling:

<pre>dft_1d</pre>

<pre>Vector length: 16384
CPU forward DFT : 6898.35 ms
CPU inverse DFT : 10217 ms

GPU H2D copy     : 0.024576 ms
GPU forward kern : 4.54451 ms
GPU forward D2H  : 3.75603 ms
GPU inverse kern : 3.75194 ms
GPU inverse D2H  : 0.11776 ms</pre>