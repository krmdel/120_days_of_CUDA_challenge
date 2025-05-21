Day 79: Implementation of comparisons of FFT algorithm for 1D signals in CUDA

1) Summary of the daily tutorial

The code assembles five implementations of the discrete Fourier transform (DFT)/fast Fourier transform (FFT) and benchmarks them against NVIDIA’s cuFFT on the same one dimensional random‐real signal. It highlights how different algorithmic ideas map to CUDA, how memory-traffic patterns change with each revision, and what speed-ups you gain from progressively more sophisticated kernels.

Mathematically, all versions compute the 1-D DFT

```math
\begin{aligned}
X[k] &= \sum_{n=0}^{N-1} x[n]\,e^{-j\,2\pi kn/N},\\
&\qquad k = 0,\dots,N-1.
\end{aligned}
```

- CUDA kernels:
    - dft_naive: one thread per output bin; direct O(N^2) summation.
    - dft_shared: tiles the input into shared memory to reuse data across threads, still O(N^2) but with fewer global loads.
    - mixed_kernel: butterfly stage for a mixed-radix 3 or 5 transform (called repeatedly by engine_mixed).
    - bitrev / radix2: bit-reverse permutation and radix-2 butterfly used by both the split-radix FFT and as a fall-back in the mixed-radix engine.
    - sr_post: post-processing stage that converts a pure radix-2 pass into a split-radix FFT.
    - blu_build, blu_pad, pointmul, blu_final – build, zero-pad, perform convolution via FFT and de-chirp for Bluestein/Chirp-Z-based FFT of arbitrary length.

2) Implementation

Compiling the code:

<pre>nvcc .\all_1d_fft.cu -o all_1d_fft</pre>

Running the code after compiling:

<pre>all_1d_fft</pre>

<pre>Vector length N = 16384

Naïve DFT   :  Σ|X|=1075708.934  H2D 0.044  Kern 2.299   D2H 0.052  Total 2.395 ms
Shared DFT  :  Σ|X|=1075708.934  H2D 0.039  Kern 1.870   D2H 0.049  Total 1.959 ms
Mixed-radix :  Σ|X|=1075715.519  H2D 0.050  Kern 0.088   D2H 0.108  Total 0.245 ms
Split-radix :  Σ|X|=1410286.666  H2D 0.047  Kern 0.067   D2H 0.066  Total 0.180 ms
Bluestein   :  Σ|X|=1075714.088  H2D 0.000  Kern 2.440   D2H 0.109  Total 2.550 ms
cuFFT       :  Σ|X|=1075715.365  H2D 0.000  Kern 0.468   D2H 0.109  Total 0.577 ms</pre>