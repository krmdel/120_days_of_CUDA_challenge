Day 80: Implementation of comparisons of FFT algorithm for 2D signals in CUDA

1) Summary of the daily tutorial

The code benchmarks seven different ways to compute a 2-D discrete Fourier transform (DFT) on the GPU. Space-domain DFTs: a naïve implementation and an optimized shared-memory/tiled version. Custom 1-D FFT engines: radix-2, mixed-radix (2/3/5), split-radix, and Bluestein (Chirp-Z) that are turned into 2-D FFTs by processing rows, transposing, then processing columns. cuFFT: NVIDIA’s reference library for comparison. Then ℓ₁-norm of each spectrum together with the individual and total timings are printed, making it easy to see both correctness and performance.

The reference operation of the 2-D DFT is as follows:

```math
\begin{aligned}
F(u,v) &= \!\sum_{y=0}^{H-1}\sum_{x=0}^{W-1} f(x,y)\,
        e^{-j\,2\pi\!\bigl(\tfrac{u x}{W}+\tfrac{v y}{H}\bigr)}
\end{aligned}
```

where f(x,y) is the spatial pixel value and (u,v) is the frequency bin.

- CUDA kernels:
	- transpose32: 32 × 32 tiled matrix transpose with an extra column to avoid shared-memory bank conflicts. Used to swap axes between the row and column FFT passes.
	- bitrev / radix2_stage: perform bit-reversal permutation and the butterfly for each radix-2 stage; together they constitute the in-place radix-2 Cooley–Tukey engine.
	- mixed_kernel: unified butterfly for radix-3 and radix-5 passes; enables a mixed-radix decomposition free of large prime factors.
	- split_post: post-processing step that converts a radix-2 FFT into a split-radix FFT, shaving a few extra FLOPs.
	- dft2d_naive_kernel: assigns one thread to every output bin and computes the double sum directly; simple but O(N²).
	- dft2d_tiled_kernel: loads 16 × 16 image tiles into shared memory, re-uses the data for all frequencies that see the tile, and uses incremental sin/cos recursion to cut down transcendental calls.
	- fft2d_generic: generic 2-D wrapper: run chosen 1-D FFT on rows → transpose → run on columns → transpose back.

2) Implementation

Compiling the code:

<pre>nvcc .\all_2d_fft.cu -o all_2d_fft</pre>

Running the code after compiling:

<pre>all_2d_fft</pre>

<pre>Image 256×256 (65536 px)

Naïve DFT         Σ|X|=4.317954e+06  H2D 0.109216 Kern 34.318336 D2H 0.180832 Total 34.608383 ms
Shared DFT         Σ|X|=4.317956e+06  H2D 0.095552 Kern 6.429632 D2H 0.153056 Total 6.678240 ms
Radix-2 FFT        Σ|X|=4.317955e+06  H2D 0.161472 Kern 13.922272 D2H 0.000000 Total 14.083744 ms
Mixed-radix FFT    Σ|X|=4.317955e+06  H2D 0.180640 Kern 13.893248 D2H 0.000000 Total 14.073888 ms
Split-radix FFT    Σ|X|=7.537057e+06  H2D 0.170912 Kern 15.413600 D2H 0.000000 Total 15.584512 ms
Bluestein FFT      Σ|X|=4.317954e+06  H2D 0.159296 Kern 146.914780 D2H 0.000000 Total 147.074081 ms
cuFFT (library)    Σ|X|=4.317955e+06  H2D 0.159520 Kern 0.018432 D2H 0.137824 Total 0.315776 ms</pre>