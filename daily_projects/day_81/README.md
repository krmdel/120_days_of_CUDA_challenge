Day 81: Implementation of comparisons of FFT algorithm for N-D signals in CUDA

1) Summary of the daily tutorial

The code implements benchmarking for discrete-Fourier-transform (DFT) on volumetric data in N-D. Starting from a naïve space-domain DFT, the code runs radix-2, mixed-radix (2/3/5), split-radix, Bluestein’s chirp-Z transform, and finally NVIDIA’s optimized cuFFT.

For every method the forward transform ultimately evaluates the standard 1-D DFT

```
math
X[k] \;=\; \sum_{n=0}^{N-1} x[n]\,e^{-j\,2\pi kn/N},
\qquad k = 0,\dots,N-1
```

and extends it to higher dimensions by applying the transform separably along the X-, Y- and Z-axes.

- CUDA kernels:
	- transpose32: cache-friendly 32\times32 tiled transpose used to switch between row-major and column-major data orders inside the separable 3-D FFT.
	- bitrev: performs the bit-reversal permutation required before radix-2 butterflies.
	- radix2_stage: one FFT stage of length-2^s butterflies.
	- mixed_kernel: handles radix-3 and radix-5 butterflies so any length N = 2^a3^b5^c is covered.
	- split_post: final recombination step that converts a pure radix-2 FFT into a split-radix FFT (better twiddle-count).
	- Bluestein:uite
	- blu_build: multiplies input by the chirp and prepares two convolution vectors
	- blu_pad: – zero-pads & mirrors the chirp so the convolution can be done with a power-of-two FFT
	- blu_mul: – element-wise product in the frequency domain
	- blu_final: post-scales and removes the chirp to recover the Bluestein result
	- Spatial:FT kernels
	- dft2d_naive_kernel: textbook double loop over every pixel for reference timings
	- dft2d_tiled_slice:– shared-memory/tiling version that re-uses sines & cosines to reduce arithmetic

2) Implementation

Compiling the code:

<pre>nvcc .\all_3d_fft.cu -o all_3d_fft</pre>

Running the code after compiling:

<pre>all_3d_fft</pre>

<pre>Volume 256×256×32 (2097152 voxels)

Naïve DFT         Σ|X|=1.383682e+08  H2D 1.978560 Kern 913.739563 D2H 3.687104 Total 919.405273 ms
Shared DFT         Σ|X|=1.383682e+08  H2D 1.880448 Kern 203.291611 D2H 3.951872 Total 209.123932 ms
Radix-2 FFT        Σ|X|=7.778622e+08  H2D 1.681856 Kern 3618.178223 D2H 0.089088 Total 3619.949219 ms
Mixed-radix FFT    Σ|X|=7.778622e+08  H2D 1.679360 Kern 3625.981934 D2H 0.169984 Total 3627.831299 ms
Split-radix FFT    Σ|X|=1.778247e+09  H2D 1.677312 Kern 4100.900879 D2H 0.052224 Total 4102.630371 ms
Bluestein FFT      Σ|X|=7.778618e+08  H2D 1.709056 Kern 36443.343750 D2H 0.138240 Total 36445.191406 ms
cuFFT 3-D          Σ|X|=7.778620e+08  H2D 3.467488 Kern 0.073728  D2H 0.000000 Total 3.541216 ms</pre>