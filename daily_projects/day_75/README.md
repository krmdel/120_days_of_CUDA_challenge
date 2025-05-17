Day 75: Implementation of Cooley–Tukey FFT variants in CUDA

1) Summary of the daily tutorial

The code performs the benchmarking five different Cooley–Tukey fast-Fourier-transform (FFT) engines on the same real-valued signal of length SIZE (default 16 384 samples).  

Each engine implements a classical variant that trades arithmetic count, memory traffic and access patterns in different ways:
- Engine 0 - iterative radix-2 DIF:baseline power-of-two FFT that performs log₂ N stages of radix-2 butterflies after an in-place bit-reversal shuffle.
- Engine 1 - mixed-radix 2/3/5:accepts any 2·3·5-smooth length by chaining tiny radix-5 and radix-3 butterflies before falling back to radix-2.
- Engine 2 - split-radix:combines a radix-2 core with a post-processing step that re-uses twiddle factors, giving the lowest known FLOP count for power-of-two N.
- Engine 3 - four-step Stockham An:out-of-place, stride-1 algorithm that alternates ping–pong buffers every stage to ensure fully coalesced accesses.
- Engine 4 - six-step:introduces two blocked transposes so that every radix-2 sub-FFT works on contiguous data, ideal for large transforms that exceed on-chip cache.

Every butterfly computes:

```math
\begin{aligned}
X[k]        &= A + W_N^k \, B,\\
X[k+\tfrac{N}{2}] &= A - W_N^k \, B,
\end{aligned}
```

with the twiddle factor:

```math
W_N^k = e^{-j \, 2\pi k / N}
```

- CUDA kernels:
	- bitrev: in-place bit-reversal permutation used by the radix-2 and split-radix engines.
	- radix2_stage: one stage of the iterative radix-2 decimation-in-frequency algorithm; each block processes many butterflies.
	- mixed_kernel : erforms either radix-3 or radix-5 butterflies (selected by a template parameter) across the current stride.
	- split_post: final twiddle-saving shuffle that converts the radix-2 output into the split-radix ordering.
	- stockham: single stage of the four-step Stockham FFT; alternates between ping and pong buffers to keep stride-1 memory access.
	- transpose: 32 × 32 tiled transpose with bank-conflict-free shared memory, used twice by the six-step algorithm.

2) Implementation

Compiling the code:

<pre>nvcc .\cooley_tukey_fft.cu -o cooley_tukey_fft</pre>

Running the code after compiling:

<pre>cooley_tukey_fft</pre>

<pre>Signal length: 16384
Engine 0 (Iterative radix-2) : 8.02154 ms
Engine 1 (Mixed-radix 2/3/5) : 0.003968 ms
Engine 2 (Split-radix) : 0.003936 ms
Engine 3 (Four-step Stockham) : 0.012224 ms
Engine 4 (Six-step) : 0.630464 ms</pre>