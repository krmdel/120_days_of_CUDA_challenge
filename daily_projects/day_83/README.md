Day 83: Implementation of finite impulse response (FIR) filter with shared memory and tiling in CUDA

1) Summary of the daily tutorial

The code benchmarks two implementations of a 1-D FIR filter and times them against each other:
- Naïve version: every thread reads all its required signal samples directly from global memory.
- Shared-memory tiled version: threads in a block first cooperatively stage a tile of the signal (plus the necessary halo of T-1 samples) into shared memory, then reuse those data to compute all outputs in the tile.

Because redundant global loads are eliminated, the tiled kernel is expected to deliver noticeably higher throughput once the tap count is large enough to amortise the extra shared-memory traffic.

Mathematically, the filter performs a causal convolution of the input signal x[n] with a tap vector h[k]:

```math
y[n] \;=\; \sum_{k=0}^{T-1} h[k]\,x[n-k], \qquad n = 0,\dots,N-1
```

Here T is the number of taps (filter length) and N is the signal length.
The taps are copied once into constant memory so that every thread can broadcast-read them efficiently.

- CUDA kernels:
	- fir_naive_kernel: One thread → one output sample. It reads every needed input sample directly from global memory, uses loop‐unrolling to issue four MACs per iteration.
	- fir_tiled_kernel: One thread block processes TILE consecutive outputs. Co-operative load: all threads first read TILE + T − 1 input samples (tile + halo) into shared memory. Each thread then re-uses the cached window to accumulate its output, issuing eight MACs per unrolled iteration. Tap coefficients are fetched from constant memory, inputs from shared memory → far fewer global transactions.

2) Implementation

Compiling the code:

<pre>nvcc .\fir_shared.cu -o fir_shared</pre>

Running the code after compiling:

<pre>fir_shared</pre>

<pre>Signal length N = 1048576   taps = 255

Naïve (global)      H2D 1.056   Kern 0.217     D2H 2.864   Total 4.137 ms
Shared-mem tile      H2D 1.056   Kern 0.842     D2H 2.103   Total 4.001 ms</pre>