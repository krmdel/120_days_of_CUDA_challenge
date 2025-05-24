Day 82: Implementation of finite impulse response (FIR) filter in CUDA

1) Summary of the daily tutorial

The code implements 1-D FIR filter, assigning one thread per output sample. Given a discrete signal x[n] of length N and a tap (impulse-response) vector h[k] of length T, the filter produces an output signal y[n] by sliding the taps across the input and accumulating the weighted sum at every position, with zero-padding for the indices that fall before the start of the signal.

```math
y[n] \;=\; \sum_{k=0}^{T-1} h[k]\,x[n-k], \qquad n = 0,\dots,N-1
```

Because each output sample is independent, the GPU launches N threads so that every thread computes exactly one output value.

- CUDA kernels:
	fir_naive_kernel: One thread per sample, loads its global output index n and iterates over all taps T (unrolled by 4 for better ILP. It finally performs the multiply-accumulate if idx = n-k is non-negative (implements left-side zero-padding) and writes the accumulated result to y[n].

2) Implementation

Compiling the code:

<pre>nvcc .\fir.cu -o fir</pre>

Running the code after compiling:

<pre>fir</pre>

<pre>Naïve FIR  —  N = 1048576,  taps = 63

GPU H2D copy : 1.104 ms
GPU Kernel   : 0.155 ms
GPU D2H copy : 0.892 ms
GPU Total    : 2.150 ms</pre>