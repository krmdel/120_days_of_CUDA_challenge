Day 78: Implementation of Goertzel filter bank for sparse FFT in CUDA

1) Summary of the daily tutorial

The code implements Goertzel algorithm, an efficient way to evaluate only a small set of DFT bins without computing a full FFT. Instead of O(N.log N) work for every bin, each target frequency k is realised as a second-order IIR filter that streams through the signal once. One thread-block per requested bin is launched: all threads in the block cooperatively run the Goertzel recurrence across the whole signal, then finish with a cheap post-processing step to obtain the complex DFT coefficient.

The Goertzel recurrence and final bin value are:

```math
\begin{aligned}
s_n &= x[n] + 2\cos\!\bigl(\tfrac{2\pi k}{N}\bigr)\,s_{n-1} - s_{n-2},\qquad n = 0,\dots,N-1.
\end{aligned}
```

```math
\begin{aligned}
X[k] \;=\; s_{N-1} \;-\; e^{-j\,\frac{2\pi k}{N}}\,s_{N-2}
\end{aligned}
```

- CUDA kernels:
	- goertzel_kernel: Computes one DFT bin X[k] with the Goertzel algorithm. Thread layout is one block per one frequency bin and one thread per strided slices of the input signal.
	
	Each thread updates its private s_{n-1},s_{n-2} over its slice. Warp–shuffle reductions sum partial states inside the warp. Two atomicAdd operations accumulate warp sums into shared memory variables s_prev, s_prev2. Thread 0 applies the final Goertzel formula and writes the complex result. Shared scalars are reset so the same block can be reused in a subsequent launch.
	
2) Implementation

Compiling the code:

<pre>nvcc .\goertzel_fft.cu -o goertzel</pre>

Running the code after compiling:

<pre>goertzel</pre>

<pre>Sparse FFT (Goertzel bank)  N=16384  K=64
Host → Device copy : 0.048128 ms
Goertzel kernel    : 0.201728 ms
Device → Host copy : 0.015232 ms
Total GPU time     : 0.265088 ms</pre>