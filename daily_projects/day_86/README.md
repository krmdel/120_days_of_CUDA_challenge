Day 86: Implementation of moving average by parallel prefix sum in CUDA

1) Summary of the daily tutorial

The code implements the computation of a sliding (moving) average over a very long 1-D signal.
- Inclusive prefix-sum (scan) transforms the raw signal x[i] into a cumulative array
```math
\text{pref}[i] \;=\;\sum_{k=0}^{i} x[k].
```
- A windowed average of width W is then a pairwise difference of two prefix values:
``` math
\text{avg}[i] \;=\; 
  \frac{ \text{pref}[i] \;-\; \bigl(i \ge W \;?\; \text{pref}[i-W] \;:\; 0\bigr) }{\,W}.
```

By breaking the problem into these two steps, it can be:
- parallelised the expensive prefix-sum with a custom kernel,
- kept the moving-average kernel embarrassingly parallel (one thread per element).

- CUDA kernels:
    - prefix_sum_naive: single-block, single-thread reference prefix scan (O N), easy to verify correctness but not fast, demonstrates how a serial scan works before optimising.
    - moving_avg_kernel: launches N / THREADS blocks, one thread per output sample, converts prefix sums into final moving-average values using the formula above, handles the left edge (i < W) without branch divergence.

2) Implementation

Compiling the code:

<pre>nvcc .\moving_avg_scan.cu -o moving_avg_scan</pre>

Running the code after compiling:

<pre>moving_avg_scan</pre>

<pre>Samples 1048576, window 64

Max |CPU - naïve| = 0

Max |CPU - thrust| = 5.72205e-06

                       H2D      Kern      D2H     Total
Naïve (1-thread)  H2D 1.816   Kern 38.344   D2H 0.894   Total 41.054 ms
Thrust scan        H2D 1.019   Kern 0.325    D2H 0.857   Total 2.201 ms</pre>