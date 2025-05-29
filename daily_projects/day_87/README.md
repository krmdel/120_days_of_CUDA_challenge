Day 87: Implementation of cross correlation for comparison of time domain and FFT with peak detection in CUDA

1) Summary of the daily tutorial

The code implements for measuring the cross correlation between two real 1-D signals in two different ways:
- Time-domain (“naïve”) method — O(N²): Each possible lag \tau is evaluated directly by summing element-wise products.
- Frequency-domain method — O(N log N): By the convolution theorem, the cross-correlation is the inverse FFT of the product of one signal’s spectrum and the complex conjugate of the other’s.

After either path can be located the peak to obtain the best alignment between the two signals.

Mathematically, the cross-correlation sequence of two length-N real signals x[n] and y[n] is

```math
c[\tau] \;=\; \sum_{n=0}^{N-1} x[n]\,y[n+\tau], 
\qquad \tau = -(N-1),\dots,(N-1)
```

Using the FFT, the same result is obtained by

```math
\mathcal{F}^{-1}\!\bigl\{\,X[k]\;\overline{Y[k]}\,\bigr\}
```

where X and Y are the forward FFTs of x and y.

- CUDA kernels:

- xcorr_naive:	Computes the time-domain cross-correlation. Each thread owns one lag, tau, and accumulates the inner product for that lag.
- cmul_conj: Performs element-wise complex multiplication. One thread per frequency bin.
- scaleCopyKernel:	Copies the first M=2N-1 real parts of the inverse FFT result to the output buffer and scales by 1 / FFT_size.

2) Implementation

Compiling the code:

<pre>nvcc .\cross_corr.cu -o cross_corr</pre>

Running the code after compiling:

<pre>cross_corr</pre>

<pre>Signal length N           : 262144
Correlation sequence size : 524287
FFT size (next pow-2)      : 524288


               H2D      Kern        D2H     Total
Time-domain   H2D 0.745   Kern 166.108   D2H 0.587   Total 167.440 ms
FFT           H2D 0.745   Kern 0.120     D2H 0.524   Total 1.388 ms</pre>