Day 77: Implementation of Bluestein/Chirp-Z in CUDA

1) Summary of the daily tutorial

The code implements Bluestein (Chirp-Z) algorithm for computing an arbitrary-length DFT using only convolutions that are power-of-two in size—so highly-optimised FFT libraries (cuFFT) can be reused, even when the transform length N is not a power of two.
The code turns the 1-D DFT:

```math
X[k] \;=\; \sum_{n=0}^{N-1} x[n]\,
          e^{-\,j\,2\pi kn / N},
\qquad k=0,\dots,N-1
```

into a three-step pipeline on the GPU:
a) Chirp pre-multiplication

```math
\begin{aligned}
a[n] &= x[n]\;e^{+\,j\pi n^{2}/N},\\
b[n] &= \;e^{-\,j\pi n^{2}/N}.
\end{aligned}
```

b) FFT-based convolution
Perform an M = 2·next_pow2(N)-1-point circular convolution:

```math
\begin{aligned}
y[k] \;=\; (a * b)[k]
\end{aligned}
```

c) Chirp post-multiplication and scaling

```math
\begin{aligned}
X[k] \;=\; \frac{1}{M}\,y[k]\,
           e^{-\,j\pi k^{2}/N}
\end{aligned}
```

Because the convolution length M is rounded up to the next power of two, all FFTs use cuFFT’s fastest radix-2 shader.

- CUDA kernels:
	- build_chirp: Generates the chirp-modulated sequences a[n] and b[n] (step 1). Each thread handles one sample n.
	- pad_B: Zero-pads b[n] to length M and mirrors the first N-1 taps to create a symmetric sequence needed by the convolution (step 2 prep).
	- pointwise_mul: Multiplies the element-wise spectra Â(k) ⋅ 𝐁̂(k) after the forward FFTs—implements the convolution in the frequency domain.
	- final_chirp: Applies the inverse chirp, divides by M to normalise the inverse FFT, and writes the final spectrum X[k] (step 3).

2) Implementation

Compiling the code:

<pre>nvcc .\bluestein_fft.cu -o bluestein</pre>

Running the code after compiling:

<pre>bluestein</pre>

<pre>Bluestein length N : 16384
Convolution M     : 32768

Host → Device copy      : 0.065152 ms
GPU kernels + FFTs      : 2.74266 ms
Device → Host copy      : 0.104608 ms
Total GPU time          : 2.91242 ms</pre>