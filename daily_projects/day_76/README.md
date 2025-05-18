Day 76: Implementation of Rader's algorithm for a prime length DFT in CUDA

1) Summary of the daily tutorial

The code implements Rader’s algorithm to convert a length-N DFT, where N is prime, into a cyclic convolution of length N-1. By mapping the input samples through a primitive root of the multiplicative group, the DFT is reduced to

```math
\begin{aligned}
X[k] \;&=\; x[0]\;+\;\sum_{m=0}^{N-2} x\!\bigl[g^{\,m}\bigr]\,
            e^{-\,j\,\tfrac{2\pi k\,g^{\,m}}{N}}\\[6pt]
        &\;=\; x[0]\;+\;\bigl(a \star b_k\bigr)[m]
\end{aligned}
```

where a_m - the input reordered by powers of the primitive root. b_m - a twiddle sequence that depends only on $begin:math:text$N$end:math:text$  

The circular convolution a * b_k is performed with cuFFT:
- Forward FFT of both sequences
- Point-wise complex multiplication
- Inverse FFT and normalisation

- CUDA kernels:
	- mul_complex: Point-wise complex multiplication. Each thread multiplies one pair of complex numbers from the two FFT outputs and writes the product. This forms the element-wise product in the convolution theorem.
	- primitive_root: Finds the smallest generator g for arbitrary prime N.
	- next_pow2: Pads the convolution length N-1 up to the nearest power of two for an efficient radix-2 FFT size.

2) Implementation

Compiling the code:

<pre>nvcc .\rader.cu -o rader</pre>

Running the code after compiling:

<pre>rader</pre>

<pre>Prime length N  : 257
Convolution len : 256
FFT size (pow2) : 256

H2D copy        : 0.03072 ms
FFT+mul+IFFT    : 2.96141 ms
D2H copy+build  : 0.255808 ms
Total GPU time  : 3.24794 ms</pre>