Day 88: Implementation of two dimensional (2D) discrete cosine transform (DCT) in CUDA

1) Summary of the daily tutorial

The code implements 8 × 8 forward and inverse 2-D DCT on a whole image in CUDA:
- Loads (or generates) an 8-bit grayscale image whose width × height are multiples of 8.
- Runs a CPU reference implementation to obtain ground-truth coefficients, a reconstructed image, timing, and PSNR.
- Copies the image to the GPU, launches two naïve CUDA kernels—one for the forward DCT and one for the inverse DCT—and copies the reconstruction back.

```math
C(u,v)=\alpha(u)\,\alpha(v)\sum_{x=0}^{7}\sum_{y=0}^{7}
        f(x,y)\,\cos\!\left[\tfrac{(2x+1)u\pi}{16}\right]
                 \cos\!\left[\tfrac{(2y+1)v\pi}{16}\right]

f(x,y)=\sum_{u=0}^{7}\sum_{v=0}^{7}\alpha(u)\,\alpha(v)\,C(u,v)\,
        \cos\!\left[\tfrac{(2x+1)u\pi}{16}\right]
        \cos\!\left[\tfrac{(2y+1)v\pi}{16}\right]
```

- CUDA kernels:
    - dct8x8_forward_naive: one thread-block per 8 × 8 tile, 64 threads (u,v) inside the block, computes a single coefficient C(u,v) with purely global-memory accesses and loops over the 64 spatial positions (x,y) accumulating the inner product with pre-computed cosine factors.
    dct8x8_inverse_naive: symmetric to the forward kernel. computes one spatial sample f(x,y) from the 64 frequency coefficients in the same block.

2) Implementation

Compiling the code:

<pre>nvcc .\2d_dct.cu -o 2d_dct</pre>

Running the code after compiling:

<pre>2d_dct</pre>

<pre>Generated random 1024×1024 image
CPU   forward+inverse    : 303.767 ms
GPU   H2D copy           : 1.79638 ms
GPU   kernel (Fwd+Inv)   : 0.925856 ms
GPU   D2H copy           : 0.896608 ms
GPU   total time         : 5.81875 ms
PSNR  CPU reconstruction : 131.477 dB
PSNR  GPU reconstruction : 131.427 dB</pre>