Day 89: Implementation of forward and inverse DCT using shared memory and tiling in CUDA

1) Summary of the daily tutorial

The code implements an 8 × 8 forward and inverse discrete-cosine transform (DCT).
- Reads (or randomly generates) a grayscale image whose width and height are multiples of 8.
- Converts the image to floating point and stores it on the GPU.
- Executes two shared-memory CUDA kernels that perform a separable 2-D DCT and its inverse, one 8 × 8 block per thread-block.
- Copies the reconstructed image back, computes PSNR.


The 2-D DCT-II applied to each 8 × 8 block f(x,y) is

```math
C(u,v)=\alpha(u)\,\alpha(v)\!
        \sum_{x=0}^{7}\sum_{y=0}^{7}
        f(x,y)\,
        \cos\!\Bigl[\tfrac{(2x+1)\,u\pi}{16}\Bigr]\,
        \cos\!\Bigl[\tfrac{(2y+1)\,v\pi}{16}\Bigr]
```

and the inverse (DCT-III) is

```math
f(x,y)=\sum_{u=0}^{7}\sum_{v=0}^{7}
       \alpha(u)\,\alpha(v)\,C(u,v)\,
       \cos\!\Bigl[\tfrac{(2x+1)\,u\pi}{16}\Bigr]\,
       \cos\!\Bigl[\tfrac{(2y+1)\,v\pi}{16}\Bigr]
```

- CUDA kernels:  
    - dct8x8_forward_shared: forward 8 × 8 DCT, loads one 8 × 8 tile into shared memory. Pass 1: each thread computes one frequency coefficient along its row and synchronises and transposes the intermediate results in shared memory. Pass 2: repeats the operation down the columns and writes the final DCT coefficients to global memory.
    - dct8x8_inverse_shared: inverse 8 × 8 DCT, loads a coefficient tile into shared memory. Pass 1: performs the inverse transform down the columns, storing temporaries in shared memory. Pass 2: processes across the rows to reconstruct pixels and stores them in global memory.

2) Implementation

Compiling the code:

<pre>nvcc .\2d_dct_tiled.cu -o 2d_dct_tiled</pre>

Running the code after compiling:

<pre>2d_dct_tiled</pre>

<pre>Generated random 1024×1024 image
CPU forward+inverse      : 88.5914 ms
GPU H2D copy             : 1.0336 ms
GPU kernels (Fwd+Inv)    : 0.115232 ms
GPU D2H copy             : 0.909728 ms
GPU total                : 4.15146 ms
PSNR CPU reconstruction  : 131.477 dB
PSNR GPU reconstruction  : 132.773 dB</pre>