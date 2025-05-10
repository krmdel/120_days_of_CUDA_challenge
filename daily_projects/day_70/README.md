Day 70: Implementation of comparison of optimized 2D DFT against cuFFT in CUDA

1) Summary of the daily tutorial

The code implements comparison of tiled CUDA kernel for the 2-D discrete Fourier transform (DFT) against NVIDIA’s highly-tuned cuFFT library. It allocates a real 2-D test image on the host and uploads it to the GPU and computes the forward 2-D DFT twice.

```math
F(u,v)=\!\sum_{y=0}^{H-1}\sum_{x=0}^{W-1} f(x,y)\,
        e^{-j\,2\pi\!\bigl(\tfrac{u x}{W}+\tfrac{v y}{H}\bigr)}
```

using incremental trigonometric recursion and shared-memory tiling;

- CUDA kernels:
    - dft2d_kernel_tiled: It performs the 2-D DFT one (u,v) point per thread. 16 × 16 tile loaded to shared memory. Incremental rotation avoids evaluating costly sinf/cosf inside the inner loops. Accumulates real/imag parts and writes a cuFloatComplex.
    - rotate: Constant-time complex multiply that updates the running cosine/sine pair when stepping to the next pixel in x or y. Used heavily inside the DFT kernel.                                                                                                     |

2) Implementation

Compiling the code:

<pre>nvcc .\dft_cufft.cu -o dft_cufft</pre>

Running the code after compiling:

<pre>dft_cufftd</pre>

<pre>Image 128×128 (16384 px)

Tiled kernel:
GPU H2D     : 0.056 ms
GPU Kernel  : 1.202 ms
GPU D2H     : 0.051 ms
GPU Total   : 1.309 ms

cuFFT       :
GPU H2D     : 0.053 ms
GPU Kernel  : 0.032 ms
GPU D2H     : 0.050 ms
GPU Total   : 0.135 ms</pre>