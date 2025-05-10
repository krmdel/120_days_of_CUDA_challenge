Day 68: Implementation of two dimensional (2D) discrete Fourier transform (DFT) in CUDA

1) Summary of the daily tutorial

    The code implements baseline 2-D discrete Fourier transform for a real-valued image. It allocates a random single-precision image on the host, computes the reference DFT on the CPU for timing / correctness comparison, copies the image to the GPU, launches a CUDA kernel in which each thread calculates one output frequency bin $u,v$, copies the complex spectrum back to the host.

        ```math
        F(u,v)=\sum_{y=0}^{H-1}\sum_{x=0}^{W-1} f(x,y)\,
        e^{-\,j\,2\pi\left(\frac{ux}{W}+\frac{vy}{H}\right)}
        ```

- CUDA kernels:
    - dft2d_kernel: Each thread maps to one output coefficient F(u,v). Iterates over every input pixel, accumulating its contribution with sincosf to avoid repeated trig calls. Stores the result as a cuFloatComplex in global memory.

2) Implementation

Compiling the code:

<pre>nvcc .\dft_2d.cu -o dft_2d</pre>

Running the code after compiling:

<pre>dft_2d</pre>

<pre>Image size: 128 x 128 (16384 pixels)

CPU DFT     : 4787.070 ms
GPU H2D     : 1.132 ms
GPU Kernel  : 8.352 ms
GPU D2H     : 0.054 ms
GPU Total   : 9.537 ms</pre>