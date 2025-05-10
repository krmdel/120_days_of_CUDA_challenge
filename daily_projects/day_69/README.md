Day 69: Implementation of two dimensional (2D) discrete Fourier transform (DFT) using shared memory & tiled kernel in CUDA

1) Summary of the daily tutorial

The code implements the computation of the 2-D DFT of a real-valued image. A single CUDA kernel, dft2d_kernel_tiled, is launched so that every thread evaluates one output frequency bin (u,v).

Each block repeatedly:
- loads an input TILE × TILE sub-image into shared memory,
- accumulates that tile’s contribution to the chosen (u,v) coefficient using an incremental rotation trick that reuses cos/sin results instead of recalculating expensive transcendental functions,
- moves to the next tile until the entire image has been processed.

```math
X(u,v)\;=\;\sum_{y=0}^{H-1}\sum_{x=0}^{W-1}
            f(x,y)\,e^{-j\,2\pi\!\left(\frac{u\,x}{W}\;+\;\frac{v\,y}{H}\right)}
```

where f(x,y) is the spatial-domain pixel value and X(u,v) is its Fourier coefficient.

- CUDA kernels:
    - dft2d_kernel_tiled: Evaluates one complex DFT coefficient per thread. Cooperative TILE × TILE loads into shared memory to reduce global-memory traffic. Uses pre-computed Δ-angle sines/cosines plus the inline rotate() helper to advance the phase by ±1 pixel with no extra trig calls. Two nested loops march over all tiles in y and x to cover the full image, updating running real/imag accumulators.
    - rotate(): Constant-time phase update (cos,sin) leftarrow (cos, sin) + Delta. Eliminates slow sincosf inside inner loops by using a complex-plane rotationr recurrence.

2) Implementation

Compiling the code:

<pre>nvcc .\dft_2d_shared.cu -o dft_2d_shared</pre>

Running the code after compiling:

<pre>dft_2d_shared</pre>

<pre>Image 128×128 (16384 px)

GPU H2D     : 0.04608 ms
GPU Kernel  : 1.15712 ms
GPU D2H     : 0.055616 ms
GPU Total   : 1.25882 ms</pre>