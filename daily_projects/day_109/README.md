Day 109: Implementation of Sobel gradient in CUDA

1) Summary of the daily tutorial

The code implements Sobel edge-detection on the GPU. It
- loads a PGM image (or creates a random one),
- computes horizontal and vertical gradients,
- derives gradient magnitude and orientation for every pixel, and
- measures RMSE between CPU and GPU outputs to prove numerical correctness.

Per-pixel operations are

```math
G_x =
\begin{bmatrix}
-1 &  0 &  1\\[-2pt]
-2 &  0 &  2\\[-2pt]
-1 &  0 &  1
\end{bmatrix} * I,
\qquad
G_y =
\begin{bmatrix}
 1 &  2 &  1\\[-2pt]
 0 &  0 &  0\\[-2pt]
-1 & -2 & -1
\end{bmatrix} * I
```

```math
M(x,y)=\sqrt{G_x^2+G_y^2},
\qquad
\theta(x,y)=\text{atan2}\!\bigl(G_y,G_x\bigr)
```

- CUDA kernels:
  - sobelKernel: one 16 × 16 thread-block per image tile
    - Loads the tile plus a 1-pixel halo into shared memory.
    - Applies Sobel filters to get Gx, Gy.
    - Writes magnitude and angle (using `sqrtf`, `atan2f`) to global memory.

2) Implementation

Compiling the code:

<pre>nvcc .\sobel_gradient.cu -o sobel_gradient</pre>

Running the code after compiling:

<pre>sobel_gradient</pre>

<pre>Generated random 4096×4096 image
CPU total               : 723.785 ms
GPU H2D copy            : 3.77613 ms
GPU kernel              : 0.297248 ms
GPU D2H copy            : 28.7937 ms
GPU total               : 101.299 ms
RMSE  magnitude         : 0
RMSE  angle             : 8.99156e-08</pre>