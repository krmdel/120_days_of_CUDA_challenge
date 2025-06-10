Day 99: Implementation of multi stream overlap for Harris corner detection in CUDA

1) Summary of the daily tutorial

The code performs the classic Harris corner detector on the GPU while overlapping data transfers and computation with two CUDA streams. The 4096 × 4096 gradient images are processed in vertical stripes (CHUNK = 256 rows). For every stripe the pipeline:

- copies Iₓ , I_y gradients to the GPU,
- computes the Harris response R, and
- copies the result back

is double-buffered across two streams (ping-pong): while stream 0 is busy on stripe n, stream 1 is already transferring stripe n + 1.

For every pixel the algorithm first builds the structure tensor:

```math
M = \begin{bmatrix}
I_x^2 & I_x I_y \\
I_x I_y & I_y^2
\end{bmatrix}_{\!\text{smoothed}}
```

and then evaluates the Harris response:

```math
R = \det(M) - k \, \left(\mathrm{trace}(M)\right)^2
```

where the smoothing is a separable 7 × 7 box filter (radius = 3) and k = 0.04.

- CUDA kernels:
  - prodKernel: computes per-pixel products Iₓ², I_y², Iₓ I_y.
  - blurH<3>: horizontal 7-tap box blur (templated radius = 3).
  - blurV<3>: vertical 7-tap box blur.
  - harrisKernel: evaluates the Harris formula above on the smoothed tensors.

2) Implementation

Compiling the code:

<pre>nvcc .\multi_stream.cu -o multi_stream</pre>

Running the code after compiling:

<pre>multi_stream</pre>

<pre>Generated random gradients 4096×4096
CPU total               : 678.62 ms
GPU total (overlapped)  : 47.0456 ms
RMSE (CPU vs GPU)       : 9.51975e+11</pre>