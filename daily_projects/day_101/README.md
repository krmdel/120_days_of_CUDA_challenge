Day 101: Implementation of 2D integral image with thrust segmented scans and tiled transpose in CUDA

1) Summary of the daily tutorial

The code computes an integral image (cumulative 2-D prefix-sum) on the GPU. For every pixel (x,y) in the input image I, the output integral image stores the sum of all pixels in the axis-aligned rectangle from the origin to that pixel:

```math
\mathrm{II}(x,y)=\sum_{i=0}^{y}\sum_{j=0}^{x} I(j,i)
```

The algorithm proceeds in four GPU stages:
- Row-wise inclusive scans with thrust::inclusive_scan_by_key produce per-row prefix sums.
- Transpose those row scans so that columns become contiguous in memory.
- Column-wise scans (again an inclusive scan by key) operate on the transposed data.
- Transpose back to restore the original layout; the result is the full integral image.

- CUDA kernels:
  - transpose32u: 32 × 32 shared-memory tile transpose for uint32_t images, uses a 32×33 tile to avoid shared-memory bank conflicts, called twice: first W x H --> H x W, then the inverse.

2) Implementation

Compiling the code:

<pre>nvcc .\2d_integral.cu -o 2d_integral</pre>

Running the code after compiling:

<pre>2d_integral</pre>

<pre>Random 4096×4096 image
CPU total               : 47.8561 ms
GPU H2D copy            : 3.75936 ms
GPU kernels             : 1.53462 ms
GPU D2H copy            : 13.6048 ms
GPU total               : 52.7001 ms
RMSE                    : 0</pre>