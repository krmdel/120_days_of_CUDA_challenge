Day 102: Implementation of pinned 2D integral image scan in CUDA

1) Summary of the daily tutorial

The code performs a high-throughput integral-image calculator that:
- Copies an 8-bit grayscale image to the GPU using pinned host memory for maximal PCIe bandwidth.
- Performs two large parallel prefix scans (rows → columns) entirely on-device, using tiled kernels and an in-place transpose to reuse the same scan code for both axes.
- Streams results back to pinned memory and reports per-stage timings plus the RMSE against a CPU reference implementation.

Integral-image formula (for pixel (x,y))

```math
I(x,y)=\sum_{i=0}^{y}\;\sum_{j=0}^{x}S(j,i)
```

where S is the source intensity and I is the integral value.

- CUDA kernels:
  - scan_u8: Row-wise prefix scan on uint8 data, producing uint32 partial sums and a per-tile tail value.	Shared memory tile (1024 × uint32) + Hillis-Steele scan → writes both the prefix and the tile sums.
  - scan_u32: Identical logic for the second pass (already uint32).	Reuses the same template to avoid code duplication.
  - scanSums: Serial prefix across tile tails for each row.	Runs in 1 block per row → negligible overhead.
  - addOff: Adds the row-level offset back to every element in its tile.	Finalises a full row scan.
  - transpose32u: 32 × 32 tiled transpose with a 33-column shared-mem buffer to defeat bank conflicts.	Used twice to swap X↔Y so we can reuse the same row-scan logic for columns.

2) Implementation

Compiling the code:

<pre>nvcc .\2d_integral_pinned.cu -o 2d_integral_pinned</pre>

Running the code after compiling:

<pre>2d_integral_pinned</pre>

<pre>Random 4096×4096
CPU total               : 46.7651 ms
GPU H2D copy            : 1.3751 ms
GPU kernels             : 1.56294 ms
GPU D2H copy            : 5.09808 ms
GPU total               : 8.06387 ms
RMSE                    : 0</pre>