Day 93: Implementation of Non-Maximum Suppression (NMS) in CUDA

1) Summary of the daily tutorial

The code implements non-maximum suppression in CUDA. Given a magnitude map, mag, and a direction map, ang (each pixel quantised to one of four principal orientations 0°, 45°, 90°, 135°), the code:

- Runs a CPU reference for ground-truth timing and accuracy.
- Launches a tiled CUDA kernel that loads a 16 × 16 patch (plus a 1-pixel halo) into shared memory and, in parallel, keeps only local maxima along the gradient direction.
- Compares CPU vs GPU run-time and computes the RMSE between their outputs.

For every interior pixel (x,y) with magnitude M(x,y) and two directional neighbours n_1 and n_2,

```math
\text{out}(y,x)\;=\;
\begin{cases}
M(y,x), & \text{if } M(y,x)\ge n_1\;\land\;M(y,x)\ge n_2,\\[4pt]
0,      & \text{otherwise}.
\end{cases}
```

- CUDA kernels:
- nmsKernel: 16 × 16 threads per block, shared-memory tile + halo, loads local patch from mag and three-bit direction from ang. For border pixels writes 0 immediately, compares centre magnitude with the two neighbours determined by the quantised angle, stores either the preserved magnitude or 0 in the output buffer.

2) Implementation

Compiling the code:

<pre>nvcc .\nms.cu -o nms</pre>

Running the code after compiling:

<pre>nms</pre>

<pre>Generated random test data (4096×4096)
CPU NMS total           : 281.515 ms
GPU H2D copy            : 18.094 ms
GPU NMS kernel          : 0.298784 ms
GPU D2H copy            : 13.6796 ms
GPU total               : 65.6443 ms
RMSE (CPU vs GPU)       : 0</pre>