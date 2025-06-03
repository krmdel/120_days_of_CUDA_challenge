Day 92: Implementation of Sobel gradient magnitude and direction in CUDA

1) Summary of the daily tutorial

The code implements Sobel edge detection in CUDA. It reads (or synthetically creates) a grayscale image:
- CPU reference: computes per-pixel gradient magnitude and a 2-bit orientation bin (0°, 45°, 90°, 135°) for validation.
- GPU path: launches a single tiled CUDA kernel that reproduces the same outputs while timing host-to-device (H2D), kernel, and device-to-host (D2H) phases.

The Sobel convolution masks are

```math
G_x = \bigl(I_{x-1,y-1} + 2\,I_{x,y-1} + I_{x+1,y-1}\bigr)
      \;-\;
      \bigl(I_{x-1,y+1} + 2\,I_{x,y+1} + I_{x+1,y+1}\bigr)
```

```math
G_y = \bigl(I_{x-1,y-1} + 2\,I_{x-1,y} + I_{x-1,y+1}\bigr)
      \;-\;
      \bigl(I_{x+1,y-1} + 2\,I_{x+1,y} + I_{x+1,y+1}\bigr)
```

Gradient magnitude and angle are then

```math
M(x,y)=\sqrt{G_x^2+G_y^2},\qquad
\theta(x,y)=\mathrm{atan2}\!\left(G_y,G_x\right)\times\frac{180}{\pi}\bmod{180^{\circ}}.
```

Pixels are quantised into four orientation bins:

| Range (deg)      | Bin |
| ---------------- | --- |
| 0–22.5 or >157.5 | 0   |
| 22.5–67.5        | 1   |
| 67.5–112.5       | 2   |
| 112.5–157.5      | 3   |

- CUDA kernels:
    - sobelKernel: 16 × 16 tiled Sobel operator with 1-pixel halo held in shared memory, loads each tile (and its surrounding halo) from global memory exactly once, computes Gx, Gy, magnitude M, and orientation bin in registers, writes M (float32) and orientation bin (uint8) back to global memory.

2) Implementation

Compiling the code:

<pre>nvcc .\sobel.cu -o sobel</pre>

Running the code after compiling:

<pre>sobel</pre>

<pre>Image size: 4096 × 4096 
CPU total              : 791.718 ms
GPU H2D                : 3.78912 ms
GPU kernel             : 0.320064 ms
GPU D2H                : 13.7304 ms
GPU total              : 51.6726 ms
RMSE (mag CPU vs GPU)  : 0</pre>