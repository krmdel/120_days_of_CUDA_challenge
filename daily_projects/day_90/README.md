Day 90: Implementation of zig-zag scanning and quantisation for JPEG style compression in CUDA

1) Summary of the daily tutorial

The code implements a JPEG encoder and compression algorithms:
- Forward 8 × 8 DCT to decorrelate image blocks.
- Zig-zag scan to reorder the 64 DCT coefficients from low- to high-frequency so that runs of zeros cluster at the end of the vector.
- Uniform quantisation with the standard JPEG luminance table, scaled by a user-controlled quality factor q_f.
- The symmetric inverse path (de-quantise → inverse zig-zag → inverse DCT) reconstructs the image so that you can measure fidelity (PSNR).

- Quantisation:

```math
q_{i} \;=\; \operatorname{round}\!\left( \frac{c_{i}}{q_f \, Q_{i}} \right)
```

- De-quantisation:

```math
\hat{c}_{i} \;=\; q_{i} \;(q_f \, Q_{i})
```

where c_i = raw DCT coefficient, Q_i = entry from the JPEG luminance Q-matrix, q_f = quality-factor scale (≥ 1).

- CUDA kernels:
    - dct8x8_forward_shared: Two-pass tiled DCT (row then column) done entirely in shared memory. Each block processes one 8 × 8 tile; pre-computed cosines live in constant memory for fast lookup.
    - zigzag_kernel: Reorders 64 coefficients to JPEG zig-zag order. A constant-memory LUT maps the thread’s row-major index to its zig-zag position.
    - quantise_kernel (1D grid, 256-thread blocks): Divides each coefficient by q_f Q_i and rounds to int16. Operates on the flattened coefficient vectors produced by zigzag_kernel.
    - dequant_invzig_kernel: Exact inverse of the previous two stages: multiplies by q_f Q_i and writes back into the 8 × 8 spatial layout using the same LUT.
    - dct8x8_inverse_shared: Shared-memory 8 × 8 inverse DCT (column pass then row pass) to reconstruct the spatial block.

2) Implementation

Compiling the code:

<pre>nvcc .\jpeg_comp.cu -o jpeg_comp</pre>

Running the code after compiling:

<pre>jpeg_comp</pre>

<pre>Generated random 1024×1024 image

qf=1
CPU forward+inverse        : 144.472 ms
GPU H2D copy               : 1.0184 ms
GPU Forward DCT            : 0.052544 ms
GPU Zig-zag                : 0.032768 ms
GPU Quantisation           : 0.032768 ms
GPU Dequant+Inv-zig-zag    : 0.050176 ms
GPU Inverse DCT            : 0.073728 ms
GPU D2H copy               : 0.918336 ms
GPU total                  : 4.25434 ms
PSNR CPU reconstruction    : 131.477 dB
PSNR GPU reconstruction    : 13.4772 dB</pre>