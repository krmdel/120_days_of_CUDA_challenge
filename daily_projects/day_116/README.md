Day 116: Implementation of FAST-9, BRIEF-256 and brute-force Hamming matcher in CUDA

1) Summary of the daily tutorial

The code implements ORB-like pipeline:
- FAST-9 corner detector finds up to 4 096 keypoints.
- Orientation-steered BRIEF-256 generates a 256-bit binary descriptor for every keypoint.
- A tiled brute-force matcher returns the smallest Hamming distance for each descriptor.

```math
H(\mathbf{d}_i,\mathbf{d}_j)\;=\;\sum_{k=0}^{255}\bigl(d_{i,k}\;\oplus\;d_{j,k}\bigr)
```

- CUDA kernels:
	- fast9Ker: Detects FAST-9 corners.	Each thread examines one pixel• 16-pixel Bresenham circle stored in constant memory. Requires 9 consecutive bright/dark pixels: atomicAdd stores passing coordinates into global list.
	- briefKer: Build BRIEF-256 descriptor. One thread per keypoint. Computes intensity centroid to get orientation θ• Rotates 256 point-pairs:

```math
\begin{pmatrix}x'\\y'\end{pmatrix}
=
\begin{pmatrix}\cos\theta&-\sin\theta\\\sin\theta&\cos\theta\end{pmatrix}
\!\begin{pmatrix}x\\y\end{pmatrix}
```

2) Implementation

Compiling the code:

<pre>nvcc .\brief_256.cu -o brief_256</pre>

Running the code after compiling:

<pre>brief_256</pre>

<pre>[Info] random frame 1920×1080

Image                   : 1920 × 1080
FAST threshold          : 25
Keypoints (GPU)         : 4096
Keypoints (CPU)         : 4096

GPU H2D copy            : 0.66592 ms
GPU FAST kernel         : 0.095232 ms
GPU BRIEF kernel        : 0.424256 ms
GPU Match kernel        : 2.8416 ms
GPU D2H copy            : 0.020928 ms
GPU total               : 4.09594 ms
CPU total               : 510.822 ms</pre>