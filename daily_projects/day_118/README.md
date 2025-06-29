Day 118: Implementation of optical flow using sparse Lucas-Kanade with 3-level pyramid in CUDA

1) Summary of the daily tutorial

The code implements optical flow for a sparse set of points between two grayscale frames. It uses a 3-level image pyramid (coarse → fine) and, at each level, applies the classic 5 × 5 Lucas–Kanade (LK) method to iteratively refine the displacement of every point on the GPU, then compares performance with a CPU reference.

```math
\underbrace{\begin{bmatrix}
\sum I_x^2 & \sum I_x I_y \\[2pt]
\sum I_x I_y & \sum I_y^2
\end{bmatrix}}_{A}
\!\!
\underbrace{\begin{bmatrix}
d_x \\ d_y
\end{bmatrix}}_{\mathbf{d}}
=
-\;
\underbrace{\begin{bmatrix}
\sum I_x\,I_t \\[2pt] \sum I_y\,I_t
\end{bmatrix}}_{\mathbf{b}}
```

The per-patch normal equations Ad = −b are solved once; the resulting d is back-projected to the next-finer level.

- CUDA kernels:
  - lk_level: For each point at the current scale:
    - Bilinearly samples the 5 × 5 neighborhood in the reference frame (I0) to obtain spatial gradients Iₓ, Iᵧ and the temporal gradient Iₜ = I₁ − I₀.
    - Accumulates the normal-equation terms A₁₁, A₁₂, A₂₂, b₁, b₂.
    - Solves the 2 × 2 linear system with a small damping term *(+1 e-3)* for numerical stability.
    - Updates the point position, mapping the pixel-space displacement back to level 0 coordinates.

2) Implementation

Compiling the code:

<pre>nvcc .\lucas_kanade.cu -o lucas_kanade</pre>

Running the code after compiling:

<pre>lucas_kanade</pre>

<pre>[Info] random frames 1920×1080

Frame size              : 1920 × 1080
Tracked points          : 8192

GPU kernels             : 0.186368 ms
GPU D2H copy            : 0.032128 ms
GPU total               : 0.230656 ms
CPU total               : 33.8732 ms

First 8 flow vectors (GPU):
(2,2)->(1.52421,5.26208)  dx=-0.475785 dy=3.26208
(17,2)->(12.9169,1.59721)  dx=-4.08313 dy=-0.402794
(32,2)->(34.6534,-5.22499)  dx=2.65337 dy=-7.22499
(47,2)->(41.7412,-4.11716)  dx=-5.25876 dy=-6.11716
(62,2)->(58.5611,-5.55777)  dx=-3.4389 dy=-7.55777
(77,2)->(76.8675,-10.3792)  dx=-0.132484 dy=-12.3792
(92,2)->(92.8079,-0.320153)  dx=0.807899 dy=-2.32015
(107,2)->(107.331,1.16891)  dx=0.331406 dy=-0.831092</pre>