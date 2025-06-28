Day 117: Implementation of feature matching with FAST-9 corner detection/BRIEF-256 descriptor generation and brute-force Hamming matcher in CUDA

1) Summary of the daily tutorial
The code implements:
- FAST-9 key-point detection
- BRIEF-256 descriptor computation with patch-rotation compensation
- Brute-force Hamming matching with the Lowe ratio test
- GPU RANSAC to count geometric inliers.

The code can ingest two grayscale images (PGM or OpenCV) or synthesise random frames, then reports per-stage timings plus the CPU baseline.

```math
\text{Corner test:}\;
\max_{\text{16‐pixel ring}}\Bigl|\,I(p_i)-I(p)\Bigr| > t,\quad
\text{9 consecutive pixels satisfy the test.}
```

```math
\text{BRIEF bit } b:\;
d_k[b] = \begin{cases}
  1 & \text{if } I\!\bigl(p+\delta_b^{(1)}\bigr) < I\!\bigl(p+\delta_b^{(2)}\bigr)\\[4pt]
  0 & \text{otherwise}
\end{cases}\!,\qquad b = 0\dots255
```

```math
\text{Hamming distance:}\;
D(A,B)=\sum_{w=0}^{7}\operatorname{popcount}\!\bigl(A_w\oplus B_w\bigr)
```

```math
\text{Lowe ratio test:}\;
D_\text{best} < r \times D_\text{second‐best},\quad r=0.75
```

```math
\text{Similarity transform (RANSAC hypothesis):}\;
q \approx s\,p + t,\quad
s=\sqrt{\dfrac{\|q_1-q_0\|^2}{\|p_1-p_0\|^2}}
```

- CUDA kernels:
fastKer: Scans every pixel, evaluates the FAST-9 16-pixel ring test, and writes accepted key-points into a global array using atomicAdd to keep a per-image counter.
briefKer: For each accepted key-point, estimates orientation via intensity centroid, rotates the 256 predefined point pairs, compares their intensities and packs the results into eight 32-bit words (the BRIEF-256 descriptor).
matchKer: Performs an exhaustive descriptor‐to‐descriptor comparison to find the best and second-best Hamming distances and stores the index of the match if the Lowe ratio passes.
ransacKer: Runs 256 iterations of two-point RANSAC inside one block, uses shared-memory reduction to keep the best inlier count, and atomically updates a global best score.

2) Implementation

Compiling the code:

<pre>nvcc .\feature_matching.cu -o feature_matching</pre>

Running the code after compiling:

<pre>feature_matching</pre>

<pre>[Info] generated two identical 1920×1080 frames
Resolution              : 1920 × 1080
kp1 / kp2               : 8192 / 8192
GPU best inliers        : 55

GPU H2D copy            : 1.19629 ms
GPU FAST kernels        : 0.180224 ms
GPU BRIEF kernels       : 0.529248 ms
GPU match kernel        : 2.77606 ms
GPU RANSAC kernel       : 4.00384 ms
GPU D2H copy            : 0.027616 ms
GPU total               : 8.71328 ms
CPU total               : 2031.12 ms</pre>