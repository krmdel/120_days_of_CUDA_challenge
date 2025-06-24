Day 113: Implementation of HOG-SVM in CUDA

1) Summary of the daily tutorial

The code implements HOG (Histogram of Oriented Gradients) and SVM (Support Vector Machine):
   - ingests or synthetically creates a grayscale image,
   - extracts per-pixel gradients,
   - aggregates them into 9-bin cell histograms,
   - ℓ2-Hys normalises overlapping 2 × 2 cell blocks,
   - assembles each 64 × 128 detection window into a 3 778--dim descriptor,
   - evaluates a linear SVM, w·x + b, for every sliding window,
   - thresholds, copies detections back, then performs CPU non-max suppression (NMS) and optional drawing.

```math
\|{\bf G}\| = \sqrt{G_x^2 + G_y^2}, \qquad
\theta      = \operatorname{atan2}(G_y,\,G_x)
```

```math
\text{cellHist}[b] \mathrel{+}= \text{mag}
\quad\text{where } b=\Bigl\lfloor\tfrac{\theta\cdot180^\circ}{20^\circ}\Bigr\rfloor
```

```math
{\bf v}= \min\!\Bigl(\frac{{\bf h}}{\sqrt{\|{\bf h}\|_2^2+\varepsilon}},\,0.2\Bigr),
\qquad
{\bf v}= \frac{{\bf v}}{\sqrt{\|{\bf v}\|_2^2+\varepsilon}}
```

```math
s = {\bf w}^\top {\bf v} \;+\; b
```

- CUDA kernels:
   - sobelKer: 16 × 16 tiled Sobel operator in shared memory; outputs gradient magnitude & orientation per pixel.
   - cellKer: One warp per 8 × 8 cell; builds 9-bin orientation histogram and reduces across the warp.
   - blockKer: Reads 2 × 2 neighbouring cells (36 values), performs ℓ2-Hys normalisation & clipping for block descriptors.
   - gatherKer: Collects a 7 × 15 grid of blocks into a single 3 780-float detection-window descriptor.
   - dotKer: Computes the SVM dot-product, w·x, for every descriptor; bias, b, is added from constant memory.
   - threshKer: Writes (x, y, score) for descriptors whose score > thr, using an atomic counter.

2) Implementation

Compiling the code:

<pre>nvcc .\hog_svm.cu -o hog_svm</pre>

Running the code after compiling:

<pre>hog_svm</pre>

<pre>[Info] no image provided – generating random 1280×720 frame
Image 1280×1280
GPU H2D copy      : 4.55987 ms
GPU kernels       : 0.0189757 ms
GPU total         : 4.57885 ms
CPU NMS           : 74.3905 ms
Detections raw/NMS: 22185 / 954
Saved hog_out.ppm</pre>