Day 100: Implementation of FAST corner detector with learned threshold in CUDA

1) Summary of the daily tutorial

The code implements a GPU pipeline that (1) computes a gradient-based, learned threshold for every pixel, (2) runs the FAST-9 corner test in parallel, (3) compacts the detected key-points into a contiguous list, and (4) estimates an ORB-style orientation for each key-point.

Gradient magnitude:

  ```math
  g(x,y)=\sqrt{[I(x{+}1,y)-I(x{-}1,y)]^{2}+[I(x,y{+}1)-I(x,y{-}1)]^{2}}
  ```

Learned per-pixel threshold:

  ```math
  T(x,y)=w_{0}+w_{1}\,g(x,y)
  ```

  where w_{0} and w_{1} are small learned scalars stored in constant memory.

FAST-9 test: a pixel is a corner if nine contiguous pixels on the Bresenham circle of radius 3 are all either brighter than I_c + T or darker than I_c - T.

- CUDA kernels:
  - gradThresh: 16 × 16-thread tiles compute g(x,y) and the learned threshold T(x,y) for each interior pixel, storing the gradient (for later experiments) and the threshold map.
  - fastKernel: Each row is processed by blocks of 32 threads. A pixel’s 16 circle neighbours are compared against I_c ± T; contiguous 9-bit masks (pre-computed in constant memory) quickly decide the FAST-9 condition and set flag[x].
  - compactKernel: A classic stream-compaction step: every thread that sees flag[i] ≠ 0 pushes its index into d_list via atomicAdd, while a single global counter tracks the total key-points.
   - orbOrient: For each key-point, 32 × 8 threads scan a 31 × 31 window, accumulate image moments xI and yI in shared memory, and compute the ORB orientation.

2) Implementation

Compiling the code:

<pre>nvcc .\fast_ml_threshold.cu -o fast_ml_threshold</pre>

Running the code after compiling:

<pre>fast_ml_threshold</pre>

<pre>Batch 4 frames  (1920×1080)
Frame0 flag diff : 0 %
GPU H2D copy         : 2.18038 ms
GPU kernels          : 13.989 ms
GPU D2H copy         : 0.092256 ms
GPU total            : 16.3234 ms (245.047 fps)
Total keypoints      : 2641641</pre>