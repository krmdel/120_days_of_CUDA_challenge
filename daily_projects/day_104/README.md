Day 104: Implementation of Viola–Jones sliding window edge detector in CUDA

1) Summary of the daily tutorial

The code implements Viola–Jones style sliding window edge detector that scans a large grayscale image (default 4096 × 4096 px or a supplied PGM) and flags every 24 × 24 window whose lower half is noticeably brighter than its upper half. It accelerates three main stages on the GPU:
- Integral-image construction: two passes of tiled, inclusive scans plus two 32 × 32 transposes turn raw pixels into a summed-area table in ~O(1) access per rectangle.
- Sliding-window evaluation: the summed-area table lets each block test many candidate windows with just a handful of memory reads.
- Result marshaling: detected windows are copied back to the host for verification against a CPU reference.

The Haar-like “edge” feature for a window’s vertical split is

```math
\text{score} = \bigl[\text{II}(x_2,\,y_2) - \text{II}(x_2,\,y_m) - \text{II}(x_1-1,\,y_2) + \text{II}(x_1-1,\,y_m)\bigr]
              \;-\;
              \bigl[\text{II}(x_2,\,y_m) - \text{II}(x_2,\,y_1-1) - \text{II}(x_1-1,\,y_m) + \text{II}(x_1-1,\,y_1-1)\bigr]
```

where II is the integral image. A detection is positive when score > threshold (50 in the sample code).

- CUDA kernels:
  - scanRow_u8: Tile-wise inclusive prefix-scan of one image row (input uint8 → output uint32).• Stores the row-sum of each 1024-pixel tile in sums for a second-stage scan.
  - scanSums: Serial prefix-scan of each row’s tile totals to propagate horizontal offsets.
  - addOff: Adds those offsets back into every element, completing the horizontal integral pass.
  - transpose32u: Shared-memory 32 × 32 tiled transpose; converts row-integral image into column-major layout so the same scan kernels can be reused vertically.
  - scanRow_u32: Same as scanRow_u8, but input/output are uint32_t (second pass of the integral image).
(second pair of) scanSums / addOff	• Scan and add vertical tile offsets to finish the integral-image construction in the original orientation.
  - vjDetector: Each 32 × 32 thread block tests up to 1024 sliding windows (stride 4).• Reads four values per window from the integral image, computes score, writes a single byte result.

2) Implementation

Compiling the code:

<pre>nvcc .\violajones.cu -o violajones</pre>

Running the code after compiling:

<pre>violajones</pre>

<pre>Random 4096×4096
CPU total               : 49.8735 ms
GPU H2D copy            : 1.3751 ms
GPU kernels             : 1.38317 ms
GPU D2H copy            : 0.092352 ms
GPU total               : 2.8753 ms
Mismatching windows     : 0</pre>