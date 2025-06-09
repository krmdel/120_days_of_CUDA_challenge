Day 98: Implementation of Non-Maximum Suppression and corner compaction in CUDA

1) Summary of the daily tutorial

The code implements the detection of 2-D interest points (e.g. Harris or FAST corners) in a response map R of size W × H, removes non-maxima with a 3 × 3 window, and packs the surviving pixel indices into a dense list.

Thresholding:

  ```math
  \text{keep}(i) = [\,R_i \; \ge \; \tau\,], \qquad   
  \tau = \text{frac}\; \times \max\nolimits_j R_j
  ```

3 × 3 non-maximum suppression:

  ```math
  \text{flag}[i] \;=\;
  \begin{cases}
    1, & R_i \ge \tau \;\;\land\;\; R_i \ge R_j \;\;\forall j \in \mathcal N(i) \\[4pt]
    0, & \text{otherwise}
  \end{cases}
  ```

  where 𝒩(i) are the eight neighbouring pixels.

Prefix-free compaction:

  ```math
  \text{list}[\,\text{atomicAdd(counter,1)}\,] \;=\; i
  ```

Every thread that finds flag[i]=1 reserves the next slot in list[] with an atomic increment.

- CUDA kernels:
    - nmsKernel: performs 3 × 3 NMS and thresholding per pixel block-parallel. Each CTA covers a 16 × 16 tile; inner pixels only (border skipped) are compared with their 8 neighbours using registers.
    - compactKernel: converts the boolean map flag[] to a dense index list list[] using an atomic counter for prefix-free writing.                                                          |
    - flagCopyKernel: lightweight device-to-device memcpy that copies the flag map into a second buffer so it can be returned to the host for an exact CPU ↔ GPU disagreement count.                         |
    
2) Implementation

Compiling the code:

<pre>nvcc .\nms_corner_compaction.cu -o nms_corner_compaction</pre>

Running the code after compiling:

<pre>nms_corner_compaction</pre>

<pre>Random response map 4096×4096   (threshold = 1 % of max)
CPU total (NMS)          : 155.2 ms   (corners = 1860969)
GPU H2D copy             : 14.2664 ms
GPU NMS kernel           : 0.241536 ms
GPU compaction kernel    : 0.643072 ms   (corners = 1860969)
GPU D2H copy             : 16.968 ms
GPU total                : 32.1414 ms
Pixel disagreement       : 0 %</pre>