Day 111: Implementation of HOG block descriptor in CUDA

1) Summary of the daily tutorial

The code computes Histogram-of-Oriented-Gradients (HOG) block descriptors on both CPU and GPU and compares their numeric accuracy (RMSE). Each block is a 2 × 2 group of adjacent cells (4 cells × 9 bins = 36 values). The descriptor is produced with the standard L2-Hys scheme: first L2-normalize, clip high values, then renormalize.

- Gather four neighbouring 9-bin cell histograms
- Initial L2-normalization:

   ```math
   \hat{\mathbf{v}} = \frac{\mathbf{v}}{\sqrt{\varepsilon + \lVert\mathbf{v}\rVert_2^{2}}}
   ```

- Clipping (Hys):

   ```math
   \hat{\mathbf{v}}_i = \min\!\left(\hat{\mathbf{v}}_i,\; c\right), \quad c=0.2
   ```

- Renormalisation:

   ```math
   \mathbf{d} = \frac{\hat{\mathbf{v}}}{\sqrt{\varepsilon + \lVert\hat{\mathbf{v}}\rVert_2^{2}}}
   ```

- CUDA kernels:
  - blockKernel: Computes one 36-dimensional L2-Hys descriptor for each block, One thread = one block (after a 1-D mapping). Uses registers for the 36-value vector, unrolled loops for speed, no shared memory needed because data reuse is trivial. |

2) Implementation

Compiling the code:

<pre>nvcc .\hog_block.cu -o hog_block</pre>

Running the code after compiling:

<pre>hog_block</pre>

<pre>Cells grid              : 512 × 256  (131072 cells)
Blocks (out)            : 511 × 255  (130305 blocks)

CPU total               : 27.5786 ms
GPU H2D copy            : 1.14848 ms
GPU kernel              : 0.347328 ms
GPU D2H copy            : 4.01638 ms
GPU total               : 14.8502 ms
RMSE  descriptors       : 5.92794e-09</pre>