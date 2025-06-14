Day 103: Implementation of box and Haar filters in CUDA

1) Summary of the daily tutorial

The code implements integral-image–based filters:
- 15 × 15 box filter ― returns the local mean intensity around every pixel.
- 24 × 24 horizontal Haar feature ― returns the difference between the lower and upper halves of a square window, a building-block for many object-detection pipelines.

The code
- Generates or loads a Gray-8 (PGM) frame.
- Builds an integral image entirely on the GPU using a two-pass, tile-wise prefix-scan + double transpose strategy.
- Applies the two filters in a single pass.

Using an integral image, the sum of any axis-aligned rectangle with corners is,

```math
S = \mathrm{II}(x_2,y_2) + \mathrm{II}(x_1-1,y_1-1)
    - \mathrm{II}(x_2,    y_1-1) - \mathrm{II}(x_1-1,y_2)
```

Box average (window width B=15):

```math
\text{box}(x,y) = \frac{S_{15\times15}(x,y)}{B^2}
```

Horizontal Haar response (window height H=24):

```math
\text{haar}(x,y) = S_{\text{bottom}}(x,y) \;-\; S_{\text{top}}(x,y)
```

- CUDA kernels:
  - scanRow_u8: Tile-wise prefix sum of each image row (uint8 → uint32), records each tile’s right-edge total in sums.
  - scanSums: Performs an in-place prefix sum over the sums array so each tile knows the accumulated value of all tiles to its left.
  - addOffsets: Adds these tile offsets back into every element of the tile, completing the row scan.
  - transpose32u: 32 × 32 shared-memory transpose (with a 33-column buffer to avoid bank conflicts), enables a column-prefix-scan by turning columns into rows.
  - scanRow_u32: Same as scanRow_u8 but for uint32 data during the second (column) pass.
  - boxKernel: For each pixel, reads four integral-image corners and writes the 15 × 15 local average (uint16).
  - haarKernel: Computes the 24 × 24 horizontal Haar response (bottom – top) and writes a 32-bit signed value.

2) Implementation

Compiling the code:

<pre>nvcc .\haar.cu -o haar</pre>

Running the code after compiling:

<pre>haar</pre>

<pre>Random 4096×4096
CPU total (integral+filters) : 261.817 ms
GPU H2D copy                 : 1.38326 ms
GPU kernels                  : 1.86579 ms
GPU D2H copy                 : 7.64106 ms
GPU total                    : 10.9193 ms
Box mismatches               : 0
Haar mismatches              : 0</pre>