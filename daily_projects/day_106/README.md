Day 106: Implementation of pitched memory and overlapped 2D transfers in CUDA

1) Summary of the daily tutorial

The code implements pitched memory and overlapped 2D transfers. A 4 K × 4 K (≈ 64 M-pixel) 8-bit image is filtered with a 15 × 15 box (mean) filter. Instead of sending the whole frame to the GPU at once, the program slices it into 256-row “stripes” that are processed in a ping-pong fashion using two CUDA streams. Each stream owns its own pair of pitched device buffers (d_src[i], d_dst[i]) so that H2D copy → kernel → D2H copy of stripe N can occur while stripe N + 1 is already on the bus — fully overlapping PCIe traffic with computation.
Pitched memory (cudaMallocPitch / cudaMemcpy2DAsync) eliminates manual row-padding and guarantees coalesced accesses even when the row length isn’t a multiple of the memory-transaction size.

```math
\text{dst}(x,y)=\frac{1}{15\times15}\sum_{i=-7}^{7}\sum_{j=-7}^{7}\text{src}(x+i,\,y+j)
```

- CUDA kernels:
    - box15: 15 × 15 mean filter operating on a 32 × 8 tile, loads a TILE_Y + 2 R by TILE_X + 2 R patch (30 × 64 bytes) into shared memory and re-uses those pixels to compute 256 output samples per block with just one global read per input byte. Handles halos so the same kernel works for every stripe, including the first and last.

2) Implementation

Compiling the code:

<pre>nvcc .\overlapped_transfers.cu -o overlapped_transfers</pre>

Running the code after compiling:

<pre>overlapped_transfers</pre>

<pre>CPU total               : 1353.72 ms
GPU H2D (sum)           : 5.13754 ms
GPU kernels (sum)       : 1.37325 ms
GPU D2H (sum)           : 1.43376 ms
GPU total               : 8.4951 ms
Mismatch pixels         : 0</pre>