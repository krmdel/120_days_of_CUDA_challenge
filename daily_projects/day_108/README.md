Day 108: Implementation of FP16 vs FP32 vs INT8 row-wise inclusive scan in CUDA

1) Summary of the daily tutorial

The code extends previous inclusive-scan examples to compare three numeric formats—FP32, FP16 (12-bit mantissa + 4-bit carry), and INT8 (saturating)—for building an integral image (summed-area table) row by row on the GPU. The code
- loads or synthesise an 8192 × 8192 grayscale image;
- computes a CPU reference integral image in 32-bit integers;
- runs three GPU pipelines that differ only in the value type used by thrust::inclusive_scan_by_key;
- re-assembles FP16 results into full 32-bit values, compare all outputs to the CPU baseline with PSNR, and time each stage with CUDA events.

The row-wise inclusive scan performed for every row r is

```math
S_{r,x} = \sum_{k=0}^{x} I_{r,k},
```

and, for the INT8 variant, a saturating version

```math
S^{\text{INT8}}_{r,x} = \min\!\left(255,\; \sum_{k=0}^{x} I_{r,k}\right).
```

- CUDA kernels:
    - packLow12: splits each 32-bit pixel into
        - the low 12 bits (stored as __half) and
        - the high 4 bits (stored as uint8_t),
        enabling a quasi-FP16 prefix scan with no overflow.
	- addHigh: after the FP16 scan, recombines the carried high bits with the scanned low-part to recover a full 16-bit prefix sum for every pixel.
    - SatAddU8 (binary functor): performs saturating byte addition so that the INT8 scan never exceeds 255.
	- RowKey (unary functor): maps a 1-D index → row-ID so inclusive_scan_by_key treats each row as an independent key.
	- CastByte: up-casts uint8_t input pixels to uint32_t before the FP32 scan.

2) Implementation

Compiling the code:

<pre>nvcc .\inclusive_scan.cu -o inclusive_scan</pre>

Running the code after compiling:

<pre>inclusive_scan</pre>

<pre>CPU total               : 192.053 ms

GPU FP32 kernel         : 0.822272 ms
PSNR  FP32 vs CPU       : -137.403 dB

GPU FP16 kernel         : 0.710656 ms
PSNR  FP16 vs CPU       : -137.405 dB

GPU INT8 kernel         : 0.90624 ms
PSNR  INT8 vs CPU       : -137.405 dB</pre>