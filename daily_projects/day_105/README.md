Day 105: Implementation of inclusive scan benchmarking in CUDA

1) Summary of the daily tutorial

The code implements a benchmark two device-side inclusive prefix-sum (scan) implementations:
- Thrust inclusive_scan: the high-level C++ STL-like interface that launches a set of internal kernels to compute the prefix sum in parallel on the GPU.
- CUB DeviceScan::InclusiveSum: a lower-level, fine-tuned primitive that usually delivers the fastest performance on NVIDIA hardware.

- Generates a large array of 32-bit unsigned integers on the host.
- Computes a reference prefix sum on the CPU.
- Copies the data to pinned host buffers and a device buffer.
- Runs the two GPU paths, timing H2D copy → kernel → D2H copy with CUDA events.
- Verifies correctness against the CPU reference and prints throughput in GB/s.

```math
\text{out}[i] \;=\; \sum_{k=0}^{i} \text{in}[k], \qquad 0 \le i < N
```

An inclusive scan returns, at every position i, the sum of all preceding elements including i.

- CUDA kernels:
  - Thrust inclusive_scan: launches a sequence of tuned kernels (load, block-scan, block-reduce, upsweep/downsweep) to compute the prefix sum, handles arbitrary iterator types and automatically chooses the best configuration.
  - CUB DeviceScan InclusiveSum: Two calls are made: one “dry run” to query the temporary-storage bytes, then the real scan. Internally uses a two-pass hierarchical scan (block-level scans followed by a block-prefix add) that maximises memory throughput and minimizes thread divergence.

2) Implementation

Compiling the code:

<pre>nvcc .\inclusive.cu -o inclusive</pre>

Running the code after compiling:

<pre>inclusive</pre>

<pre>Elements: 268435456 (1.0 GiB)

CPU serial scan: 766.489 ms

[Thrust inclusive_scan]
H2D    :   86.941 ms
Kernel :    2.338 ms   (918.4 GB/s)
D2H    :   81.353 ms
Total  :  170.649 ms
Mismatch: 0

[CUB DeviceScan]
H2D    :   86.909 ms
Kernel :    1.838 ms   (1168.5 GB/s)
D2H    :   81.356 ms
Total  :  170.117 ms
Mismatch: 0</pre>