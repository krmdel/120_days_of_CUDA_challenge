Day 115: Implementation of FAST-9 corner detector in CUDA

1) Summary of the daily tutorial

The code implements the FAST-9 corner detector on both CPU and GPU and compares the number of detected key-points. The GPU version parallelises the test for every pixel inside a 16-pixel border, performs the two-step FAST check (4-point early reject + 16-point circle test), and writes key-points to global memory with an atomic counter.

FAST-9 decision rule

```math
\text{corner}(p)=
\begin{cases}
1 &
\bigl(\exists\,9\text{ contiguous } q_i \in C_{16} :
\;|I(q_i)-I(p)| > t\bigr) \\[6pt]
0 & \text{otherwise}
\end{cases}
```

where C_16 is the 16-pixel circle of radius ≈ 8–9 px, I denotes intensity, and t is the threshold.

- CUDA kernels:
	- fast9Kernel: Launch configuration: 16 × 16 threads; one thread per image pixel. Early reject on four compass pixels at radius 9. Full 16-pixel circle test with a rolled 24-iteration loop to find ≥ 9 contiguous bright or dark pixels. atomicAdd on a global counter to reserve an output slot; writes (x,y) to KP array.

2) Implementation

Compiling the code:

<pre>nvcc .\fast8_corner.cu -o fast8_corner</pre>

Running the code after compiling:

<pre>fast8_corner</pre>

<pre>[Info] random frame 1920×1080
Image                   : 1920 × 1080
Threshold               : 20
GPU keypoints           : 80000
CPU keypoints           : 869785

GPU H2D copy            : 0.620608 ms
GPU kernel              : 0.256 ms
GPU D2H copy            : 0.296032 ms
GPU total               : 1.1928 ms
CPU total               : 197.235 ms</pre>