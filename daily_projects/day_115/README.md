Day 115: Implementation of FAST-9 corner detector in CUDA

1) Summary of the daily tutorial

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