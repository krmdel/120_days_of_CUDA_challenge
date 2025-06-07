Day 96: Implementation of Harris response with shared memory in CUDA

1) Summary of the daily tutorial

The code implements the corner-detection by porting Harris response computation to the GPU and increases performance with shared-memory reduction.
For every pixel, the structure-tensor elements I_x^2, I_y^2 and I_{xy} are already known (from gradient images). The kernel computes each pixel’s Harris score:

  ```math
  R = (I_x^2\,I_y^2 - I_{xy}^2) - k\,(I_x^2 + I_y^2)^2
  ```
  
- writes the score back to global memory
- performs an in-block reduction in shared memory to keep the maximum response per block (handy for later non-maximum-suppression steps).

- CUDA kernels:
    - harrisKernel: All-in-one pixel loop: computes R for the thread’s pixel, stores R, reduces to block-wide max in shared memory, uses constant memory for scalar k, dynamic shared memory sized as threads * sizeof(float) for fast reduction and final max per block is written to blockMax.

2) Implementation

Compiling the code:

<pre>nvcc .\harris_shared.cu -o harris_shared</pre>

Running the code after compiling:

<pre>harris_shared</pre>

<pre>Generated random tensors 4096×4096
CPU total               : 54.1351 ms
GPU H2D copy            : 42.5955 ms
GPU kernel              : 0.248832 ms
GPU D2H copy            : 13.8444 ms
GPU total               : 90.4558 ms
RMSE (CPU vs GPU)       : 0.000610891</pre>