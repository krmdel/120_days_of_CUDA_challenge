Day 7: Convolution

1) Resources:

The lecture 8 of GPU Computing by Dr. Izzat El Hajj  
Lecture 8: https://youtu.be/xEVyTZG1wlk?si=cn7zvcHYE0A5ROp2

2) Topics Covered:

- Fundamentals of Convolution
- Applications of Convolution

3) Summary of the Lecture:  

- Every output element is a weighted sum of the neighboring input elements (e.g., image bluring is a special case of convolution where all weights are the same)
- Convolution mask determines the weights of the neighboring elements
- Some applications of convolution are signal processing, image processing etc. For example, Gaussian blur sharpen, edge detection etc.
- Convolution can be applied in 1D, 2D, 3D.
- Mask:
    - is typically a small matrix
    - is constant (weights do not change)
    - is accessed by all threads in the grid
    - is stored in constant memory for optimization

4) Implementation:

The code performs matrix multiplication on the GPU with memory coalescing and thread coarsening

Compiling the code:  

<pre>nvcc matmul_optimized.cu -o matmulopt</pre>

Runnung the code after compiling: 
<pre>matmulopt</pre>

The code was executed on the NVIDIA GeForce RTX 3080Ti Laptop GPU - 16 GB. The output of the code is as follows:

<pre>=== Baseline Kernel ===
Time for memory allocation:    0.948224 ms
Time for host-to-device copy:  1.015040 ms
Time for kernel execution:     3.461152 ms
Time for device-to-host copy:  0.889920 ms

=== Coalesced + Coarsened Kernel ===
Time for memory allocation:    0.787680 ms
Time for host-to-device copy:  0.988288 ms
Time for kernel execution:     1.052672 ms
Time for device-to-host copy:  0.888544 ms
</pre>
