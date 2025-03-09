Day 6: Profiling

1) Resources:

The lecture 7 of GPU Computing by Dr. Izzat El Hajj  
Lecture 7: https://youtu.be/zHY7iF_2RyU?si=7C5c-oBvj_SzmTjk

2) Topics Covered:

- Profiling: how to know the bottlenecks

3) Summary of the Lecture:  

- NVIDIA profiler to analyze the application  to reveal the bottlenecks in the code
- The profiler provides the time taken by the kernel, memory transfer etc.

For example:  

Running the profiler:  
<pre> nvprof ./vecadd </pre>

Exporting the profiler output to a file:
<pre> nvprof -o timeline.prof ./vecadd </pre>

Exporting the profiler output to a file with profiling metrics (e.g. occupancy):
<pre> nvprof -m all -o timeline.prof ./vecadd </pre>

To view the profiler output:
<pre> nvvp & </pre>

4) Implementation:

The code performs matrix multiplication on the GPU for tiled matrix operations

Compiling the code:  

<pre> nvcc matmul_tiled_sync.cu -o matmultiled </pre>

Running the code after compiling: 
<pre> matmultiledsync </pre>

The code was executed on the NVIDIA GeForce RTX 3080Ti Laptop GPU - 16 GB. The output of the code is as follows:

<pre>Matrix dimension: 1024 x 1024
=== Baseline Tiled Kernel ===
Alloc:           1.027072 ms
Host->Device:    0.943168 ms
Kernel:          2.806176 ms
Device->Host:    1.104352 ms
Total:           5.880768 ms

=== Extended Tiled Kernel (Extra sync) ===
Alloc:           0.767904 ms
Host->Device:    0.897600 ms
Kernel:          1.828128 ms
Device->Host:    0.486560 ms
Total:           3.980192 ms
</pre>
