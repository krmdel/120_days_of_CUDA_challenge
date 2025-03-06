Day 4: Memory and Tiling

1) Resources:

The lecture 5 of GPU Computing by Dr. Izzat El Hajj  
Lecture 5: https://youtu.be/31ZyYkoClT4?si=GekGWjfkUBABmL7Y

2) Topics Covered:

- Performance metrics
- Performance bound: memory/compute-bound
- Memory in GPU architecture
- Tiled Matrix-Matrix Multiplication

3) Summary of the Lecture:  

- Performance metrics:  
    FLOPS rate: floating point operations per second
    Memory Bandwidth: bytes per second  

Example:

+ Vector Addition: z[i] = x[i] + y[i] --> 1 FP operation for every 2 FP values (8 Bytes), 1/8 = 0.125 OP/B --> memory-bound/no reuse
+ Matrix Multiplication: 1 FP add + 1 FP mul for every 2 FP values (8 Bytes), 2/8 = 0.25 OP/B --> compute-bound/reuse  
Data Load: (2 input matrices) x (N^2 values) x (4 Bytes) = 8N^2 B
Operations: (N^2 dot products) x (N adds + N muls each) = 2N^3 OP
Compute-to-memory ratio: 0.25N OP/B

- Memory in GPU architecture:  
    - Global Memory
    - Constant Memory
    - Shared Memory
    - Registers
    
Memory      Scope       Lifetime  
---------------------------------  
Global      Grid      Application  
Constant    Grid      Application  
Shared      Block     Block  
Registers   Thread    Thread  
Global      Thread    Thread  

cudaMalloc(...) --> Global Memory

- Tiled Matrix-Matrix Multiplication:  
    - Divide the matrix into tiles  
    - Load tiles into shared memory  
    - Perform matrix multiplication on the tiles  
    - Boundary conditions: the thread in input matrices projected to output matrix can correspond to different orientation in the tile. Therefore, memory access management should be carefully implemented (e.g., different boundary conditions are required for different accesses)

4) Implementation:

The code performs matrix multiplication on the GPU for tiled matrix operations

Compiling the code:  

<pre> nvcc matmul_tiled_equaldim.cu -o matmultiledeq </pre>  

<pre> nvcc matmul_tiled_equaldim.cu -o matmultileduneq </pre>

Runnung the code after compiling: 
<pre> matmultiledeq </pre>  

<pre> matmultileduneq </pre>

The code was executed on the NVIDIA GeForce RTX 3080Ti Laptop GPU - 16 GB. The output of the code is as follows:

matmultiledeq:

<pre>Time for memory allocation: 1.361728 ms
Time for host-to-device copy: 0.916384 ms
Time for kernel execution: 2.566272 ms
Time for device-to-host copy: 1.052384 ms
</pre>

matmultileduneq:

<pre>Time for memory allocation: 1.268736 ms
Time for host-to-device copy: 0.895936 ms
Time for kernel execution: 2.686624 ms
Time for device-to-host copy: 1.192928 ms
</pre>
