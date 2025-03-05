Day 3: GPU Architecture

1) Resources:

The lecture 4 of GPU Computing by Dr. Izzat El Hajj  
Lecture 4: https://youtu.be/pBQJAwogMoE?si=T9vRwQLJWqRMD5TW  

2) Topics Covered:

- GPU architecture and subcomponents
- Streaming Multiprocessors (SMs) and CUDA cores
- Thread blocks and warps
- Thread scheduling
- Synchronization and occupancy

3) Summary of the Lecture:  

- All threads in the same block is assigned to the same SM, the remaining blocks wait until other blocks finish their execution before they are assigned to the SM
- Synchronization is done:  
    - Barrier Synchronization: wait for all threads in the block to reach to be executed before proceeding
    - Memory Synchronization: wait for all threads in the block to access fast memory
- All threads in the block assigned to SM simultaneously, block cannot be assigned to an SM until it secures enough resources for all its threads
- Transparent Scalability: Threads in different blocks are not synchoronized but blocks can be executed in any order. This works in a way that the blocks are executed sequentially if device has few SMs but in parallel if device has many SMs
- Do not write code to synchoronize blocks! If some blocks are not scheduled and executed, all other blocks would wait for them if all blocks are synchoronized, leading to deadlock
- Thread Scheduling:  
    - Threads assigned to an SM run concurrently  
    - SM has scheduler to manage execution  
    - Blocks are further divided into warps which are the unit of scheduling  
    - Size of warp is device specific but typically 32 threads  
    - Threads in a warp are scheduled together and execute Single Instruction Multiple Data (SIMD) model  
    - SIMD model: one instruction is fetched and executed by all threads in a warp, each processing different data  
        - Advantage of SIMD: share the same instruction fetch/dispatch unit across cores  
        - Disadvantage of SIMD: different threads taking different execution paths result in Control Divergence  
    - Control Divergence: warp does a pass over each unique execution path and in each pass, threads taking the path execute while the others are disabled  
    - SIMD Efficiency: the percentage of threads/cores enabled during SIMD execution  
    - Latency Hiding: when a warp needs to wait for a high latency operation, the scheduler switches to another warp for execution  
- Occupancy: ratio of the warps/threads active on the SM to the maximum allowed. Maximizing occupancy in general improves latency hiding  

Example: GPU allows assignment of certain numbers of threads per SM, threads per block, blocks per SM. Therefore, if 1024 threads/block, 2048 threads/SM, 32 blocks/SM are allowed:  

(2048 threads)/(256 threads/block) = 8 blocks < 32 blocks/SM --> Feasible  
(2048 threads)/(32 threads/block) = 64 blocks > 32 blocks/SM --> Infeasible  
(2048 threads)/(768 threads/block) = 2 blocks & 512 unassigned threads which cannot be used (2048 is not divisible by 768) --> Infeasible  

4) Implementation:

The code performs matrix multiplication on the GPU for two matrices having same dimensions (NxN, NxN --> NxN) but also in different dimensions (NxM, MxK --> NxK)

Compiling the code:  

<pre> nvcc matmulequaldim.cu -o matmuleq </pre>
<pre> nvcc matmulunequaldim.cu -o matmuluneq </pre>
<pre> nvcc resource.cu -o resource </pre>

Runnung the code after compiling: 
<pre> matmuleq </pre>
<pre> matmuluneq </pre>
<pre> resource </pre>

The code was executed on the NVIDIA GeForce RTX 3080Ti Laptop GPU - 16 GB. The output of the code is as follows:

matmuleq:

<pre>Time for memory allocation: 0.846848 ms
Time for host-to-device copy: 0.926560 ms
Time for kernel execution: 3.344672 ms
Time for device-to-host copy: 0.933760 ms
</pre>

matmuluneq:

<pre>Time for memory allocation: 0.757920 ms
Time for host-to-device copy: 0.561920 ms
Time for kernel execution: 1.790944 ms
Time for device-to-host copy: 0.803136 ms
</pre>

resource:

<pre>GPU Name: NVIDIA GeForce RTX 3080 Ti Laptop GPU
Total Global Memory: 15 GB
Number of SMs: 58
Max threads per multiprocessor: 1536
Max threads per block: 1024
Max blocks per multiprocessor: 16
Warp size: 32
</pre>