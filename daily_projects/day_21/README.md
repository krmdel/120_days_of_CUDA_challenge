Day 21: Dynamic Parallelism

1) Resources:

The lecture 22 of GPU Computing by Dr. Izzat El Hajj  
Lecture 22: https://youtu.be/R3d_ECmHAiI?si=etMnTQJJPe2H3s29

2) Topics Covered:

- Dynamic parallelism

3) Summary of the Lecture:  

- CUDA dynamic parallelism refers to the ability of threads executing on the GPU to launch new grids
- Nested parallelism:
    - Dynamic parallelism is useful for programming applications with nested parallelism where each thread discovers more work that can be parallelized
    - Dynamic parallelism is particularly useful when the amount of nested work is unkown, so enough threads cannot be launched upfront

- Applications of dynamic parallelism:
    - Amount of nested work may be unknown if:
        - Nested parallel work is irregular (varies across threads): e.g., graph algorithms (each vertex has a different number of neighbors), Bezier curves (each curve needs different number of points to draw)
        - Nested parallel work is recursive with unkown depth: e.g., tree traversal algorithms (e.g., quadtrees, octrees), divide and conquer algorithms (e.g., quicksort)

- Dynamic parallelism API:
    - The device code for calling a kernel to launch a grid is the same as the host code
    - Memory is needed for buffering grid launches that have not strated executing:
        - The limit on the number of dynamic launches is referred to as the pending launch count
        - By default, the runtime supporss 2048 launches, and exceeding this limit will cause an error
        - The limit can be incerased by setting: cudaDeviceSetLimit(cudaLimitDevRuntimePendingLaunchCount, < new limit >)

- Streams:
    - Recall: without specifying a stream when calling a kernel, grids get launched into a default stream
    - For device launches, threads in the same block share the same default: launches by threads in the same block are serilaized
- Per-Thread Streams:
    - Parallelism can be improved by creating a different steam per thread
        - Approach 1: use stream API just like on host
        - Approach 2: use compiler flag --default-stream per-thread

- Optimizations:
    - Pitfalls:
        - Launching very small grids may not be worth the overhead (more efficient to serialize)
        - Lauching too many grids causes queuing delays
    - Optimization: apply a threshold to the launch
        - Only launch the large grids that are worth the overhead and serialize the rest
        - Threshold  value is data dependent and can be tuned
    - Optimization: aggregate launches
        - Have one thread collect the work of multiple threads and launch a single grid on their behalf

- Offloading driver code:
    - In some applications, the host code that drives the computation launches multiple consecutive grids to synchronize across all threads between launches (e.g., BFS launches a new grid for each level)
    - Another application of dynamic parallelism is to offload the driver code to the device: main advantage is to free up the host to do other things

- Memory visibility:
    - Operations on global memory made by a parent thread before the launch are visible to the child
        - Operations made by the child are visible to the parent after the child returns and the parent has synchronized
    - A thread's local memory and a block's shared memory cannot be accessed by child threads

- Nesting Depth: The nesting depth refers to how deeply dynamically launched grids may launch other grids (determined by the hardware, the typical value is 24)

4) Implementation:

The code performs graph processing II on the GPU

Compiling the code:  

<pre>nvcc .\graph_optimization.cu -o graphopt</pre>

Running the code after compiling: 
<pre> graphopt </pre>

<pre>Top-Down BFS Timings (ms): Copy H->D: 0.482304, Kernel: 0.996096, Copy D->H: 0.001728, Total: 1.522848
Bottom-Up BFS Timings (ms): Copy H->D: 0.319136, Kernel: 0.242688, Copy D->H: 0.001824, Total: 0.586048
Direction-Optimized BFS Timings (ms): Copy H->D: 0.349824, Kernel: 0.201088, Copy D->H: 0.001728, Total: 0.576448
Edge-Centric BFS Timings (ms): Copy H->D: 0.342080, Kernel: 0.240512, Copy D->H: 0.001728, Total: 0.619232
Optimized Multi-Block BFS Timings (ms): Copy H->D: 0.311488, Kernel: 0.227168, Copy D->H: 0.001760, Total: 0.563776
Optimized Single-Block BFS Timings (ms): Copy H->D: 0.000000, Avg Kernel: 0.029180, Copy D->H: 0.001632, Total: 0.227168</pre>
