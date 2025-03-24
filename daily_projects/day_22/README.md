Day 22: Potpourri

1) Resources:

The lecture 23 of GPU Computing by Dr. Izzat El Hajj  
Lecture 23: https://youtu.be/wCyNd662aic?si=1Gt3pqUY7fuddnvD

2) Topics Covered:

- Multi GPU programming
- Interconnect
- Memory management
- Events
- Tensor cores
- Libraries
- Other programming interfaces
- Other hardware
- Comparison with CPU

3) Summary of the Lecture:  

- Multi GPU programming:
    - Multiple GPUs on the same node
        - Use cudaGetDeviceCount(...) to get the find how many
        - Use cudaSetDevice(...) to select one
        - Typically us emultiple CPU threads with one driving each GPU (e.g., using OpenMP)
    - Multiple GPUs on different nodes
        - Typically use MPI for programming multiple nodes, and each MPI rank interacts with its GPU normally
        - Important consideration: overlapping MPI communication with kernel execution
    - Interconnect: 
        - PCIe is used in many systems to connect CPUs and GPUs together
        - NVLINK provides faster interconnect between multiple GPUs on the same system (and some CPUs)

- Unified Virtual Addressing (UVA):
    - The same virtual address space is used across the host and device(s)
    - Advantage: data location is known from the pointer value, so do not need to specify copy direction: just use cudaMemcpyDefault

- Zero-copy Memory:
    - Enables device threads to directly access host memory
    - Memory has to be pinned:
        - Use cudaHostAlloc(&ptr, size, cudaHostAllocMapped) to allocate memory
        - Use cudaHostGetDevicePointer(...) to get corresponding device pointer: unnecessary if system supports UVA

- Events:
    - We have been synchronizing after every copy or kernel call to collect timing information
        - In practice, this approach interferes with execution and slows things down
        - Events can be used to collect timing information without synchronization
            - cudaEventCreate(&event) 
            - cudaEventRecord(event, stream=0)
            - cudaEventSynchronize(event)
            - cudaEventElapsedTime(&result, startEvent, endEvent)
            - cudaEventDestroy(event)
        - The profiler can also be used to collect timing information

- Tensor Cores:
    - Tensor cores are programmable matrix-multiply-and-accumulate introduced with the Volta architecture
        - Important for accelerating deep learning workloads
    - In Volta V100,
        - 640 tensor cores (8 per SM)
        - Each core is capable of a 4x4 matrix-matrix multiply:
            D = A * B + C

- Libraries:
    - Many libraries for common parallel primitives:
        - Thrust: reduction, scan, filter and sort
        - cuBLAS: basic dense linear algebra
            - NVBLAS: multi-GPU library on top of cuBLAS
        - cuSPARSE: sparse-dense linear algebra
        - cuSOLVER: factorization and solve routines
            - Built on top of cuBLAS and cuSPARSE
        - cuDNN: deep neural networks
            - Used by Caffe, MxNet, TensorFlow, PyTorch, etc.
        - nvGRAPH: graph processing
        - cuFFT: fast Fourier transforms
        - NPP: signal, image, and video processing

- Other Programming Interfaces:
    - OpenCL: cross-platform parallel programming. Similar to CUDA but open source and more portable across non-NVIDIA GPUs
    - OpenACC: directive-based GPU programming
        - Annotate code similar to OpenMP
        - C++AMP: Library for implementing GPU programs directly in C++

- Other GPU Hardware:
    - Focus has been on NVIDIA GPUs
    - Other GPU hardware:
        - AMD: Radeon
        - ARM: Mali
        - Intel
    - Integrated GPUs:
        - CPU and GPU on the same chip (same physical memory)
        - No memory copies or data transfer over PCIe

- Comparison with CPU:
    - Comparison of GPU with CPU has been with single-threaded non-vectorized code
        - Exaggerates GPU speedup
        - This practice is generally frowned upon
    - More fair to compare with parallel and vectorized CPU code

4) Implementation:

The code performs Intra Warp Synchronization on the GPU

Compiling the code:  

<pre>nvcc .\intrawarpsync.cu -o warpsync</pre>

Running the code after compiling: 
<pre> warpsync </pre>

<pre>Timing for Baseline Reduction
Time Host-to-Device: 0.470112 ms
Kernel Time: 0.821184 ms
Time Device-to-Host: 0.043776 ms
Timing for Warp Shuffle Reduction
Time Host-to-Device: 0.453280 ms
Kernel Time: 0.049760 ms
Time Device-to-Host: 0.053472 ms</pre>
