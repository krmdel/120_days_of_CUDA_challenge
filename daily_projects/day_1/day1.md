Day 1: Introduction to GPU Computing and Data Parallel Programming

1) Resources:

The lecture 1&2 of GPU Computing by Dr. Izzat El Hajj
Lecture 1: https://youtu.be/4pkbXmE4POc?si=KBuRCSvqUNrdQjR6
Lecture 2: https://youtu.be/iE-xGWBQtH0?si=srMbPD6fx12dCcs_

2) Topics Covered:
- The basics of GPU architecture
- The basics of CUDA programming
- The implementation of vector addition using CUDA

3) Summary of the Lecture:
The memory management and computation strategy:

- Allocate memory on the device (GPU)
- Copy the data from the host (CPU) to the device (GPU)
- Launch the kernel (function) on the device (GPU)
- Copy the data from the device (GPU) to the host (CPU)
- Deallocate the memory on the device (GPU)

The one GPU thread is responsible for performing an operation via kernel. The kernel is a function that is executed on the GPU. The kernel is launched with a specific number of blocks and threads. 

4) Implementation:

The vector addition is implemented in CUDA. The vector addition is a simple operation that adds two vectors and stores the result in the third vector. The vector addition is implemented in the CPU and GPU. The time taken by the CPU and GPU is compared. The GPU kernel is launched with 512 threads per block. The vector addition is implemented in the vector_addition.cu file.

Compiling the code: nvcc vector_addition.cu -o vecadd
Runnung the code after compiling: vecadd

The code was executed on the NVIDIA GeForce RTX 3080Ti Laptop GPU - 16 GB. The output of the code is as follows:

CPU vecadd elapsed time: 72.6874 ms
GPU kernel elapsed time: 1.83034 ms
Overall GPU vecadd elapsed time: 55.8308 ms