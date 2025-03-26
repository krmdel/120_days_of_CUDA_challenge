#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void childKernel(int *d_data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        d_data[idx] *= 2;
    }
}

__global__ void parentKernel(int *d_data, int n) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int threadsPerBlock = 256;
        int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;
        childKernel<<<blocks, threadsPerBlock>>>(d_data, n);
    }
}

int main(void) {
    int n = 1 << 20; // 1M elements
    size_t size = n * sizeof(int);

    // Allocate and initialize host memory
    int *h_data = (int*) malloc(size);
    for (int i = 0; i < n; i++) {
        h_data[i] = i;
    }

    // Allocate device memory
    int *d_data;
    cudaMalloc((void**)&d_data, size);

    cudaEvent_t startTotal, stopTotal, start, stop;
    cudaEventCreate(&startTotal);
    cudaEventCreate(&stopTotal);
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(startTotal, 0);

    // Host to Device copy timing
    cudaEventRecord(start, 0);
    cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float timeHTD;
    cudaEventElapsedTime(&timeHTD, start, stop);

    // Kernel execution timing (parent and dynamic child kernel)
    cudaEventRecord(start, 0);
    // Launch parent kernel with 1 block and 32 threads
    parentKernel<<<1, 32>>>(d_data, n);
    // Synchronize on the host to wait for the parent kernel and all child kernels
    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float timeKernel;
    cudaEventElapsedTime(&timeKernel, start, stop);

    // Device to Host copy timing
    cudaEventRecord(start, 0);
    cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float timeDTH;
    cudaEventElapsedTime(&timeDTH, start, stop);

    cudaEventRecord(stopTotal, 0);
    cudaEventSynchronize(stopTotal);
    float totalTime;
    cudaEventElapsedTime(&totalTime, startTotal, stopTotal);

    printf("Time for CPU to GPU copy: %f ms\n", timeHTD);
    printf("Time for kernel execution: %f ms\n", timeKernel);
    printf("Time for GPU to CPU copy: %f ms\n", timeDTH);
    printf("Total time: %f ms\n", totalTime);

    // Clean up
    cudaEventDestroy(startTotal);
    cudaEventDestroy(stopTotal);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_data);
    free(h_data);

    return 0;
}
