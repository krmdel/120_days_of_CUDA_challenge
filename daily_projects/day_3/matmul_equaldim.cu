#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void matmul_kernel(float* a, float* b, float* c, unsigned int N) {
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0f;
        for (unsigned int i = 0; i < N; i++) {
            sum += a[row * N + i] * b[i * N + col];
        }
        c[row * N + col] = sum;
    }
}

void matmul_gpu(float* a, float* b, float* c, unsigned int N) {
    cudaEvent_t start_alloc, stop_alloc;
    cudaEvent_t start_h2d, stop_h2d;
    cudaEvent_t start_kernel, stop_kernel;
    cudaEvent_t start_d2h, stop_d2h;
    float time_alloc, time_h2d, time_kernel, time_d2h;

    cudaEventCreate(&start_alloc);
    cudaEventCreate(&stop_alloc);
    cudaEventCreate(&start_h2d);
    cudaEventCreate(&stop_h2d);
    cudaEventCreate(&start_kernel);
    cudaEventCreate(&stop_kernel);
    cudaEventCreate(&start_d2h);
    cudaEventCreate(&stop_d2h);

    float *a_d, *b_d, *c_d;

    // Allocate memory on GPU
    cudaEventRecord(start_alloc, 0);
    cudaMalloc((void**)&a_d, N * N * sizeof(float));
    cudaMalloc((void**)&b_d, N * N * sizeof(float));
    cudaMalloc((void**)&c_d, N * N * sizeof(float));
    cudaDeviceSynchronize();
    cudaEventRecord(stop_alloc, 0);
    cudaEventSynchronize(stop_alloc);
    cudaEventElapsedTime(&time_alloc, start_alloc, stop_alloc);
    printf("Time for memory allocation: %f ms\n", time_alloc);

    // Copy data from CPU to GPU
    cudaEventRecord(start_h2d, 0);
    cudaMemcpy(a_d, a, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    cudaEventRecord(stop_h2d, 0);
    cudaEventSynchronize(stop_h2d);
    cudaEventElapsedTime(&time_h2d, start_h2d, stop_h2d);
    printf("Time for host-to-device copy: %f ms\n", time_h2d);

    // Calling the kernel and performing the operation
    cudaEventRecord(start_kernel, 0);
    dim3 numThreadsPerBlock(32, 32);
    dim3 numBlocks((N + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x,
                   (N + numThreadsPerBlock.y - 1) / numThreadsPerBlock.y);
    matmul_kernel<<<numBlocks, numThreadsPerBlock>>>(a_d, b_d, c_d, N);
    cudaDeviceSynchronize();
    cudaEventRecord(stop_kernel, 0);
    cudaEventSynchronize(stop_kernel);
    cudaEventElapsedTime(&time_kernel, start_kernel, stop_kernel);
    printf("Time for kernel execution: %f ms\n", time_kernel);

    // Copy the result from GPU to CPU
    cudaEventRecord(start_d2h, 0);
    cudaMemcpy(c, c_d, N * N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    cudaEventRecord(stop_d2h, 0);
    cudaEventSynchronize(stop_d2h);
    cudaEventElapsedTime(&time_d2h, start_d2h, stop_d2h);
    printf("Time for device-to-host copy: %f ms\n", time_d2h);

    // Free GPU memory
    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(c_d);
    cudaDeviceSynchronize();

    // Clean up events
    cudaEventDestroy(start_alloc);
    cudaEventDestroy(stop_alloc);
    cudaEventDestroy(start_h2d);
    cudaEventDestroy(stop_h2d);
    cudaEventDestroy(start_kernel);
    cudaEventDestroy(stop_kernel);
    cudaEventDestroy(start_d2h);
    cudaEventDestroy(stop_d2h);
}

int main() {
    unsigned int N = 1024; // Dimension of the square matrices.

    // Allocate memory on CPU for matrices A, B, and C
    float *a = (float*)malloc(N * N * sizeof(float));
    float *b = (float*)malloc(N * N * sizeof(float));
    float *c = (float*)malloc(N * N * sizeof(float));

    // Initialize matrices A and B with sample values
    for (int i = 0; i < N * N; i++) {
        a[i] = 1.0f; // For example, fill A with 1's.
        b[i] = 2.0f; // For example, fill B with 2's.
    }

    // Call the function to perform matrix multiplication
    matmul_gpu(a, b, c, N);

    // Free CPU memory
    free(a);
    free(b);
    free(c);

    return 0;
}
