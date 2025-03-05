#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// Kernel for matrix multiplication:
// A is N x M, B is M x K, and C is N x K.
__global__ void matmul_kernel(float* A, float* B, float* C,
                              unsigned int N, unsigned int M, unsigned int K) {
    // Compute the row index for the output matrix C
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    // Compute the column index for the output matrix C
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Check boundary conditions for the output matrix
    if (row < N && col < K) {
        float sum = 0.0f;
        // Accumulate over the shared dimension M
        for (unsigned int i = 0; i < M; i++) {
            // A is stored in row-major order: element at (row, i)
            // B is stored in row-major order: element at (i, col)
            sum += A[row * M + i] * B[i * K + col];
        }
        // Write the result to C at (row, col)
        C[row * K + col] = sum;
    }
}

void matmul_gpu(float* A, float* B, float* C,
                unsigned int N, unsigned int M, unsigned int K) {
    // Create CUDA events for timing different stages
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

    float *A_d, *B_d, *C_d;

    // Allocate memory on GPU
    cudaEventRecord(start_alloc, 0);
    cudaMalloc((void**)&A_d, N * M * sizeof(float));
    cudaMalloc((void**)&B_d, M * K * sizeof(float));
    cudaMalloc((void**)&C_d, N * K * sizeof(float));
    cudaDeviceSynchronize();
    cudaEventRecord(stop_alloc, 0);
    cudaEventSynchronize(stop_alloc);
    cudaEventElapsedTime(&time_alloc, start_alloc, stop_alloc);
    printf("Time for memory allocation: %f ms\n", time_alloc);

    // Copy data from CPU to GPU
    cudaEventRecord(start_h2d, 0);
    cudaMemcpy(A_d, A, N * M * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    cudaEventRecord(stop_h2d, 0);
    cudaEventSynchronize(stop_h2d);
    cudaEventElapsedTime(&time_h2d, start_h2d, stop_h2d);
    printf("Time for host-to-device copy: %f ms\n", time_h2d);

    // Launch the kernel
    cudaEventRecord(start_kernel, 0);
    // Define a 2D grid for the output matrix which is N x K
    dim3 numThreadsPerBlock(32, 32);
    dim3 numBlocks((K + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x,
                   (N + numThreadsPerBlock.y - 1) / numThreadsPerBlock.y);
    matmul_kernel<<<numBlocks, numThreadsPerBlock>>>(A_d, B_d, C_d, N, M, K);
    cudaDeviceSynchronize();
    cudaEventRecord(stop_kernel, 0);
    cudaEventSynchronize(stop_kernel);
    cudaEventElapsedTime(&time_kernel, start_kernel, stop_kernel);
    printf("Time for kernel execution: %f ms\n", time_kernel);

    // Copy the result from GPU to CPU
    cudaEventRecord(start_d2h, 0);
    cudaMemcpy(C, C_d, N * K * sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    cudaEventRecord(stop_d2h, 0);
    cudaEventSynchronize(stop_d2h);
    cudaEventElapsedTime(&time_d2h, start_d2h, stop_d2h);
    printf("Time for device-to-host copy: %f ms\n", time_d2h);

    // Free GPU memory
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
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
    // Dimensions for the multiplication: A is N x M, B is M x K, C is N x K.
    unsigned int N = 1024; // Number of rows in A and C
    unsigned int M = 512;  // Number of columns in A and rows in B
    unsigned int K = 768;  // Number of columns in B and C

    // Allocate memory on CPU for matrices A, B, and C
    float *A = (float*)malloc(N * M * sizeof(float));
    float *B = (float*)malloc(M * K * sizeof(float));
    float *C = (float*)malloc(N * K * sizeof(float));

    // Initialize matrices A and B with sample values
    for (unsigned int i = 0; i < N * M; i++) {
        A[i] = 1.0f; // For example, fill A with 1's.
    }
    for (unsigned int i = 0; i < M * K; i++) {
        B[i] = 2.0f; // For example, fill B with 2's.
    }

    // Call the function to perform matrix multiplication
    matmul_gpu(A, B, C, N, M, K);

    // Optionally, you could print some elements of C here for verification.

    // Free CPU memory
    free(A);
    free(B);
    free(C);

    return 0;
}
