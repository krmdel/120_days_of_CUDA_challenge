#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define TILE_WIDTH 32

// Tiled Matrix Multiplication Kernel using Shared Memory
__global__ void matmul_tiled_kernel(float* A, float* B, float* C, unsigned int N) {
    __shared__ float ds_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float ds_B[TILE_WIDTH][TILE_WIDTH];

    // Calculate block index and thread index
    unsigned int bx = blockIdx.x;
    unsigned int by = blockIdx.y;
    unsigned int tx = threadIdx.x;
    unsigned int ty = threadIdx.y;

    // Determine the row and column of the C element to work on
    unsigned int Row = by * TILE_WIDTH + ty;
    unsigned int Col = bx * TILE_WIDTH + tx;
    
    float Cvalue = 0.0f;

    // Loop over the tiles of A and B required to compute C[Row][Col]
    // The number of phases is ceil(N/TILE_WIDTH)
    for (unsigned int ph = 0; ph < (N + TILE_WIDTH - 1) / TILE_WIDTH; ph++) {
        // Load a tile of matrix A into shared memory
        if (Row < N && (ph * TILE_WIDTH + tx) < N)
            ds_A[ty][tx] = A[Row * N + ph * TILE_WIDTH + tx];
        else
            ds_A[ty][tx] = 0.0f;

        // Load a tile of matrix B into shared memory
        if (Col < N && (ph * TILE_WIDTH + ty) < N)
            ds_B[ty][tx] = B[(ph * TILE_WIDTH + ty) * N + Col];
        else
            ds_B[ty][tx] = 0.0f;

        __syncthreads();  // Make sure the tile is loaded

        // Multiply the two tiles together
        for (unsigned int k = 0; k < TILE_WIDTH; k++) {
            Cvalue += ds_A[ty][k] * ds_B[k][tx];
        }
        __syncthreads();  // Ensure all threads have computed with the current tile
    }

    // Write the computed value to global memory
    if (Row < N && Col < N)
        C[Row * N + Col] = Cvalue;
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

    // Allocate memory on the GPU
    cudaEventRecord(start_alloc, 0);
    cudaMalloc((void**)&a_d, N * N * sizeof(float));
    cudaMalloc((void**)&b_d, N * N * sizeof(float));
    cudaMalloc((void**)&c_d, N * N * sizeof(float));
    cudaDeviceSynchronize();
    cudaEventRecord(stop_alloc, 0);
    cudaEventSynchronize(stop_alloc);
    cudaEventElapsedTime(&time_alloc, start_alloc, stop_alloc);
    printf("Time for memory allocation: %f ms\n", time_alloc);

    // Copy data from host to device
    cudaEventRecord(start_h2d, 0);
    cudaMemcpy(a_d, a, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    cudaEventRecord(stop_h2d, 0);
    cudaEventSynchronize(stop_h2d);
    cudaEventElapsedTime(&time_h2d, start_h2d, stop_h2d);
    printf("Time for host-to-device copy: %f ms\n", time_h2d);

    // Launch the tiled matrix multiplication kernel
    cudaEventRecord(start_kernel, 0);
    dim3 numThreadsPerBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 numBlocks((N + TILE_WIDTH - 1) / TILE_WIDTH,
                   (N + TILE_WIDTH - 1) / TILE_WIDTH);
    matmul_tiled_kernel<<<numBlocks, numThreadsPerBlock>>>(a_d, b_d, c_d, N);
    cudaDeviceSynchronize();
    cudaEventRecord(stop_kernel, 0);
    cudaEventSynchronize(stop_kernel);
    cudaEventElapsedTime(&time_kernel, start_kernel, stop_kernel);
    printf("Time for kernel execution: %f ms\n", time_kernel);

    // Copy the result from device to host
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
    unsigned int N = 1024; // Dimension of the square matrices

    // Allocate memory on CPU for matrices A, B, and C
    float *a = (float*)malloc(N * N * sizeof(float));
    float *b = (float*)malloc(N * N * sizeof(float));
    float *c = (float*)malloc(N * N * sizeof(float));

    // Initialize matrices A and B with sample values
    for (int i = 0; i < N * N; i++) {
        a[i] = 1.0f; // For example, fill A with 1's
        b[i] = 2.0f; // For example, fill B with 2's
    }

    // Perform tiled matrix multiplication on the GPU
    matmul_gpu(a, b, c, N);

    // Free CPU memory
    free(a);
    free(b);
    free(c);

    return 0;
}
