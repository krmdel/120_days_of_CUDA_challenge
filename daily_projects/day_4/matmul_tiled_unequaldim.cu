#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define TILE_WIDTH 32

// Tiled Matrix Multiplication Kernel for A (MxK), B (KxN) and C (MxN)
__global__ void matmul_tiled_kernel(float* A, float* B, float* C, int M, int K, int N) {
    __shared__ float ds_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float ds_B[TILE_WIDTH][TILE_WIDTH];

    int Row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int Col = blockIdx.x * TILE_WIDTH + threadIdx.x;

    float Cvalue = 0.0f;

    for (int ph = 0; ph < (K + TILE_WIDTH - 1) / TILE_WIDTH; ph++) {
        if (Row < M && ph * TILE_WIDTH + threadIdx.x < K)
            ds_A[threadIdx.y][threadIdx.x] = A[Row * K + ph * TILE_WIDTH + threadIdx.x];
        else
            ds_A[threadIdx.y][threadIdx.x] = 0.0f;

        if (Col < N && ph * TILE_WIDTH + threadIdx.y < K)
            ds_B[threadIdx.y][threadIdx.x] = B[(ph * TILE_WIDTH + threadIdx.y) * N + Col];
        else
            ds_B[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads(); 

        for (int k = 0; k < TILE_WIDTH; k++) {
            Cvalue += ds_A[threadIdx.y][k] * ds_B[k][threadIdx.x];
        }
        __syncthreads();  
    }

    if (Row < M && Col < N) {
        C[Row * N + Col] = Cvalue;
    }
}

void matmul_gpu(float* A, float* B, float* C, int M, int K, int N) {
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

    // Allocate memory on GPU for matrices A, B, and C
    cudaEventRecord(start_alloc, 0);
    cudaMalloc((void**)&A_d, M * K * sizeof(float));
    cudaMalloc((void**)&B_d, K * N * sizeof(float));
    cudaMalloc((void**)&C_d, M * N * sizeof(float));
    cudaDeviceSynchronize();
    cudaEventRecord(stop_alloc, 0);
    cudaEventSynchronize(stop_alloc);
    cudaEventElapsedTime(&time_alloc, start_alloc, stop_alloc);
    printf("Time for memory allocation: %f ms\n", time_alloc);

    // Copy matrices from CPU to GPU
    cudaEventRecord(start_h2d, 0);
    cudaMemcpy(A_d, A, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B, K * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    cudaEventRecord(stop_h2d, 0);
    cudaEventSynchronize(stop_h2d);
    cudaEventElapsedTime(&time_h2d, start_h2d, stop_h2d);
    printf("Time for host-to-device copy: %f ms\n", time_h2d);

    // Configure and launch the tiled matrix multiplication kernel
    cudaEventRecord(start_kernel, 0);
    dim3 threadsPerBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 blocksPerGrid((N + TILE_WIDTH - 1) / TILE_WIDTH, (M + TILE_WIDTH - 1) / TILE_WIDTH);
    matmul_tiled_kernel<<<blocksPerGrid, threadsPerBlock>>>(A_d, B_d, C_d, M, K, N);
    cudaDeviceSynchronize();
    cudaEventRecord(stop_kernel, 0);
    cudaEventSynchronize(stop_kernel);
    cudaEventElapsedTime(&time_kernel, start_kernel, stop_kernel);
    printf("Time for kernel execution: %f ms\n", time_kernel);

    // Copy the result matrix from GPU to CPU
    cudaEventRecord(start_d2h, 0);
    cudaMemcpy(C, C_d, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    cudaEventRecord(stop_d2h, 0);
    cudaEventSynchronize(stop_d2h);
    cudaEventElapsedTime(&time_d2h, start_d2h, stop_d2h);
    printf("Time for device-to-host copy: %f ms\n", time_d2h);

    // Free GPU memory and clean up events
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
    cudaDeviceSynchronize();

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
   
    int M = 1023;
    int K = 1025; 
    int N = 1027; 

    float *A = (float*)malloc(M * K * sizeof(float));
    float *B = (float*)malloc(K * N * sizeof(float));
    float *C = (float*)malloc(M * N * sizeof(float));

    for (int i = 0; i < M * K; i++) {
        A[i] = 1.0f;
    }
    for (int i = 0; i < K * N; i++) {
        B[i] = 2.0f;
    }

    // Perform tiled matrix multiplication on the GPU
    matmul_gpu(A, B, C, M, K, N);

    free(A);
    free(B);
    free(C);
    return 0;
}
