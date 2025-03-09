#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void matmul_kernel_baseline(const float* A, const float* B, float* C, unsigned int N)
{
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0f;
        for (unsigned int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}


#define BLOCK_SIZE 16
#define COARSENING 2

__global__ void matmul_kernel_coalesced_coarse(const float* A, const float* B, float* C, unsigned int N)
{
    
    unsigned int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    unsigned int col_start = blockIdx.x * (BLOCK_SIZE * COARSENING) + (threadIdx.x * COARSENING);

    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE * COARSENING];

    float sum[COARSENING];
    #pragma unroll
    for (int c = 0; c < COARSENING; c++) {
        sum[c] = 0.0f;
    }

    for (int tile_k = 0; tile_k < N; tile_k += BLOCK_SIZE) {
        if (row < N && (tile_k + threadIdx.x) < N) {
            As[threadIdx.y][threadIdx.x] = A[row * N + (tile_k + threadIdx.x)];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }

        #pragma unroll
        for (int c = 0; c < COARSENING; c++) {
            unsigned int b_col = col_start + c;
            unsigned int b_row = tile_k + threadIdx.y;

            if (b_row < N && b_col < N) {
                Bs[threadIdx.y][threadIdx.x * COARSENING + c] = B[b_row * N + b_col];
            } else {
                Bs[threadIdx.y][threadIdx.x * COARSENING + c] = 0.0f;
            }
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE; k++) {
            float a_val = As[threadIdx.y][k];
            #pragma unroll
            for (int c = 0; c < COARSENING; c++) {
                float b_val = Bs[k][threadIdx.x * COARSENING + c];
                sum[c] += a_val * b_val;
            }
        }

        __syncthreads();
    }

    if (row < N) {
        #pragma unroll
        for (int c = 0; c < COARSENING; c++) {
            unsigned int out_col = col_start + c;
            if (out_col < N) {
                C[row * N + out_col] = sum[c];
            }
        }
    }
}

void matmul_gpu_baseline(const float* A, const float* B, float* C, unsigned int N)
{
    // Timers
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

    // Allocate GPU memory
    float *a_d, *b_d, *c_d;
    cudaEventRecord(start_alloc, 0);
    cudaMalloc((void**)&a_d, N * N * sizeof(float));
    cudaMalloc((void**)&b_d, N * N * sizeof(float));
    cudaMalloc((void**)&c_d, N * N * sizeof(float));
    cudaDeviceSynchronize();
    cudaEventRecord(stop_alloc, 0);
    cudaEventSynchronize(stop_alloc);
    cudaEventElapsedTime(&time_alloc, start_alloc, stop_alloc);

    // Copy from CPU to GPU
    cudaEventRecord(start_h2d, 0);
    cudaMemcpy(a_d, A, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, B, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    cudaEventRecord(stop_h2d, 0);
    cudaEventSynchronize(stop_h2d);
    cudaEventElapsedTime(&time_h2d, start_h2d, stop_h2d);

    // Launch kernel and pefrom matrix multiplication
    cudaEventRecord(start_kernel, 0);
    dim3 threadsPerBlock(32, 32);
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (N + threadsPerBlock.y - 1) / threadsPerBlock.y);
    matmul_kernel_baseline<<<numBlocks, threadsPerBlock>>>(a_d, b_d, c_d, N);
    cudaDeviceSynchronize();
    cudaEventRecord(stop_kernel, 0);
    cudaEventSynchronize(stop_kernel);
    cudaEventElapsedTime(&time_kernel, start_kernel, stop_kernel);

    // Copy from GPU to CPU
    cudaEventRecord(start_d2h, 0);
    cudaMemcpy(C, c_d, N * N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    cudaEventRecord(stop_d2h, 0);
    cudaEventSynchronize(stop_d2h);
    cudaEventElapsedTime(&time_d2h, start_d2h, stop_d2h);

    // Print timing
    printf("=== Baseline Kernel ===\n");
    printf("Time for memory allocation:    %f ms\n", time_alloc);
    printf("Time for host-to-device copy:  %f ms\n", time_h2d);
    printf("Time for kernel execution:     %f ms\n", time_kernel);
    printf("Time for device-to-host copy:  %f ms\n\n", time_d2h);

    // Cleanup
    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(c_d);

    cudaEventDestroy(start_alloc);
    cudaEventDestroy(stop_alloc);
    cudaEventDestroy(start_h2d);
    cudaEventDestroy(stop_h2d);
    cudaEventDestroy(start_kernel);
    cudaEventDestroy(stop_kernel);
    cudaEventDestroy(start_d2h);
    cudaEventDestroy(stop_d2h);
}

void matmul_gpu_coalesced_coarse(const float* A, const float* B, float* C, unsigned int N)
{
    // Timers
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

    // Allocate GPU memory
    float *a_d, *b_d, *c_d;
    cudaEventRecord(start_alloc, 0);
    cudaMalloc((void**)&a_d, N * N * sizeof(float));
    cudaMalloc((void**)&b_d, N * N * sizeof(float));
    cudaMalloc((void**)&c_d, N * N * sizeof(float));
    cudaDeviceSynchronize();
    cudaEventRecord(stop_alloc, 0);
    cudaEventSynchronize(stop_alloc);
    cudaEventElapsedTime(&time_alloc, start_alloc, stop_alloc);

    // Copy from CPU to GPU
    cudaEventRecord(start_h2d, 0);
    cudaMemcpy(a_d, A, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, B, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    cudaEventRecord(stop_h2d, 0);
    cudaEventSynchronize(stop_h2d);
    cudaEventElapsedTime(&time_h2d, start_h2d, stop_h2d);

    // Launch Optimized Kernel
    cudaEventRecord(start_kernel, 0);

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((N + (BLOCK_SIZE * COARSENING) - 1) / (BLOCK_SIZE * COARSENING),
              (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    matmul_kernel_coalesced_coarse<<<grid, block>>>(a_d, b_d, c_d, N);
    cudaDeviceSynchronize();
    cudaEventRecord(stop_kernel, 0);
    cudaEventSynchronize(stop_kernel);
    cudaEventElapsedTime(&time_kernel, start_kernel, stop_kernel);

    // Copy from GPU to CPU
    cudaEventRecord(start_d2h, 0);
    cudaMemcpy(C, c_d, N * N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    cudaEventRecord(stop_d2h, 0);
    cudaEventSynchronize(stop_d2h);
    cudaEventElapsedTime(&time_d2h, start_d2h, stop_d2h);

    // Print timing
    printf("=== Coalesced + Coarsened Kernel ===\n");
    printf("Time for memory allocation:    %f ms\n", time_alloc);
    printf("Time for host-to-device copy:  %f ms\n", time_h2d);
    printf("Time for kernel execution:     %f ms\n", time_kernel);
    printf("Time for device-to-host copy:  %f ms\n\n", time_d2h);

    // Cleanup
    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(c_d);

    cudaEventDestroy(start_alloc);
    cudaEventDestroy(stop_alloc);
    cudaEventDestroy(start_h2d);
    cudaEventDestroy(stop_h2d);
    cudaEventDestroy(start_kernel);
    cudaEventDestroy(stop_kernel);
    cudaEventDestroy(start_d2h);
    cudaEventDestroy(stop_d2h);
}

int main()
{
    unsigned int N = 1024;

    // Allocate host memory
    float *A = (float*)malloc(N * N * sizeof(float));
    float *B = (float*)malloc(N * N * sizeof(float));
    float *C = (float*)malloc(N * N * sizeof(float));
    float *C_optimized = (float*)malloc(N * N * sizeof(float));

    for (int i = 0; i < N * N; i++) {
        A[i] = 1.0f;
        B[i] = 2.0f;
    }

    // Baseline
    matmul_gpu_baseline(A, B, C, N);

    // Optimized
    matmul_gpu_coalesced_coarse(A, B, C_optimized, N);

    // Free host memory
    free(A);
    free(B);
    free(C);
    free(C_optimized);

    return 0;
}
