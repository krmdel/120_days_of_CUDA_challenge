#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <iostream>
#include <cuda_runtime.h>

#define TILE_WIDTH 32

void matmul_cpu_tiled(const float* A, const float* B, float* C, unsigned int N)
{
    for (unsigned int i = 0; i < N*N; i++) {
        C[i] = 0.0f;
    }
    for (unsigned int iBlock = 0; iBlock < N; iBlock += TILE_WIDTH) {
        for (unsigned int jBlock = 0; jBlock < N; jBlock += TILE_WIDTH) {
            for (unsigned int kBlock = 0; kBlock < N; kBlock += TILE_WIDTH) {
                for (unsigned int i = iBlock; i < iBlock + TILE_WIDTH && i < N; i++) {
                    for (unsigned int j = jBlock; j < jBlock + TILE_WIDTH && j < N; j++) {
                        float sum = C[i*N + j];
                        for (unsigned int k = kBlock; k < kBlock + TILE_WIDTH && k < N; k++) {
                            sum += A[i*N + k] * B[k*N + j];
                        }
                        C[i*N + j] = sum;
                    }
                }
            }
        }
    }
}

__global__ void matmul_tiled_no_shared_kernel(const float* A,
                                              const float* B,
                                              float*       C,
                                              unsigned int N)
{
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0f;
        for (unsigned int k = 0; k < N; k++) {
            sum += A[row*N + k] * B[k*N + col];
        }
        C[row*N + col] = sum;
    }
}

__global__ void matmul_tiled_shared_kernel(const float* A,
                                           const float* B,
                                           float*       C,
                                           unsigned int N)
{
    __shared__ float ds_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float ds_B[TILE_WIDTH][TILE_WIDTH];

    unsigned int bx = blockIdx.x;
    unsigned int by = blockIdx.y;
    unsigned int tx = threadIdx.x;
    unsigned int ty = threadIdx.y;

    unsigned int Row = by * TILE_WIDTH + ty;
    unsigned int Col = bx * TILE_WIDTH + tx;
    
    float Cvalue = 0.0f;

    for (unsigned int ph = 0; ph < (N + TILE_WIDTH - 1) / TILE_WIDTH; ph++) {
        if (Row < N && (ph * TILE_WIDTH + tx) < N) {
            ds_A[ty][tx] = A[Row*N + (ph*TILE_WIDTH + tx)];
        } else {
            ds_A[ty][tx] = 0.0f;
        }

        if (Col < N && (ph * TILE_WIDTH + ty) < N) {
            ds_B[ty][tx] = B[(ph*TILE_WIDTH + ty)*N + Col];
        } else {
            ds_B[ty][tx] = 0.0f;
        }

        for (unsigned int k = 0; k < TILE_WIDTH; k++) {
            Cvalue += ds_A[ty][k] * ds_B[k][tx];
        }

    }

    if (Row < N && Col < N) {
        C[Row*N + Col] = Cvalue;
    }
}

void matmul_gpu_no_shared(const float* h_A,
                          const float* h_B,
                          float*       h_C,
                          unsigned int N)
{
    // Create CUDA events for timing
    cudaEvent_t startAlloc, stopAlloc;
    cudaEvent_t startCopyHtoD, stopCopyHtoD;
    cudaEvent_t startKernel, stopKernel;
    cudaEvent_t startCopyDtoH, stopCopyDtoH;

    cudaEventCreate(&startAlloc);   cudaEventCreate(&stopAlloc);
    cudaEventCreate(&startCopyHtoD);cudaEventCreate(&stopCopyHtoD);
    cudaEventCreate(&startKernel);  cudaEventCreate(&stopKernel);
    cudaEventCreate(&startCopyDtoH);cudaEventCreate(&stopCopyDtoH);

    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;

    // Allocating memory on the GPU
    cudaEventRecord(startAlloc);
    cudaMalloc((void**)&d_A, N*N*sizeof(float));
    cudaMalloc((void**)&d_B, N*N*sizeof(float));
    cudaMalloc((void**)&d_C, N*N*sizeof(float));
    cudaDeviceSynchronize();
    cudaEventRecord(stopAlloc);
    cudaEventSynchronize(stopAlloc);

    float msAlloc = 0.0f;
    cudaEventElapsedTime(&msAlloc, startAlloc, stopAlloc);

    // Copy data from CPU to GPU
    cudaEventRecord(startCopyHtoD);
    cudaMemcpy(d_A, h_A, N*N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N*N*sizeof(float), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    cudaEventRecord(stopCopyHtoD);
    cudaEventSynchronize(stopCopyHtoD);

    float msHtoD = 0.0f;
    cudaEventElapsedTime(&msHtoD, startCopyHtoD, stopCopyHtoD);

    // Perform the matrix multiplication
    dim3 blockSize(TILE_WIDTH, TILE_WIDTH);
    dim3 gridSize((N + TILE_WIDTH - 1)/TILE_WIDTH,
                  (N + TILE_WIDTH - 1)/TILE_WIDTH);

    cudaEventRecord(startKernel);
    matmul_tiled_no_shared_kernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();
    cudaEventRecord(stopKernel);
    cudaEventSynchronize(stopKernel);

    float msKernel = 0.0f;
    cudaEventElapsedTime(&msKernel, startKernel, stopKernel);

    // Copy the result back to CPU
    cudaEventRecord(startCopyDtoH);
    cudaMemcpy(h_C, d_C, N*N*sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    cudaEventRecord(stopCopyDtoH);
    cudaEventSynchronize(stopCopyDtoH);

    float msDtoH = 0.0f;
    cudaEventElapsedTime(&msDtoH, startCopyDtoH, stopCopyDtoH);

    float totalTime = msAlloc + msHtoD + msKernel + msDtoH;

    std::cout << "======== GPU Tiled (NO Shared) ========\n";
    std::cout << "Alloc time (ms):       " << msAlloc  << "\n";
    std::cout << "Host->Device (ms):     " << msHtoD   << "\n";
    std::cout << "Kernel time (ms):      " << msKernel << "\n";
    std::cout << "Device->Host (ms):     " << msDtoH   << "\n";
    std::cout << "Total GPU time (ms):   " << totalTime<< "\n\n";

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    cudaEventDestroy(startAlloc);   cudaEventDestroy(stopAlloc);
    cudaEventDestroy(startCopyHtoD);cudaEventDestroy(stopCopyHtoD);
    cudaEventDestroy(startKernel);  cudaEventDestroy(stopKernel);
    cudaEventDestroy(startCopyDtoH);cudaEventDestroy(stopCopyDtoH);
}

void matmul_gpu_shared(const float* h_A,
                       const float* h_B,
                       float*       h_C,
                       unsigned int N)
{
    // Create CUDA events
    cudaEvent_t startAlloc, stopAlloc;
    cudaEvent_t startCopyHtoD, stopCopyHtoD;
    cudaEvent_t startKernel, stopKernel;
    cudaEvent_t startCopyDtoH, stopCopyDtoH;

    cudaEventCreate(&startAlloc);    cudaEventCreate(&stopAlloc);
    cudaEventCreate(&startCopyHtoD); cudaEventCreate(&stopCopyHtoD);
    cudaEventCreate(&startKernel);   cudaEventCreate(&stopKernel);
    cudaEventCreate(&startCopyDtoH); cudaEventCreate(&stopCopyDtoH);

    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;

    // Allocating memory on the GPU
    cudaEventRecord(startAlloc);
    cudaMalloc((void**)&d_A, N*N*sizeof(float));
    cudaMalloc((void**)&d_B, N*N*sizeof(float));
    cudaMalloc((void**)&d_C, N*N*sizeof(float));
    cudaDeviceSynchronize();
    cudaEventRecord(stopAlloc);
    cudaEventSynchronize(stopAlloc);

    float msAlloc = 0.0f;
    cudaEventElapsedTime(&msAlloc, startAlloc, stopAlloc);

    // Copy data from CPU to GPU
    cudaEventRecord(startCopyHtoD);
    cudaMemcpy(d_A, h_A, N*N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N*N*sizeof(float), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    cudaEventRecord(stopCopyHtoD);
    cudaEventSynchronize(stopCopyHtoD);

    float msHtoD = 0.0f;
    cudaEventElapsedTime(&msHtoD, startCopyHtoD, stopCopyHtoD);

    // Perform the matrix multiplication
    dim3 blockSize(TILE_WIDTH, TILE_WIDTH);
    dim3 gridSize((N + TILE_WIDTH - 1)/TILE_WIDTH,
                  (N + TILE_WIDTH - 1)/TILE_WIDTH);

    cudaEventRecord(startKernel);
    matmul_tiled_shared_kernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();
    cudaEventRecord(stopKernel);
    cudaEventSynchronize(stopKernel);

    float msKernel = 0.0f;
    cudaEventElapsedTime(&msKernel, startKernel, stopKernel);

    // Copy the result back to CPU
    cudaEventRecord(startCopyDtoH);
    cudaMemcpy(h_C, d_C, N*N*sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    cudaEventRecord(stopCopyDtoH);
    cudaEventSynchronize(stopCopyDtoH);

    float msDtoH = 0.0f;
    cudaEventElapsedTime(&msDtoH, startCopyDtoH, stopCopyDtoH);

    float totalTime = msAlloc + msHtoD + msKernel + msDtoH;

    std::cout << "======== GPU Tiled (WITH Shared) ========\n";
    std::cout << "Alloc time (ms):       " << msAlloc  << "\n";
    std::cout << "Host->Device (ms):     " << msHtoD   << "\n";
    std::cout << "Kernel time (ms):      " << msKernel << "\n";
    std::cout << "Device->Host (ms):     " << msDtoH   << "\n";
    std::cout << "Total GPU time (ms):   " << totalTime<< "\n\n";

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    cudaEventDestroy(startAlloc);   cudaEventDestroy(stopAlloc);
    cudaEventDestroy(startCopyHtoD);cudaEventDestroy(stopCopyHtoD);
    cudaEventDestroy(startKernel);  cudaEventDestroy(stopKernel);
    cudaEventDestroy(startCopyDtoH);cudaEventDestroy(stopCopyDtoH);
}


int main()
{
    const unsigned int N = 1024;

    auto cpu_start_alloc = std::chrono::high_resolution_clock::now();
    float* h_A = (float*)malloc(N*N*sizeof(float));
    float* h_B = (float*)malloc(N*N*sizeof(float));
    float* h_C = (float*)malloc(N*N*sizeof(float));
    auto cpu_stop_alloc = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> dur_alloc = cpu_stop_alloc - cpu_start_alloc;
    double msAllocCpu = dur_alloc.count();

    auto cpu_start_copy = std::chrono::high_resolution_clock::now();
    for (unsigned int i = 0; i < N*N; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }
    auto cpu_stop_copy = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> dur_copy = cpu_stop_copy - cpu_start_copy;
    double msCopyCpu = dur_copy.count();

    auto cpu_start_kernel = std::chrono::high_resolution_clock::now();
    matmul_cpu_tiled(h_A, h_B, h_C, N);
    auto cpu_stop_kernel = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> dur_kernel = cpu_stop_kernel - cpu_start_kernel;
    double msKernelCpu = dur_kernel.count();

    double totalCpu = msAllocCpu + msCopyCpu + msKernelCpu;

    std::cout << "======== CPU Tiled (No Shared) ========\n";
    std::cout << "Alloc time (ms):    " << msAllocCpu  << "\n";
    std::cout << "Copy/init time (ms):" << msCopyCpu   << "\n";
    std::cout << "Kernel time (ms):   " << msKernelCpu << "\n";
    std::cout << "Total CPU time (ms):" << totalCpu    << "\n\n";

    // Matrix multiplication without shared memory on the GPU
    matmul_gpu_no_shared(h_A, h_B, h_C, N);

    // Matrix multiplication with shared memory on the GPU
    matmul_gpu_shared(h_A, h_B, h_C, N);

    // Cleanup
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
