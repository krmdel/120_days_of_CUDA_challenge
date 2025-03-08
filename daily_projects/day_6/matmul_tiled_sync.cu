#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define TILE_WIDTH 32

__global__ void matmul_tiled_kernel_sync(const float* A, const float* B, float* C, unsigned int N)
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

    unsigned int numPhases = (N + TILE_WIDTH - 1) / TILE_WIDTH;
    for (unsigned int ph = 0; ph < numPhases; ph++)
    {
        unsigned int tiledACol = ph * TILE_WIDTH + tx;
        if (Row < N && tiledACol < N) {
            ds_A[ty][tx] = A[Row*N + tiledACol];
        } else {
            ds_A[ty][tx] = 0.0f;
        }

        unsigned int tiledBRow = ph * TILE_WIDTH + ty;
        if (Col < N && tiledBRow < N) {
            ds_B[ty][tx] = B[tiledBRow*N + Col];
        } else {
            ds_B[ty][tx] = 0.0f;
        }

        __syncthreads();  // Sync threads in block

        for (int k = 0; k < TILE_WIDTH; k++) {
            Cvalue += ds_A[ty][k] * ds_B[k][tx];
        }

        __syncthreads(); // Sync threads in block
    }

    // Store result
    if (Row < N && Col < N) {
        C[Row*N + Col] = Cvalue;
    }
}

inline void checkCuda(cudaError_t result, const char* msg)
{
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s %s\n", msg, cudaGetErrorString(result));
        exit(EXIT_FAILURE);
    }
}

void matmul_gpu_tiled_base(const float* h_A, const float* h_B, float* h_C, unsigned int N)
{
    // Create timing events
    cudaEvent_t startAlloc, stopAlloc;
    cudaEvent_t startCopyH2D, stopCopyH2D;
    cudaEvent_t startKernel, stopKernel;
    cudaEvent_t startCopyD2H, stopCopyD2H;

    checkCuda(cudaEventCreate(&startAlloc),   "EventCreate startAlloc");
    checkCuda(cudaEventCreate(&stopAlloc),    "EventCreate stopAlloc");
    checkCuda(cudaEventCreate(&startCopyH2D), "EventCreate startCopyH2D");
    checkCuda(cudaEventCreate(&stopCopyH2D),  "EventCreate stopCopyH2D");
    checkCuda(cudaEventCreate(&startKernel),  "EventCreate startKernel");
    checkCuda(cudaEventCreate(&stopKernel),   "EventCreate stopKernel");
    checkCuda(cudaEventCreate(&startCopyD2H), "EventCreate startCopyD2H");
    checkCuda(cudaEventCreate(&stopCopyD2H),  "EventCreate stopCopyD2H");

    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;

    // Allocate memory on GPU
    cudaEventRecord(startAlloc);
    checkCuda(cudaMalloc((void**)&d_A, N*N*sizeof(float)), "malloc d_A");
    checkCuda(cudaMalloc((void**)&d_B, N*N*sizeof(float)), "malloc d_B");
    checkCuda(cudaMalloc((void**)&d_C, N*N*sizeof(float)), "malloc d_C");
    cudaDeviceSynchronize();
    cudaEventRecord(stopAlloc);
    cudaEventSynchronize(stopAlloc);

    float msAlloc;
    cudaEventElapsedTime(&msAlloc, startAlloc, stopAlloc);

    // Copy data from CPU to GPU
    cudaEventRecord(startCopyH2D);
    checkCuda(cudaMemcpy(d_A, h_A, N*N*sizeof(float), cudaMemcpyHostToDevice), "memcpy A->d_A");
    checkCuda(cudaMemcpy(d_B, h_B, N*N*sizeof(float), cudaMemcpyHostToDevice), "memcpy B->d_B");
    cudaDeviceSynchronize();
    cudaEventRecord(stopCopyH2D);
    cudaEventSynchronize(stopCopyH2D);

    float msH2D;
    cudaEventElapsedTime(&msH2D, startCopyH2D, stopCopyH2D);

    // Perform the matrix multiplication
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH);
    dim3 gridDim((N + TILE_WIDTH - 1)/TILE_WIDTH, (N + TILE_WIDTH - 1)/TILE_WIDTH);

    // *** FIX: Record kernel events properly ***
    cudaEventRecord(startKernel);
    matmul_tiled_kernel_sync<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();
    cudaEventRecord(stopKernel);
    cudaEventSynchronize(stopKernel);

    float msKernel;
    cudaEventElapsedTime(&msKernel, startKernel, stopKernel);

    // Copy the result from GPU to CPU
    cudaEventRecord(startCopyD2H);
    checkCuda(cudaMemcpy(h_C, d_C, N*N*sizeof(float), cudaMemcpyDeviceToHost), "memcpy d_C->C");
    cudaDeviceSynchronize();
    cudaEventRecord(stopCopyD2H);
    cudaEventSynchronize(stopCopyD2H);

    float msD2H;
    cudaEventElapsedTime(&msD2H, startCopyD2H, stopCopyD2H);

    float total = msAlloc + msH2D + msKernel + msD2H;
    printf("=== Baseline Tiled Kernel ===\n");
    printf("Alloc:           %f ms\n", msAlloc);
    printf("Host->Device:    %f ms\n", msH2D);
    printf("Kernel:          %f ms\n", msKernel);
    printf("Device->Host:    %f ms\n", msD2H);
    printf("Total:           %f ms\n\n", total);

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    cudaEventDestroy(startAlloc);
    cudaEventDestroy(stopAlloc);
    cudaEventDestroy(startCopyH2D);
    cudaEventDestroy(stopCopyH2D);
    cudaEventDestroy(startKernel);
    cudaEventDestroy(stopKernel);
    cudaEventDestroy(startCopyD2H);
    cudaEventDestroy(stopCopyD2H);
}

void matmul_gpu_tiled_extended(const float* h_A, const float* h_B, float* h_C, unsigned int N)
{
    // Create timing events
    cudaEvent_t startAlloc, stopAlloc;
    cudaEvent_t startCopyH2D, stopCopyH2D;
    cudaEvent_t startKernel, stopKernel;
    cudaEvent_t startCopyD2H, stopCopyD2H;

    checkCuda(cudaEventCreate(&startAlloc),   "EventCreate startAlloc");
    checkCuda(cudaEventCreate(&stopAlloc),    "EventCreate stopAlloc");
    checkCuda(cudaEventCreate(&startCopyH2D), "EventCreate startCopyH2D");
    checkCuda(cudaEventCreate(&stopCopyH2D),  "EventCreate stopCopyH2D");
    checkCuda(cudaEventCreate(&startKernel),  "EventCreate startKernel");
    checkCuda(cudaEventCreate(&stopKernel),   "EventCreate stopKernel");
    checkCuda(cudaEventCreate(&startCopyD2H), "EventCreate startCopyD2H");
    checkCuda(cudaEventCreate(&stopCopyD2H),  "EventCreate stopCopyD2H");

    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;

    // Allocate memory on GPU
    cudaEventRecord(startAlloc);
    checkCuda(cudaMalloc((void**)&d_A, N*N*sizeof(float)), "malloc d_A");
    checkCuda(cudaMalloc((void**)&d_B, N*N*sizeof(float)), "malloc d_B");
    checkCuda(cudaMalloc((void**)&d_C, N*N*sizeof(float)), "malloc d_C");
    cudaDeviceSynchronize();
    cudaEventRecord(stopAlloc);
    cudaEventSynchronize(stopAlloc);

    float msAlloc;
    cudaEventElapsedTime(&msAlloc, startAlloc, stopAlloc);

    // Copy data from CPU to GPU
    cudaEventRecord(startCopyH2D);
    checkCuda(cudaMemcpy(d_A, h_A, N*N*sizeof(float), cudaMemcpyHostToDevice), "memcpy A->d_A");
    checkCuda(cudaMemcpy(d_B, h_B, N*N*sizeof(float), cudaMemcpyHostToDevice), "memcpy B->d_B");
    cudaDeviceSynchronize();
    cudaEventRecord(stopCopyH2D);
    cudaEventSynchronize(stopCopyH2D);

    float msH2D;
    cudaEventElapsedTime(&msH2D, startCopyH2D, stopCopyH2D);

    // Perform the matrix multiplication
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH);
    dim3 gridDim((N + TILE_WIDTH - 1)/TILE_WIDTH, (N + TILE_WIDTH - 1)/TILE_WIDTH);

    // *** FIX: Record kernel events properly ***
    cudaEventRecord(startKernel);
    matmul_tiled_kernel_sync<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();
    cudaEventRecord(stopKernel);
    cudaEventSynchronize(stopKernel);

    float msKernel;
    cudaEventElapsedTime(&msKernel, startKernel, stopKernel);

    // Copy the result from GPU to CPU
    cudaEventRecord(startCopyD2H);
    checkCuda(cudaMemcpy(h_C, d_C, N*N*sizeof(float), cudaMemcpyDeviceToHost), "memcpy d_C->C");
    cudaDeviceSynchronize();
    cudaEventRecord(stopCopyD2H);
    cudaEventSynchronize(stopCopyD2H);

    float msD2H;
    cudaEventElapsedTime(&msD2H, startCopyD2H, stopCopyD2H);

    float total = msAlloc + msH2D + msKernel + msD2H;
    printf("=== Extended Tiled Kernel (Extra sync) ===\n");
    printf("Alloc:           %f ms\n", msAlloc);
    printf("Host->Device:    %f ms\n", msH2D);
    printf("Kernel:          %f ms\n", msKernel);
    printf("Device->Host:    %f ms\n", msD2H);
    printf("Total:           %f ms\n\n", total);

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    cudaEventDestroy(startAlloc);
    cudaEventDestroy(stopAlloc);
    cudaEventDestroy(startCopyH2D);
    cudaEventDestroy(stopCopyH2D);
    cudaEventDestroy(startKernel);
    cudaEventDestroy(stopKernel);
    cudaEventDestroy(startCopyD2H);
    cudaEventDestroy(stopCopyD2H);
}

int main(int argc, char** argv)
{
    unsigned int N = 1024;
    if (argc > 1) {
        N = atoi(argv[1]);
    }
    printf("Matrix dimension: %u x %u\n", N, N);

    // Allocate memory on CPU
    float* h_A = (float*)malloc(N*N*sizeof(float));
    float* h_B = (float*)malloc(N*N*sizeof(float));
    float* h_C = (float*)malloc(N*N*sizeof(float));

    for (unsigned int i = 0; i < N*N; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    // Matrix multiplication with tiled kernel
    matmul_gpu_tiled_base(h_A, h_B, h_C, N);

    // Matrix multiplication with extended tiled kernel
    matmul_gpu_tiled_extended(h_A, h_B, h_C, N);

    // Cleanup
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
