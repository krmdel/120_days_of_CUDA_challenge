#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define BLOCK_SIZE 256

__global__ void reduceKernel_baseline(float *d_in, float *d_out, int n) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (BLOCK_SIZE * 2) + threadIdx.x;
    float sum = 0.0f;
    if (i < n)
        sum = d_in[i];
    if (i + BLOCK_SIZE < n)
        sum += d_in[i + BLOCK_SIZE];
    sdata[tid] = sum;
    __syncthreads();

    for (unsigned int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0)
        d_out[blockIdx.x] = sdata[0];
}

__global__ void reduceKernel_warpShuffle(float *d_in, float *d_out, int n) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (BLOCK_SIZE * 2) + threadIdx.x;
    float sum = 0.0f;
    if (i < n)
        sum = d_in[i];
    if (i + BLOCK_SIZE < n)
        sum += d_in[i + BLOCK_SIZE];
    sdata[tid] = sum;
    __syncthreads();

    for (unsigned int s = BLOCK_SIZE / 2; s > 32; s >>= 1) {
        if (tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid < 32) {
        float val = sdata[tid];
        for (int offset = 16; offset > 0; offset /= 2) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
        if (tid == 0)
            sdata[0] = val;
    }
    
    __syncthreads();
    if (tid == 0)
        d_out[blockIdx.x] = sdata[0];
}

int main(){
    int numElements = 1 << 20;
    size_t size = numElements * sizeof(float);

    float *h_in = (float*) malloc(size);
    for (int i = 0; i < numElements; i++){
        h_in[i] = 1.0f;  // Initialize all elements to 1.0f for simplicity.
    }

    float result_baseline = 0.0f;
    float result_warpShuffle = 0.0f;

    // Allocate device memory
    float *d_in = NULL, *d_out = NULL;
    cudaMalloc((void**)&d_in, size);
    int gridSize = (numElements + BLOCK_SIZE * 2 - 1) / (BLOCK_SIZE * 2);
    size_t outSize = gridSize * sizeof(float);
    cudaMalloc((void**)&d_out, outSize);

    // Create CUDA events for timing
    cudaEvent_t startHtoD, stopHtoD;
    cudaEvent_t startKernel, stopKernel;
    cudaEvent_t startDtoH, stopDtoH;
    cudaEventCreate(&startHtoD); cudaEventCreate(&stopHtoD);
    cudaEventCreate(&startKernel); cudaEventCreate(&stopKernel);
    cudaEventCreate(&startDtoH); cudaEventCreate(&stopDtoH);

    // Copy data from CPU to GPU
    cudaEventRecord(startHtoD, 0);
    cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);
    cudaEventRecord(stopHtoD, 0);
    cudaEventSynchronize(stopHtoD);
    float timeHtoD = 0.0f;
    cudaEventElapsedTime(&timeHtoD, startHtoD, stopHtoD);

    // Launch the baseline reduction kernel
    cudaEventRecord(startKernel, 0);
    reduceKernel_baseline<<<gridSize, BLOCK_SIZE, BLOCK_SIZE * sizeof(float)>>>(d_in, d_out, numElements);
    cudaEventRecord(stopKernel, 0);
    cudaEventSynchronize(stopKernel);
    float timeKernel = 0.0f;
    cudaEventElapsedTime(&timeKernel, startKernel, stopKernel);

    // Copy the partial results back to CPU
    float *h_partial = (float*) malloc(outSize);
    cudaEventRecord(startDtoH, 0);
    cudaMemcpy(h_partial, d_out, outSize, cudaMemcpyDeviceToHost);
    cudaEventRecord(stopDtoH, 0);
    cudaEventSynchronize(stopDtoH);
    float timeDtoH = 0.0f;
    cudaEventElapsedTime(&timeDtoH, startDtoH, stopDtoH);

    // Perform the reduction on CPU
    for (int i = 0; i < gridSize; i++){
        result_baseline += h_partial[i];
    }

    printf("Timing for Baseline Reduction\n");
    printf("Time Host-to-Device: %f ms\n", timeHtoD);
    printf("Kernel Time: %f ms\n", timeKernel);
    printf("Time Device-to-Host: %f ms\n", timeDtoH);
    
    free(h_partial);

    cudaEventRecord(startHtoD, 0);
    cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);
    cudaEventRecord(stopHtoD, 0);
    cudaEventSynchronize(stopHtoD);
    float timeHtoD_warp = 0.0f;
    cudaEventElapsedTime(&timeHtoD_warp, startHtoD, stopHtoD);

    // Launch the warp shuffle reduction kernel
    cudaEventRecord(startKernel, 0);
    reduceKernel_warpShuffle<<<gridSize, BLOCK_SIZE, BLOCK_SIZE * sizeof(float)>>>(d_in, d_out, numElements);
    cudaEventRecord(stopKernel, 0);
    cudaEventSynchronize(stopKernel);
    float timeKernel_warp = 0.0f;
    cudaEventElapsedTime(&timeKernel_warp, startKernel, stopKernel);

    // Copy the partial results back to CPU
    float *h_partial_warp = (float*) malloc(outSize);
    cudaEventRecord(startDtoH, 0);
    cudaMemcpy(h_partial_warp, d_out, outSize, cudaMemcpyDeviceToHost);
    cudaEventRecord(stopDtoH, 0);
    cudaEventSynchronize(stopDtoH);
    float timeDtoH_warp = 0.0f;
    cudaEventElapsedTime(&timeDtoH_warp, startDtoH, stopDtoH);

    // Perform the reduction on CPU
    for (int i = 0; i < gridSize; i++){
        result_warpShuffle += h_partial_warp[i];
    }
   
    printf("Timing for Warp Shuffle Reduction\n");
    printf("Time Host-to-Device: %f ms\n", timeHtoD_warp);
    printf("Kernel Time: %f ms\n", timeKernel_warp);
    printf("Time Device-to-Host: %f ms\n", timeDtoH_warp);
    
    free(h_partial_warp);

    // Cleanup
    cudaFree(d_in);
    cudaFree(d_out);
    cudaEventDestroy(startHtoD);
    cudaEventDestroy(stopHtoD);
    cudaEventDestroy(startKernel);
    cudaEventDestroy(stopKernel);
    cudaEventDestroy(startDtoH);
    cudaEventDestroy(stopDtoH);
    free(h_in);

    return 0;
}
