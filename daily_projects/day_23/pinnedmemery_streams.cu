#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define cudaCheckError(call) {                                             \
    cudaError_t err = call;                                                \
    if( err != cudaSuccess ) {                                             \
        fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__,  \
                cudaGetErrorString(err));                                  \
        exit(EXIT_FAILURE);                                                \
    }                                                                      \
}

__global__ void vectorAdd(const float *A, const float *B, float *C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n)
        C[i] = A[i] + B[i];
}

int main(void) {
    const int totalElements = 1 << 20;
    const int bytes = totalElements * sizeof(float);
    const int nStreams = 4;
    const int chunkSize = totalElements / nStreams;
    const int chunkBytes = chunkSize * sizeof(float);
    
    // Allocate pinned host memory for input and output arrays (streaming)
    float *h_A, *h_B, *h_C;
    cudaCheckError(cudaMallocHost((void**)&h_A, bytes));
    cudaCheckError(cudaMallocHost((void**)&h_B, bytes));
    cudaCheckError(cudaMallocHost((void**)&h_C, bytes));
    
    // Initialize input data on the host (streaming)
    for (int i = 0; i < totalElements; i++) {
        h_A[i] = static_cast<float>(i);
        h_B[i] = static_cast<float>(i * 2);
    }
    
    // Allocate device memory for vectors (streaming)
    float *d_A, *d_B, *d_C;
    cudaCheckError(cudaMalloc((void**)&d_A, bytes));
    cudaCheckError(cudaMalloc((void**)&d_B, bytes));
    cudaCheckError(cudaMalloc((void**)&d_C, bytes));
    
    // Create CUDA streams and per-stream events for timing (streaming)
    cudaStream_t streams[nStreams];
    cudaEvent_t startEvents[nStreams], htodEvents[nStreams], kernelEvents[nStreams], dtohEvents[nStreams];
    for (int i = 0; i < nStreams; i++) {
        cudaCheckError(cudaStreamCreate(&streams[i]));
        cudaCheckError(cudaEventCreate(&startEvents[i]));
        cudaCheckError(cudaEventCreate(&htodEvents[i]));
        cudaCheckError(cudaEventCreate(&kernelEvents[i]));
        cudaCheckError(cudaEventCreate(&dtohEvents[i]));
    }
    
    cudaEvent_t overallStart, overallStop;
    cudaCheckError(cudaEventCreate(&overallStart));
    cudaCheckError(cudaEventCreate(&overallStop));
    
    cudaCheckError(cudaEventRecord(overallStart, 0));
    
    const int threadsPerBlock = 256;
    
    // Enqueue asynchronous operations in each stream (streaming)
    for (int i = 0; i < nStreams; i++) {
        int offset = i * chunkSize;
        cudaCheckError(cudaEventRecord(startEvents[i], streams[i]));
        
        cudaCheckError(cudaMemcpyAsync(d_A + offset, h_A + offset, chunkBytes, cudaMemcpyHostToDevice, streams[i]));
        cudaCheckError(cudaMemcpyAsync(d_B + offset, h_B + offset, chunkBytes, cudaMemcpyHostToDevice, streams[i]));
        cudaCheckError(cudaEventRecord(htodEvents[i], streams[i]));
        
        int blocks = (chunkSize + threadsPerBlock - 1) / threadsPerBlock;
        vectorAdd<<<blocks, threadsPerBlock, 0, streams[i]>>>(d_A + offset, d_B + offset, d_C + offset, chunkSize);
        cudaCheckError(cudaEventRecord(kernelEvents[i], streams[i]));
        
        cudaCheckError(cudaMemcpyAsync(h_C + offset, d_C + offset, chunkBytes, cudaMemcpyDeviceToHost, streams[i]));
        cudaCheckError(cudaEventRecord(dtohEvents[i], streams[i]));
    }
    
    cudaCheckError(cudaDeviceSynchronize());
    
    cudaCheckError(cudaEventRecord(overallStop, 0));
    cudaCheckError(cudaEventSynchronize(overallStop));
    
    float maxHtod = 0, maxKernel = 0, maxDtoh = 0, maxTotal = 0;
    for (int i = 0; i < nStreams; i++) {
        float t_htod = 0, t_kernel = 0, t_dtoh = 0, t_total = 0;
        cudaCheckError(cudaEventElapsedTime(&t_htod, startEvents[i], htodEvents[i]));
        cudaCheckError(cudaEventElapsedTime(&t_kernel, htodEvents[i], kernelEvents[i]));
        cudaCheckError(cudaEventElapsedTime(&t_dtoh, kernelEvents[i], dtohEvents[i]));
        cudaCheckError(cudaEventElapsedTime(&t_total, startEvents[i], dtohEvents[i]));
        
        if(t_htod > maxHtod) maxHtod = t_htod;
        if(t_kernel > maxKernel) maxKernel = t_kernel;
        if(t_dtoh > maxDtoh) maxDtoh = t_dtoh;
        if(t_total > maxTotal) maxTotal = t_total;
    }
    
    float overallTime = 0;
    cudaCheckError(cudaEventElapsedTime(&overallTime, overallStart, overallStop));
    
    printf("Streaming (pinned memory) timings:\n");
    printf("Max host to device copy time per stream: %f ms\n", maxHtod);
    printf("Max kernel execution time per stream: %f ms\n", maxKernel);
    printf("Max device to host copy time per stream: %f ms\n", maxDtoh);
    printf("Max total time per stream: %f ms\n", maxTotal);
    printf("Overall elapsed time: %f ms\n", overallTime);
       
    // Baseline
    float *h_A_sync = (float*)malloc(bytes);
    float *h_B_sync = (float*)malloc(bytes);
    float *h_C_sync = (float*)malloc(bytes);
    for (int i = 0; i < totalElements; i++) {
        h_A_sync[i] = static_cast<float>(i);
        h_B_sync[i] = static_cast<float>(i * 2);
    }
    
    float *d_A_sync, *d_B_sync, *d_C_sync;
    cudaCheckError(cudaMalloc((void**)&d_A_sync, bytes));
    cudaCheckError(cudaMalloc((void**)&d_B_sync, bytes));
    cudaCheckError(cudaMalloc((void**)&d_C_sync, bytes));
    
    cudaEvent_t baseStart, baseHtoD, baseKernel, baseDtoH, baseStop;
    cudaCheckError(cudaEventCreate(&baseStart));
    cudaCheckError(cudaEventCreate(&baseHtoD));
    cudaCheckError(cudaEventCreate(&baseKernel));
    cudaCheckError(cudaEventCreate(&baseDtoH));
    cudaCheckError(cudaEventCreate(&baseStop));
    
    cudaCheckError(cudaEventRecord(baseStart, 0));
    cudaCheckError(cudaMemcpy(d_A_sync, h_A_sync, bytes, cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(d_B_sync, h_B_sync, bytes, cudaMemcpyHostToDevice));
    cudaCheckError(cudaEventRecord(baseHtoD, 0));
    
    int blocks = (totalElements + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocks, threadsPerBlock>>>(d_A_sync, d_B_sync, d_C_sync, totalElements);
    cudaCheckError(cudaEventRecord(baseKernel, 0));
    
    cudaCheckError(cudaMemcpy(h_C_sync, d_C_sync, bytes, cudaMemcpyDeviceToHost));
    cudaCheckError(cudaEventRecord(baseDtoH, 0));
    
    cudaCheckError(cudaEventRecord(baseStop, 0));
    cudaCheckError(cudaEventSynchronize(baseStop));
    
    float baseHtoDTime = 0, baseKernelTime = 0, baseDtoHTime = 0, baseTotalTime = 0;
    cudaCheckError(cudaEventElapsedTime(&baseHtoDTime, baseStart, baseHtoD));
    cudaCheckError(cudaEventElapsedTime(&baseKernelTime, baseHtoD, baseKernel));
    cudaCheckError(cudaEventElapsedTime(&baseDtoHTime, baseKernel, baseDtoH));
    cudaCheckError(cudaEventElapsedTime(&baseTotalTime, baseStart, baseStop));
    
    printf("Baseline timings:\n");
    printf("Host to device copy time: %f ms\n", baseHtoDTime);
    printf("Kernel execution time: %f ms\n", baseKernelTime);
    printf("Device to host copy time: %f ms\n", baseDtoHTime);
    printf("Total time: %f ms\n", baseTotalTime);
    
    // Clean up streaming
    for (int i = 0; i < nStreams; i++) {
        cudaCheckError(cudaEventDestroy(startEvents[i]));
        cudaCheckError(cudaEventDestroy(htodEvents[i]));
        cudaCheckError(cudaEventDestroy(kernelEvents[i]));
        cudaCheckError(cudaEventDestroy(dtohEvents[i]));
        cudaCheckError(cudaStreamDestroy(streams[i]));
    }
    cudaCheckError(cudaEventDestroy(overallStart));
    cudaCheckError(cudaEventDestroy(overallStop));
    
    cudaCheckError(cudaFree(d_A));
    cudaCheckError(cudaFree(d_B));
    cudaCheckError(cudaFree(d_C));
    cudaCheckError(cudaFreeHost(h_A));
    cudaCheckError(cudaFreeHost(h_B));
    cudaCheckError(cudaFreeHost(h_C));
    
    // Clean up baseline
    cudaCheckError(cudaEventDestroy(baseStart));
    cudaCheckError(cudaEventDestroy(baseHtoD));
    cudaCheckError(cudaEventDestroy(baseKernel));
    cudaCheckError(cudaEventDestroy(baseDtoH));
    cudaCheckError(cudaEventDestroy(baseStop));
    
    cudaCheckError(cudaFree(d_A_sync));
    cudaCheckError(cudaFree(d_B_sync));
    cudaCheckError(cudaFree(d_C_sync));
    free(h_A_sync);
    free(h_B_sync);
    free(h_C_sync);
    
    return 0;
}
