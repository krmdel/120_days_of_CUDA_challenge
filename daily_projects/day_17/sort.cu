// sort.cu
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <algorithm>

#define ARRAY_SIZE 1024
#define BLOCK_SIZE 256

// Macro for checking CUDA errors
#define CHECK_CUDA(call) {                                   \
    cudaError_t err = call;                                  \
    if(err != cudaSuccess){                                  \
        fprintf(stderr, "CUDA error in %s at line %d: %s\n",  \
                __FILE__, __LINE__, cudaGetErrorString(err));\
        exit(err);                                           \
    }                                                        \
}

__global__ void radixSortBaselineKernel(int* d_input, int* d_output, int digitWidth, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        int val = d_input[idx];
        int divisor = (digitWidth == 1) ? 10 : 100;
        int digit = val % divisor;
        d_output[idx] = digit;
    }
}

__global__ void radixSortOptimizedKernel(int* d_input, int* d_output, int digitWidth, int size) {
    __shared__ int s_data[BLOCK_SIZE];
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        s_data[threadIdx.x] = d_input[idx];
    }
    __syncthreads();

    int val = s_data[threadIdx.x];
    int divisor = (digitWidth == 1) ? 10 : 100;
    int digit = val % divisor;
    
    if (idx < size) {
        d_output[idx] = digit;
    }
}

__global__ void mergeSortBaselineKernel(int* d_input, int* d_output, int width, int size) {
    int blockId = blockIdx.x;
    int start = blockId * (width * 2);
    if (start >= size) return;
    
    int mid = min(start + width, size);
    int end = min(start + 2 * width, size);
    
    int i = start;
    int j = mid;
    int k = start;
    while (i < mid && j < end) {
        if (d_input[i] < d_input[j]) {
            d_output[k++] = d_input[i++];
        } else {
            d_output[k++] = d_input[j++];
        }
    }
    while (i < mid) {
        d_output[k++] = d_input[i++];
    }
    while (j < end) {
        d_output[k++] = d_input[j++];
    }
}

__global__ void mergeSortOptimizedKernel(int* d_input, int* d_output, int width, int size) {
    int blockId = blockIdx.x;
    int start = blockId * (width * 2);
    if (start >= size) return;
    
    int mid = min(start + width, size);
    int end = min(start + 2 * width, size);
    
    int i = start;
    int j = mid;
    int k = start;
    while (i < mid && j < end) {
        if (d_input[i] < d_input[j]) {
            d_output[k++] = d_input[i++];
        } else {
            d_output[k++] = d_input[j++];
        }
    }
    while (i < mid) {
        d_output[k++] = d_input[i++];
    }
    while (j < end) {
        d_output[k++] = d_input[j++];
    }
}

void runRadixSort(int* h_input, int* h_output, int size, int digitWidth, bool optimized) {
    int *d_input, *d_output;
    size_t bytes = size * sizeof(int);
    CHECK_CUDA(cudaMalloc(&d_input, bytes));
    CHECK_CUDA(cudaMalloc(&d_output, bytes));
    
    // Copy  data from CPU to GPU
    CHECK_CUDA(cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice));
    
    // Create events for timing
    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);
    
    int grid = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    // Launch kernel
    cudaEventRecord(startEvent, 0);
    if (!optimized) {
        radixSortBaselineKernel<<<grid, BLOCK_SIZE>>>(d_input, d_output, digitWidth, size);
    } else {
        radixSortOptimizedKernel<<<grid, BLOCK_SIZE>>>(d_input, d_output, digitWidth, size);
    }
    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);
    
    float kernelTime;
    cudaEventElapsedTime(&kernelTime, startEvent, stopEvent);
    printf("Radix Sort (digit width: %d, optimized: %d) kernel time: %f ms\n", digitWidth, optimized, kernelTime);
    
    // Copy result back to CPU
    CHECK_CUDA(cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost));
    
    // Free device memory and events
    cudaFree(d_input);
    cudaFree(d_output);
    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);
}

void runMergeSort(int* h_input, int* h_output, int size, bool optimized) {
    int *d_input, *d_output;
    size_t bytes = size * sizeof(int);
    CHECK_CUDA(cudaMalloc(&d_input, bytes));
    CHECK_CUDA(cudaMalloc(&d_output, bytes));
    
    // Copy data from CPU to GPU
    CHECK_CUDA(cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice));
    
    // Create CUDA events for timing
    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);
    
    // Launch kernel
    cudaEventRecord(startEvent, 0);
    for (int width = 1; width < size; width *= 2) {
        int numBlocks = (size + (width * 2 - 1)) / (width * 2);
        if (!optimized) {
            mergeSortBaselineKernel<<<numBlocks, 1>>>(d_input, d_output, width, size);
        } else {
            mergeSortOptimizedKernel<<<numBlocks, 1>>>(d_input, d_output, width, size);
        }
        int* temp = d_input;
        d_input = d_output;
        d_output = temp;
    }
    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);
    
    float kernelTime;
    cudaEventElapsedTime(&kernelTime, startEvent, stopEvent);
    printf("Merge Sort (optimized: %d) total kernel time: %f ms\n", optimized, kernelTime);
    
    // Copy data back to CPU
    CHECK_CUDA(cudaMemcpy(h_output, d_input, bytes, cudaMemcpyDeviceToHost));
    
    // Free device memory and events
    cudaFree(d_input);
    cudaFree(d_output);
    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);
}

int main() {
    int size = ARRAY_SIZE;
    int *h_input = (int*) malloc(size * sizeof(int));
    int *h_output = (int*) malloc(size * sizeof(int));
    
    // Initialize random data for sorting
    for (int i = 0; i < size; i++) {
        h_input[i] = rand() % 1000;
    }
    
    printf("Running Radix Sort Baseline and Optimized:\n");
    printf("Comparing least significant digit (width=1) vs two digits (width=2):\n");
    for (int digitWidth = 1; digitWidth <= 2; digitWidth++) {
        runRadixSort(h_input, h_output, size, digitWidth, false);
        runRadixSort(h_input, h_output, size, digitWidth, true);
    }
    
    // Reinitialize data (unsorted array) for merge sort
    for (int i = 0; i < size; i++) {
        h_input[i] = rand() % 1000;
    }
    
    printf("Running Merge Sort Baseline and Optimized:\n");
    runMergeSort(h_input, h_output, size, false);
    runMergeSort(h_input, h_output, size, true);
    
    free(h_input);
    free(h_output);
    
    return 0;
}
