#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include <cstdlib>
#include <cstdio>

#define NUM_BINS 256
#define THREADS_PER_BLOCK 256

__global__ void histogramPrivatization(const unsigned char* data, int size, unsigned int* hist) {
    __shared__ unsigned int s_hist[NUM_BINS];

    int tid = threadIdx.x;
    for (int i = tid; i < NUM_BINS; i += blockDim.x) {
        s_hist[i] = 0;
    }
    __syncthreads();

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    while (index < size) {
        atomicAdd(&s_hist[data[index]], 1);
        index += stride;
    }
    __syncthreads();

    for (int i = tid; i < NUM_BINS; i += blockDim.x) {
        atomicAdd(&hist[i], s_hist[i]);
    }
}

__global__ void histogramThreadCoarsening(const unsigned char* data, int size, unsigned int* hist) {
    __shared__ unsigned int s_hist[NUM_BINS];

    int tid = threadIdx.x;
    for (int i = tid; i < NUM_BINS; i += blockDim.x) {
        s_hist[i] = 0;
    }
    __syncthreads();

    int globalId = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    int stride = blockDim.x * gridDim.x * 2;
    while (globalId < size) {
        atomicAdd(&s_hist[data[globalId]], 1);
        if (globalId + blockDim.x * gridDim.x < size) {
            atomicAdd(&s_hist[data[globalId + blockDim.x * gridDim.x]], 1);
        }
        globalId += stride;
    }
    __syncthreads();

    for (int i = tid; i < NUM_BINS; i += blockDim.x) {
        atomicAdd(&hist[i], s_hist[i]);
    }
}

void cpuHistogram(const unsigned char* data, int size, unsigned int* hist) {
    for (int i = 0; i < NUM_BINS; i++) {
        hist[i] = 0;
    }
    for (int i = 0; i < size; i++) {
        hist[data[i]]++;
    }
}

int main() {
    int size = 1 << 20;
    size_t dataSize = size * sizeof(unsigned char);

    unsigned char* h_data = (unsigned char*)malloc(dataSize);
    for (int i = 0; i < size; i++) {
        h_data[i] = rand() % NUM_BINS;
    }

    // Compute histogram on CPU and measure time.
    unsigned int h_hist_cpu[NUM_BINS];
    auto cpu_start = std::chrono::high_resolution_clock::now();
    cpuHistogram(h_data, size, h_hist_cpu);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    double cpu_time = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();

    // Allocate device memory and copy data form CPU to GPU
    unsigned char* d_data;
    unsigned int* d_hist;
    cudaMalloc((void**)&d_data, dataSize);
    cudaMalloc((void**)&d_hist, NUM_BINS * sizeof(unsigned int));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Privatization with shared memory
    cudaMemset(d_hist, 0, NUM_BINS * sizeof(unsigned int));

    cudaEventRecord(start);
    cudaMemcpy(d_data, h_data, dataSize, cudaMemcpyHostToDevice);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float copyH2D_time;
    cudaEventElapsedTime(&copyH2D_time, start, stop);

    int blocks = (size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    // Launch the privatization kernel
    cudaEventRecord(start);
    histogramPrivatization<<<blocks, THREADS_PER_BLOCK>>>(d_data, size, d_hist);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float kernel_time;
    cudaEventElapsedTime(&kernel_time, start, stop);

    unsigned int h_hist_gpu[NUM_BINS];
    cudaEventRecord(start);
    cudaMemcpy(h_hist_gpu, d_hist, NUM_BINS * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float copyD2H_time;
    cudaEventElapsedTime(&copyD2H_time, start, stop);

    float total_time = copyH2D_time + kernel_time + copyD2H_time;

    std::cout << "=== Privatization Kernel Timing (ms) ===\n";
    std::cout << "Copy H2D: " << copyH2D_time << " ms\n";
    std::cout << "Kernel Execution: " << kernel_time << " ms\n";
    std::cout << "Copy D2H: " << copyD2H_time << " ms\n";
    std::cout << "Total GPU Time: " << total_time << " ms\n";
    std::cout << "CPU Histogram Time: " << cpu_time << " ms\n";

    // Thread Coarsening Kernel
    cudaMemset(d_hist, 0, NUM_BINS * sizeof(unsigned int));

    // Copy data from CPU to GPU
    cudaEventRecord(start);
    cudaMemcpy(d_data, h_data, dataSize, cudaMemcpyHostToDevice);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float copyH2D_time2;
    cudaEventElapsedTime(&copyH2D_time2, start, stop);

    int blocks_coarsened = (blocks < 2) ? 1 : blocks / 2;

    // Launch thread coarsening kernel
    cudaEventRecord(start);
    histogramThreadCoarsening<<<blocks_coarsened, THREADS_PER_BLOCK>>>(d_data, size, d_hist);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float kernel_time2;
    cudaEventElapsedTime(&kernel_time2, start, stop);

    // Copy from GPU to CPU
    cudaEventRecord(start);
    cudaMemcpy(h_hist_gpu, d_hist, NUM_BINS * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float copyD2H_time2;
    cudaEventElapsedTime(&copyD2H_time2, start, stop);

    float total_time2 = copyH2D_time2 + kernel_time2 + copyD2H_time2;

    std::cout << "\n=== Thread Coarsening Kernel Timing (ms) ===\n";
    std::cout << "Copy H2D: " << copyH2D_time2 << " ms\n";
    std::cout << "Kernel Execution: " << kernel_time2 << " ms\n";
    std::cout << "Copy D2H: " << copyD2H_time2 << " ms\n";
    std::cout << "Total GPU Time: " << total_time2 << " ms\n";

    // Cleanup
    cudaFree(d_data);
    cudaFree(d_hist);
    free(h_data);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
