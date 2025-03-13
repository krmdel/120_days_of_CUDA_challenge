#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <chrono>

#define BLOCK_SIZE 256

using Clock = std::chrono::high_resolution_clock;

__global__ void sumReduction(float *input, float *output, int n) {
    __shared__ float sharedMem[BLOCK_SIZE];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    sharedMem[tid] = (idx < n) ? input[idx] : 0.0f;
    __syncthreads();

    for(int s = blockDim.x / 2; s > 0; s >>= 1) {
        if(tid < s) sharedMem[tid] += sharedMem[tid + s];
        __syncthreads();
    }

    if(tid == 0) output[blockIdx.x] = sharedMem[0];
}

void cpuReduction(float *data, int n, float *sum) {
    *sum = 0.0f;
    for(int i = 0; i < n; ++i) {
        *sum += data[i];
    }
}

int main() {
    int n = 1 << 24;
    size_t bytes = n * sizeof(float);

    float *h_input = (float*)malloc(bytes);
    float cpu_sum;

    for(int i = 0; i < n; i++)
        h_input[i] = 1.001f;

    // CPU timing
    auto cpuStart = Clock::now();
    cpuReduction(h_input, n, &cpu_sum);
    auto cpuEnd = Clock::now();

    // Allocate memory on GPU
    float *d_input;
    cudaMalloc(&d_input, bytes);

    // Copy data from CPU to GPU
    auto gpuStart = Clock::now();
    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);
    auto copyToGpuEnd = Clock::now();

    int gridSize = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    float *d_temp_input, *d_temp_output;
    cudaMalloc(&d_temp_input, gridSize * sizeof(float));
    cudaMalloc(&d_temp_output, gridSize * sizeof(float));

    // GPU Sum Reduction timing
    auto kernelStart = Clock::now();
    sumReduction<<<gridSize, BLOCK_SIZE>>>(d_input, d_temp_input, n);
    cudaDeviceSynchronize();

    int s = gridSize;
    while (s > 1) {
        int threads = (s > BLOCK_SIZE) ? BLOCK_SIZE : s;
        int blocks = (s + threads - 1) / threads;
        sumReduction<<<blocks, threads>>>(d_temp_input, d_temp_output, s);
        cudaDeviceSynchronize();

        float *tmp = d_temp_input;
        d_temp_input = d_temp_output;
        d_temp_output = tmp;

        s = blocks;
    }
    auto kernelEnd = Clock::now();

    float *partial_sums = (float*)malloc(s * sizeof(float));

    auto copyFromGpuStart = Clock::now();
    cudaMemcpy(partial_sums, d_temp_input, s * sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    auto copyFromGpuEnd = Clock::now();

    float gpu_sum = 0.0f;
    for(int i = 0; i < s; i++) gpu_sum += partial_sums[i];

    printf("CPU Sum: %.4f\n", cpu_sum);
    printf("GPU Sum: %.4f\n", gpu_sum);

    printf("\nTiming Comparison:\n");
    printf("CPU time: %.3f ms\n", std::chrono::duration<double, std::milli>(cpuEnd - cpuStart).count());
    printf("GPU copy to device: %.3f ms\n", std::chrono::duration<double>(copyToGpuEnd - gpuStart).count() * 1000);
    printf("GPU kernel time: %.3f ms\n", std::chrono::duration<double, std::milli>(kernelEnd - copyToGpuEnd).count());
    printf("GPU copy from device: %.3f ms\n", std::chrono::duration<double, std::milli>(Clock::now() - copyFromGpuStart).count());
    printf("GPU total time: %.3f ms\n", std::chrono::duration<double, std::milli>(copyFromGpuStart - gpuStart + Clock::now() - copyFromGpuStart).count());

    cudaFree(d_input);
    cudaFree(d_temp_input);
    cudaFree(d_temp_output);
    free(partial_sums);
    free(h_input);

    return 0;
}
