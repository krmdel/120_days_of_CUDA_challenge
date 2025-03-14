#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <chrono>

#define BLOCK_SIZE 256

using Clock = std::chrono::high_resolution_clock;

__global__ void optimizedReduction(float *input, float *output, int n) {
    __shared__ float sharedMem[BLOCK_SIZE];

    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    float sum = 0;

    if(idx < n) sum = input[idx];
    if(idx + blockDim.x < n) sum += input[idx + blockDim.x];

    sharedMem[tid] = sum;
    __syncthreads();

    for(unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
        if(tid < s) sharedMem[tid] += sharedMem[tid + s];
        __syncthreads();
    }

    if (tid < 32) {
        volatile float *vsmem = sharedMem;
        vsmem[tid] += vsmem[tid + 32];
        vsmem[tid] += vsmem[tid + 16];
        vsmem[tid] += vsmem[tid + 8];
        vsmem[tid] += vsmem[tid + 4];
        vsmem[tid] += vsmem[tid + 2];
        vsmem[tid] += vsmem[tid + 1];
    }

    if(tid == 0) output[blockIdx.x] = sharedMem[0];
}

int main() {
    int n = 1 << 24;
    size_t bytes = n * sizeof(float);

    float *h_input = (float*)malloc(bytes);
    float cpu_sum = 0.0f;

    for(int i = 0; i < n; i++) h_input[i] = 1.001f;

    auto cpuStart = Clock::now();
    for(int i = 0; i < n; ++i) cpu_sum += h_input[i];
    auto cpuEnd = Clock::now();

    float *d_input, *d_output;
    cudaMalloc(&d_input, bytes);
    auto copyToGPUStart = Clock::now();
    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);
    auto copyToGPUEnd = Clock::now();

    int gridSize = (n + BLOCK_SIZE * 2 - 1) / (BLOCK_SIZE * 2);
    cudaMalloc(&d_output, gridSize * sizeof(float));

    float gpu_sum;

    auto kernelStart = Clock::now();
    optimizedReduction<<<gridSize, BLOCK_SIZE>>>(d_input, d_output, n);
    cudaDeviceSynchronize();

    int s = gridSize;
    while(s > 1) {
        int threads = (s > BLOCK_SIZE * 2) ? BLOCK_SIZE : s / 2;
        int blocks = (s + threads * 2 - 1) / (threads * 2);
        optimizedReduction<<<blocks, threads>>>(d_output, d_output, s);
        cudaDeviceSynchronize();
        s = blocks;
    }
    auto kernelEnd = Clock::now();

    auto copyFromGPUStart = Clock::now();
    cudaMemcpy(&gpu_sum, d_output, sizeof(float), cudaMemcpyDeviceToHost);
    auto copyFromGPUEnd = Clock::now();

    printf("CPU Sum: %.4f\n", cpu_sum);
    printf("GPU Sum: %.4f\n", gpu_sum);

    printf("\nTiming Comparison:\n");
    printf("CPU Time: %.3f ms\n", std::chrono::duration<double, std::milli>(cpuEnd - cpuStart).count());
    printf("GPU copy to GPU Time: %.3f ms\n", std::chrono::duration<double, std::milli>(copyToGPUEnd - copyToGPUStart).count());
    printf("GPU Kernel Time: %.3f ms\n", std::chrono::duration<double, std::milli>(kernelEnd - kernelStart).count());
    printf("GPU copy from GPU Time: %.3f ms\n", std::chrono::duration<double, std::milli>(copyFromGPUEnd - copyFromGPUStart).count());
    printf("GPU Total Time: %.3f ms\n", std::chrono::duration<double, std::milli>(copyToGPUEnd - copyToGPUStart + kernelEnd - kernelStart + copyFromGPUEnd - copyFromGPUStart).count());

    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);

    return 0;
}
