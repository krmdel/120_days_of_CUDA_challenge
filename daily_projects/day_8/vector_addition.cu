#include <chrono>
#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>

__host__ __device__ float f(float x, float y) {
    return x + y;
}

void vecadd_cpu(float* x, float* y, float* z, int N) {
    for (unsigned int i = 0; i < N; i++) {
        z[i] = f(x[i], y[i]);
    }
}

__global__ void vecadd_kernel(float* x, float* y, float* z, int N) {
    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N) {
        z[i] = f(x[i], y[i]);
    }
}

void vecadd_gpu(float* x, float* y, float* z, int N) {
    // Allocate memory on GPU
    float *x_d, *y_d, *z_d;
    cudaMalloc((void**)&x_d, N * sizeof(float));
    cudaMalloc((void**)&y_d, N * sizeof(float));
    cudaMalloc((void**)&z_d, N * sizeof(float));

    // Copy data to GPU
    cudaMemcpy(x_d, x, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(y_d, y, N * sizeof(float), cudaMemcpyHostToDevice);

    // Setup kernel launch parameters
    const unsigned int numThreadsPerBlock = 512;
    const unsigned int numBlocks = (N + numThreadsPerBlock - 1) / numThreadsPerBlock;

    // Create CUDA events for kernel timing
    cudaEvent_t kernel_start, kernel_stop;
    cudaEventCreate(&kernel_start);
    cudaEventCreate(&kernel_stop);

    // Record kernel start time, launch kernel, then record stop time
    cudaEventRecord(kernel_start);
    vecadd_kernel<<< numBlocks, numThreadsPerBlock >>>(x_d, y_d, z_d, N);
    cudaError_t err = cudaDeviceSynchronize(); // Wait for the kernel to finish to execute the timing
    if (err != cudaSuccess) {
        std::cerr << "CUDA error in file '" << __FILE__ << "' in line " << __LINE__ << ": " << cudaGetErrorString(err) << "\n";
        exit(EXIT_FAILURE);
    }
    cudaEventRecord(kernel_stop);

    // Wait for the kernel to finish
    cudaEventSynchronize(kernel_stop);

    // Calculate and print kernel elapsed time
    float kernel_time_ms = 0;
    cudaEventElapsedTime(&kernel_time_ms, kernel_start, kernel_stop);
    std::cout << "GPU kernel elapsed time: " << kernel_time_ms << " ms\n";

    // Copy result back to CPU
    cudaMemcpy(z, z_d, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Clean up GPU memory and CUDA events for kernel timing
    cudaFree(x_d);
    cudaFree(y_d);
    cudaFree(z_d);
    cudaEventDestroy(kernel_start);
    cudaEventDestroy(kernel_stop);
}

int main(int argc, char** argv) {
    cudaDeviceSynchronize();

    // Allocate memory on host
    unsigned int N = (argc > 1) ? atoi(argv[1]) : (1 << 25);
    float* x = (float*)malloc(N * sizeof(float));
    float* y = (float*)malloc(N * sizeof(float));
    float* z = (float*)malloc(N * sizeof(float));
    for (unsigned int i = 0; i < N; i++) {
        x[i] = rand();
        y[i] = rand();
    }

    // CPU timing using chrono
    auto start_cpu = std::chrono::high_resolution_clock::now();
    vecadd_cpu(x, y, z, N);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> cpu_duration = end_cpu - start_cpu;
    std::cout << "CPU vecadd elapsed time: " << cpu_duration.count() * 1000 << " ms\n";

    // Overall GPU timing (including memory transfers) using CUDA events
    cudaEvent_t overall_start, overall_stop;
    cudaEventCreate(&overall_start);
    cudaEventCreate(&overall_stop);

    cudaEventRecord(overall_start);
    vecadd_gpu(x, y, z, N);
    cudaEventRecord(overall_stop);

    cudaEventSynchronize(overall_stop);
    float overall_gpu_time_ms = 0;
    cudaEventElapsedTime(&overall_gpu_time_ms, overall_start, overall_stop);
    std::cout << "Overall GPU vecadd elapsed time: " << overall_gpu_time_ms << " ms\n";

    // Clean up host memory and overall timing events
    free(x);
    free(y);
    free(z);
    cudaEventDestroy(overall_start);
    cudaEventDestroy(overall_stop);

    return 0;
}
