// linear_regression_forward.cu
#include <iostream>
#include <cuda_runtime.h>
#include <chrono>
#include <cstdlib>
#include <ctime>

// forward pass on GPU
__global__ void linear_forward(float* X, float* w, float* b, float* y_pred, int num_samples, int num_features) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_samples) {
        float y = 0.0f;
        for (int j = 0; j < num_features; j++) {
            y += X[idx * num_features + j] * w[j];
        }
        y += *b;
        y_pred[idx] = y;
    }
}

// forward pass on CPU
void linear_forward_cpu(float* X, float* w, float b, float* y_pred, int num_samples, int num_features) {
    for (int i = 0; i < num_samples; i++) {
        float y = 0.0f;
        for (int j = 0; j < num_features; j++) {
            y += X[i * num_features + j] * w[j];
        }
        y += b;
        y_pred[i] = y;
    }
}

int main() {
    const int num_samples = 1000;     // scale up to 100000 or more
    const int num_features = 10;      // scale up to 1000 or more

    std::srand(std::time(0));

    // Dynamically allocate large arrays
    float* h_X = new float[num_samples * num_features];
    float* h_w = new float[num_features];
    float* h_y_pred = new float[num_samples];
    float* h_y_true = new float[num_samples];
    float* h_y_cpu = new float[num_samples];
    float h_b = static_cast<float>(rand()) / RAND_MAX;

    // Random initialization
    for (int i = 0; i < num_samples * num_features; ++i)
        h_X[i] = static_cast<float>(rand()) / RAND_MAX;

    for (int i = 0; i < num_features; ++i)
        h_w[i] = static_cast<float>(rand()) / RAND_MAX;

    // Generate true values
    linear_forward_cpu(h_X, h_w, h_b, h_y_true, num_samples, num_features);

    // GPU memory allocation
    float *d_X, *d_w, *d_b, *d_y_pred;
    cudaMalloc(&d_X, sizeof(float) * num_samples * num_features);
    cudaMalloc(&d_w, sizeof(float) * num_features);
    cudaMalloc(&d_b, sizeof(float));
    cudaMalloc(&d_y_pred, sizeof(float) * num_samples);

    // CUDA timing
    cudaEvent_t start, stop;
    float time_memcpy_h2d, time_kernel, time_memcpy_d2h, total_gpu_time;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Host to device
    cudaEventRecord(start);
    cudaMemcpy(d_X, h_X, sizeof(float) * num_samples * num_features, cudaMemcpyHostToDevice);
    cudaMemcpy(d_w, h_w, sizeof(float) * num_features, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, &h_b, sizeof(float), cudaMemcpyHostToDevice);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_memcpy_h2d, start, stop);

    // Launch kernel
    int blockSize = 1024;
    int gridSize = (num_samples + blockSize - 1) / blockSize;

    cudaEventRecord(start);
    linear_forward<<<gridSize, blockSize>>>(d_X, d_w, d_b, d_y_pred, num_samples, num_features);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_kernel, start, stop);

    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA kernel error: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    cudaDeviceSynchronize(); // Make sure kernel is finished

    // Copy back result
    cudaEventRecord(start);
    cudaMemcpy(h_y_pred, d_y_pred, sizeof(float) * num_samples, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_memcpy_d2h, start, stop);

    total_gpu_time = time_memcpy_h2d + time_kernel + time_memcpy_d2h;

    // CPU version
    auto cpu_start = std::chrono::high_resolution_clock::now();
    linear_forward_cpu(h_X, h_w, h_b, h_y_cpu, num_samples, num_features);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_duration = cpu_end - cpu_start;

    // Print first 10 results
    std::cout << "\nFirst 10 Predictions Comparison:\n";
    std::cout << "Index\tTrue y\tGPU y\tCPU y\n";
    for (int i = 0; i < 10; i++) {
        std::cout << i << "\t"
                  << h_y_true[i] << "\t"
                  << h_y_pred[i] << "\t"
                  << h_y_cpu[i] << "\n";
    }

    // Timings
    std::cout << "\nGPU Timing (ms):\n";
    std::cout << "Host to Device:       " << time_memcpy_h2d << " ms\n";
    std::cout << "Kernel execution:     " << time_kernel << " ms\n";
    std::cout << "Device to Host:       " << time_memcpy_d2h << " ms\n";
    std::cout << "Total GPU time:       " << total_gpu_time << " ms\n";

    std::cout << "\nTotal CPU time:       " << cpu_duration.count() << " ms\n";

    // Cleanup
    cudaFree(d_X);
    cudaFree(d_w);
    cudaFree(d_b);
    cudaFree(d_y_pred);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    delete[] h_X;
    delete[] h_w;
    delete[] h_y_pred;
    delete[] h_y_true;
    delete[] h_y_cpu;

    return 0;
}
