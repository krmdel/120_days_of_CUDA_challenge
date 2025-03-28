// linear_regression_full_cpu_gpu.cu
#include <iostream>
#include <cuda_runtime.h>
#include <chrono>
#include <cstdlib>
#include <ctime>

// === GPU kernels ===
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

__global__ void compute_gradients(float* X, float* y_true, float* y_pred, float* grad_w, float* grad_b, int num_samples, int num_features) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_samples) {
        float error = y_pred[idx] - y_true[idx];
        for (int j = 0; j < num_features; j++) {
            atomicAdd(&grad_w[j], error * X[idx * num_features + j] / num_samples);
        }
        atomicAdd(grad_b, error / num_samples);
    }
}

__global__ void update_weights(float* w, float* grad_w, float* b, float* grad_b, float lr, int num_features) {
    int idx = threadIdx.x;
    if (idx < num_features) {
        w[idx] -= lr * grad_w[idx];
    }
    if (idx == 0) {
        *b -= lr * (*grad_b);
    }
}

// === CPU functions ===
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

void compute_gradients_cpu(float* X, float* y_true, float* y_pred, float* grad_w, float& grad_b, int num_samples, int num_features) {
    for (int j = 0; j < num_features; j++) grad_w[j] = 0.0f;
    grad_b = 0.0f;
    for (int i = 0; i < num_samples; i++) {
        float error = y_pred[i] - y_true[i];
        for (int j = 0; j < num_features; j++) {
            grad_w[j] += error * X[i * num_features + j] / num_samples;
        }
        grad_b += error / num_samples;
    }
}

void update_weights_cpu(float* w, float* grad_w, float& b, float grad_b, float lr, int num_features) {
    for (int j = 0; j < num_features; j++) {
        w[j] -= lr * grad_w[j];
    }
    b -= lr * grad_b;
}

int main() {
    const int num_samples = 10000;
    const int num_features = 1000;
    const float lr = 0.01;

    std::srand(std::time(0));

    // Allocate host memory
    float* h_X = new float[num_samples * num_features];
    float* h_w = new float[num_features];
    float* h_w_cpu = new float[num_features];
    float* h_y_true = new float[num_samples];
    float* h_y_pred_gpu = new float[num_samples];
    float* h_y_pred_cpu = new float[num_samples];
    float* h_grad_w_cpu = new float[num_features];
    float h_grad_b_cpu = 0.0f;

    float h_b = static_cast<float>(rand()) / RAND_MAX;
    float h_b_cpu = h_b;

    for (int i = 0; i < num_samples * num_features; ++i)
        h_X[i] = static_cast<float>(rand()) / RAND_MAX;
    for (int i = 0; i < num_features; ++i)
        h_w[i] = static_cast<float>(rand()) / RAND_MAX;

    std::copy(h_w, h_w + num_features, h_w_cpu);
    linear_forward_cpu(h_X, h_w, h_b, h_y_true, num_samples, num_features);

    // GPU memory
    float *d_X, *d_w, *d_b, *d_y_true, *d_y_pred, *d_grad_w, *d_grad_b;
    cudaMalloc(&d_X, sizeof(float) * num_samples * num_features);
    cudaMalloc(&d_w, sizeof(float) * num_features);
    cudaMalloc(&d_b, sizeof(float));
    cudaMalloc(&d_y_true, sizeof(float) * num_samples);
    cudaMalloc(&d_y_pred, sizeof(float) * num_samples);
    cudaMalloc(&d_grad_w, sizeof(float) * num_features);
    cudaMalloc(&d_grad_b, sizeof(float));

    int blockSize = 1024;
    int gridSize = (num_samples + blockSize - 1) / blockSize;

    // CUDA Events
    cudaEvent_t start, stop;
    float time_h2d, time_kernel, time_d2h, total_gpu_time;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Host to Device
    cudaEventRecord(start);
    cudaMemcpy(d_X, h_X, sizeof(float) * num_samples * num_features, cudaMemcpyHostToDevice);
    cudaMemcpy(d_w, h_w, sizeof(float) * num_features, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, &h_b, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y_true, h_y_true, sizeof(float) * num_samples, cudaMemcpyHostToDevice);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_h2d, start, stop);

    // Kernel: forward + backward + update
    cudaEventRecord(start);
    linear_forward<<<gridSize, blockSize>>>(d_X, d_w, d_b, d_y_pred, num_samples, num_features);
    cudaDeviceSynchronize();

    cudaMemset(d_grad_w, 0, sizeof(float) * num_features);
    cudaMemset(d_grad_b, 0, sizeof(float));
    compute_gradients<<<gridSize, blockSize>>>(d_X, d_y_true, d_y_pred, d_grad_w, d_grad_b, num_samples, num_features);
    cudaDeviceSynchronize();

    update_weights<<<1, num_features>>>(d_w, d_grad_w, d_b, d_grad_b, lr, num_features);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_kernel, start, stop);

    // Device to Host
    cudaEventRecord(start);
    cudaMemcpy(h_y_pred_gpu, d_y_pred, sizeof(float) * num_samples, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_w, d_w, sizeof(float) * num_features, cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_b, d_b, sizeof(float), cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_d2h, start, stop);

    total_gpu_time = time_h2d + time_kernel + time_d2h;

    // CPU full pass
    auto cpu_start = std::chrono::high_resolution_clock::now();
    linear_forward_cpu(h_X, h_w_cpu, h_b_cpu, h_y_pred_cpu, num_samples, num_features);
    compute_gradients_cpu(h_X, h_y_true, h_y_pred_cpu, h_grad_w_cpu, h_grad_b_cpu, num_samples, num_features);
    update_weights_cpu(h_w_cpu, h_grad_w_cpu, h_b_cpu, h_grad_b_cpu, lr, num_features);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_time = cpu_end - cpu_start;

    // === Output ===
    std::cout << "\nFirst 10 Predictions (after 1 update):\n";
    std::cout << "Idx\tTrue y\tGPU y\tCPU y\n";
    for (int i = 0; i < 10; i++) {
        std::cout << i << "\t" << h_y_true[i] << "\t" << h_y_pred_gpu[i] << "\t" << h_y_pred_cpu[i] << "\n";
    }

    std::cout << "\nGPU Timing (ms):\n";
    std::cout << "Copy Host to Device: " << time_h2d << " ms\n";
    std::cout << "Kernel Execution:    " << time_kernel << " ms\n";
    std::cout << "Copy Device to Host: " << time_d2h << " ms\n";
    std::cout << "Total GPU time:      " << total_gpu_time << " ms\n";

    std::cout << "\nCPU Timing:\n";
    std::cout << "Total CPU time:      " << cpu_time.count() << " ms\n";

    // Cleanup
    cudaFree(d_X); cudaFree(d_w); cudaFree(d_b); cudaFree(d_y_true);
    cudaFree(d_y_pred); cudaFree(d_grad_w); cudaFree(d_grad_b);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    delete[] h_X; delete[] h_w; delete[] h_w_cpu; delete[] h_y_true;
    delete[] h_y_pred_gpu; delete[] h_y_pred_cpu; delete[] h_grad_w_cpu;

    return 0;
}
