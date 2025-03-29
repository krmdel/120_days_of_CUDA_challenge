#include <iostream>
#include <cuda_runtime.h>
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <algorithm> 
#include <cstring> 

// GPU Kernels for Logistic Regression
__global__ void logistic_forward(float* X, float* w, float* b, float* y_pred, int num_samples, int num_features) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < num_samples) {
        float z = 0.0f;
        for (int j = 0; j < num_features; j++) {
            z += X[idx * num_features + j] * w[j];
        }
        z += *b;
        y_pred[idx] = 1.0f / (1.0f + expf(-z));
    }
}

// Kernel for gradient computations
__global__ void compute_gradients(float* X, float* y_true, float* y_pred, float* grad_w, float* grad_b, int num_samples, int num_features) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < num_samples) {
        float error = y_pred[idx] - y_true[idx];
        for(int j = 0; j < num_features; j++) {
            // Using atomicAdd because multiple threads update the same gradient values.
            atomicAdd(&grad_w[j], error * X[idx * num_features + j] / num_samples);
        }
        atomicAdd(grad_b, error / num_samples);
    }
}

// Kernel for updating model parameters using gradient descent.
__global__ void update_weights(float* w, float* grad_w, float* b, float* grad_b, float lr, int num_features) {
    int idx = threadIdx.x;
    if(idx < num_features) {
        w[idx] -= lr * grad_w[idx];
    }
    if(idx == 0) {
        *b -= lr * (*grad_b);
    }
}


// CPU Implementations for Logistic Regression

// Sigmoid function on CPU
float sigmoid(float z) {
    return 1.0f / (1.0f + std::exp(-z));
}

// CPU forward pass: computes sigmoid
void logistic_forward_cpu(float* X, float* w, float b, float* y_pred, int num_samples, int num_features) {
    for (int i = 0; i < num_samples; i++) {
        float z = 0.0f;
        for (int j = 0; j < num_features; j++) {
            z += X[i * num_features + j] * w[j];
        }
        z += b;
        y_pred[i] = sigmoid(z);
    }
}

// CPU gradient computation for logistic regression
void compute_gradients_cpu(float* X, float* y_true, float* y_pred, float* grad_w, float &grad_b, int num_samples, int num_features) {
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

// CPU update of weights
void update_weights_cpu(float* w, float* grad_w, float &b, float grad_b, float lr, int num_features) {
    for (int j = 0; j < num_features; j++) {
        w[j] -= lr * grad_w[j];
    }
    b -= lr * grad_b;
}

// Compute binary cross-entropy loss on CPU
float compute_loss_cpu(float* y_true, float* y_pred, int num_samples) {
    float loss = 0.0f;
    float eps = 1e-7f;
    for (int i = 0; i < num_samples; i++) {
        loss += -(y_true[i] * std::log(y_pred[i] + eps) + (1 - y_true[i]) * std::log(1 - y_pred[i] + eps));
    }
    return loss / num_samples;
}

int main() {

    // Synthetic Data Generation Parameters
    int num_samples = 10000;    // Total samples
    int num_features = 1000;     // Number of features
    int epochs = 10;            // Number of training epochs
    int batch_size = 100;       // Mini-batch size
    float lr = 0.01f;           // Learning rate

    std::srand(std::time(0));

    // Allocate host memory for features and labels
    float* h_X = new float[num_samples * num_features];
    float* h_y = new float[num_samples];  // binary labels

    // Create a "true" logistic model for generating synthetic data.
    float* true_w = new float[num_features];
    float true_b = static_cast<float>(rand()) / RAND_MAX;
    for (int j = 0; j < num_features; j++) {
        true_w[j] = static_cast<float>(rand()) / RAND_MAX;
    }
    // Generate synthetic features and binary labels using the true model.
    for (int i = 0; i < num_samples; i++) {
        float z = 0.0f;
        for (int j = 0; j < num_features; j++) {
            h_X[i * num_features + j] = static_cast<float>(rand()) / RAND_MAX;
            z += h_X[i * num_features + j] * true_w[j];
        }
        z += true_b;
        float prob = 1.0f / (1.0f + std::exp(-z));
        h_y[i] = (static_cast<float>(rand()) / RAND_MAX < prob) ? 1.0f : 0.0f;
    }


    // For GPU training
    float* h_w = new float[num_features];
    float h_b = 0.0f;  // initialize bias to zero
    // For CPU training
    float* h_w_cpu = new float[num_features];
    float h_b_cpu = 0.0f;
    for (int j = 0; j < num_features; j++) {
        h_w[j] = static_cast<float>(rand()) / RAND_MAX;
        h_w_cpu[j] = h_w[j]; // same initial weights
    }

    // CPU Training and Inference
    int cpu_num_batches = num_samples / batch_size;
    int* indices_cpu = new int[num_samples];
    for (int i = 0; i < num_samples; i++) indices_cpu[i] = i;

    std::cout << "\nStarting CPU Training...\n";
    double cpu_training_time_total = 0.0;
    for (int epoch = 0; epoch < epochs; epoch++) {
        std::random_shuffle(indices_cpu, indices_cpu + num_samples);
        float epoch_loss = 0.0f;
        auto epoch_start = std::chrono::high_resolution_clock::now();
        for (int b = 0; b < cpu_num_batches; b++) {
            float* batch_X = new float[batch_size * num_features];
            float* batch_y = new float[batch_size];
            float* batch_y_pred = new float[batch_size];
            float* batch_grad_w = new float[num_features];
            float batch_grad_b = 0.0f;
            for (int i = 0; i < batch_size; i++) {
                int idx = indices_cpu[b * batch_size + i];
                for (int j = 0; j < num_features; j++) {
                    batch_X[i * num_features + j] = h_X[idx * num_features + j];
                }
                batch_y[i] = h_y[idx];
            }
            // Forward pass (CPU)
            logistic_forward_cpu(batch_X, h_w_cpu, h_b_cpu, batch_y_pred, batch_size, num_features);
            float batch_loss = compute_loss_cpu(batch_y, batch_y_pred, batch_size);
            epoch_loss += batch_loss;
            // Compute gradients (CPU)
            compute_gradients_cpu(batch_X, batch_y, batch_y_pred, batch_grad_w, batch_grad_b, batch_size, num_features);
            // Update weights (CPU)
            update_weights_cpu(h_w_cpu, batch_grad_w, h_b_cpu, batch_grad_b, lr, num_features);

            delete[] batch_X;
            delete[] batch_y;
            delete[] batch_y_pred;
            delete[] batch_grad_w;
        }
        auto epoch_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> epoch_duration = epoch_end - epoch_start;
        cpu_training_time_total += epoch_duration.count();
        std::cout << "CPU Epoch " << epoch+1 << "/" << epochs << " Loss: " 
                  << (epoch_loss / cpu_num_batches) << " Epoch Time (CPU): " 
                  << epoch_duration.count() << " ms\n";
    }
    std::cout << "Total CPU Training Time: " << cpu_training_time_total << " ms\n";

    // CPU Inference
    float* h_y_pred_cpu_infer = new float[num_samples];
    auto cpu_infer_start = std::chrono::high_resolution_clock::now();
    logistic_forward_cpu(h_X, h_w_cpu, h_b_cpu, h_y_pred_cpu_infer, num_samples, num_features);
    auto cpu_infer_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_infer_time = cpu_infer_end - cpu_infer_start;
    std::cout << "\nCPU Inference Time: " << cpu_infer_time.count() << " ms\n";


    // GPU Training and Inference

    // Allocate GPU memory for mini-batch training
    float *d_X, *d_y, *d_w, *d_b, *d_y_pred, *d_grad_w, *d_grad_b;
    cudaMalloc(&d_X, sizeof(float) * batch_size * num_features);
    cudaMalloc(&d_y, sizeof(float) * batch_size);
    cudaMalloc(&d_w, sizeof(float) * num_features);
    cudaMalloc(&d_b, sizeof(float));
    cudaMalloc(&d_y_pred, sizeof(float) * batch_size);
    cudaMalloc(&d_grad_w, sizeof(float) * num_features);
    cudaMalloc(&d_grad_b, sizeof(float));

    // Copy initial weights and bias to GPU (for GPU training)
    cudaMemcpy(d_w, h_w, sizeof(float) * num_features, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, &h_b, sizeof(float), cudaMemcpyHostToDevice);

    float time_h2d_total = 0.0f;
    float time_kernel_total = 0.0f;
    float time_d2h_total = 0.0f;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    std::cout << "\nStarting GPU Training...\n";
    int num_batches = num_samples / batch_size;
    int* indices = new int[num_samples];
    for (int i = 0; i < num_samples; i++) indices[i] = i;

    for (int epoch = 0; epoch < epochs; epoch++) {
        std::random_shuffle(indices, indices + num_samples);
        float epoch_loss = 0.0f;
        auto epoch_start = std::chrono::high_resolution_clock::now();
        for (int b = 0; b < num_batches; b++) {
            // Prepare mini-batch from CPU memory
            float* batch_X = new float[batch_size * num_features];
            float* batch_y = new float[batch_size];
            for (int i = 0; i < batch_size; i++) {
                int idx = indices[b * batch_size + i];
                for (int j = 0; j < num_features; j++) {
                    batch_X[i * num_features + j] = h_X[idx * num_features + j];
                }
                batch_y[i] = h_y[idx];
            }

            // Copy mini-batch from CPU to GPU
            cudaEventRecord(start);
            cudaMemcpy(d_X, batch_X, sizeof(float) * batch_size * num_features, cudaMemcpyHostToDevice);
            cudaMemcpy(d_y, batch_y, sizeof(float) * batch_size, cudaMemcpyHostToDevice);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            float time_h2d;
            cudaEventElapsedTime(&time_h2d, start, stop);
            time_h2d_total += time_h2d;

            // Forward pass kernel launch
            int blockSize = 256;
            int gridSize = (batch_size + blockSize - 1) / blockSize;
            cudaEventRecord(start);
            logistic_forward<<<gridSize, blockSize>>>(d_X, d_w, d_b, d_y_pred, batch_size, num_features);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            float time_kernel;
            cudaEventElapsedTime(&time_kernel, start, stop);
            time_kernel_total += time_kernel;

            // Copy predictions back from GPU for loss computation on CPU
            cudaEventRecord(start);
            float* h_y_pred = new float[batch_size];
            cudaMemcpy(h_y_pred, d_y_pred, sizeof(float) * batch_size, cudaMemcpyDeviceToHost);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            float time_d2h;
            cudaEventElapsedTime(&time_d2h, start, stop);
            time_d2h_total += time_d2h;

            float batch_loss = compute_loss_cpu(batch_y, h_y_pred, batch_size);
            epoch_loss += batch_loss;

            // Zero out gradients on GPU memory
            cudaMemset(d_grad_w, 0, sizeof(float) * num_features);
            cudaMemset(d_grad_b, 0, sizeof(float));

            // Launch gradient computation kernel
            cudaEventRecord(start);
            compute_gradients<<<gridSize, blockSize>>>(d_X, d_y, d_y_pred, d_grad_w, d_grad_b, batch_size, num_features);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            float time_kernel_grad;
            cudaEventElapsedTime(&time_kernel_grad, start, stop);
            time_kernel_total += time_kernel_grad;

            // Update weights kernel launch
            cudaEventRecord(start);
            update_weights<<<1, num_features>>>(d_w, d_grad_w, d_b, d_grad_b, lr, num_features);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            float time_kernel_update;
            cudaEventElapsedTime(&time_kernel_update, start, stop);
            time_kernel_total += time_kernel_update;

            delete[] batch_X;
            delete[] batch_y;
            delete[] h_y_pred;
        }
        auto epoch_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> epoch_duration = epoch_end - epoch_start;
        std::cout << "GPU Epoch " << epoch+1 << "/" << epochs << " Loss: " 
                  << (epoch_loss / num_batches) << " Epoch Time (CPU): " 
                  << epoch_duration.count() << " ms\n";
    }

    // Copy the final model parameters from GPU to CPU.
    cudaMemcpy(h_w, d_w, sizeof(float) * num_features, cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_b, d_b, sizeof(float), cudaMemcpyDeviceToHost);

    // GPU Inference
    std::cout << "\nStarting GPU Inference...\n";
    int num_test_samples = num_samples;
    float* h_y_pred_infer = new float[num_test_samples];

    // Allocate GPU memory for inference
    float *d_X_infer, *d_y_pred_infer;
    cudaMalloc(&d_X_infer, sizeof(float) * num_test_samples * num_features);
    cudaMalloc(&d_y_pred_infer, sizeof(float) * num_test_samples);

    // Copy test data to GPU and time it
    cudaEventRecord(start);
    cudaMemcpy(d_X_infer, h_X, sizeof(float) * num_test_samples * num_features, cudaMemcpyHostToDevice);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float time_h2d_infer;
    cudaEventElapsedTime(&time_h2d_infer, start, stop);

    // Launch inference kernel
    int blockSize = 256;
    int gridSize = (num_test_samples + blockSize - 1) / blockSize;
    cudaEventRecord(start);
    logistic_forward<<<gridSize, blockSize>>>(d_X_infer, d_w, d_b, d_y_pred_infer, num_test_samples, num_features);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float time_kernel_infer;
    cudaEventElapsedTime(&time_kernel_infer, start, stop);

    // Copy inference results back to CPU
    cudaEventRecord(start);
    cudaMemcpy(h_y_pred_infer, d_y_pred_infer, sizeof(float) * num_test_samples, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float time_d2h_infer;
    cudaEventElapsedTime(&time_d2h_infer, start, stop);

    std::cout << "\nGPU Inference timings:\n";
    std::cout << "Host to Device: " << time_h2d_infer << " ms\n";
    std::cout << "Kernel Execution: " << time_kernel_infer << " ms\n";
    std::cout << "Device to Host: " << time_d2h_infer << " ms\n";

    // For comparison, perform CPU inference again on the GPU test data using CPU functions.
    auto cpu_infer_start2 = std::chrono::high_resolution_clock::now();
    logistic_forward_cpu(h_X, h_w, h_b, h_y_pred_infer, num_test_samples, num_features);
    auto cpu_infer_end2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_infer_time2 = cpu_infer_end2 - cpu_infer_start2;
    std::cout << "\nCPU Inference Time (after GPU training): " << cpu_infer_time2.count() << " ms\n";

    std::cout << "\nTotal GPU Training Timings:\n";
    std::cout << "Total Host to Device (training): " << time_h2d_total << " ms\n";
    std::cout << "Total Kernel Execution (training): " << time_kernel_total << " ms\n";
    std::cout << "Total Device to Host (training): " << time_d2h_total << " ms\n";


    // Cleanup
    cudaFree(d_X);
    cudaFree(d_y);
    cudaFree(d_w);
    cudaFree(d_b);
    cudaFree(d_y_pred);
    cudaFree(d_grad_w);
    cudaFree(d_grad_b);
    cudaFree(d_X_infer);
    cudaFree(d_y_pred_infer);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    delete[] h_X;
    delete[] h_y;
    delete[] true_w;
    delete[] h_w;
    delete[] h_w_cpu;
    delete[] h_y_pred_cpu_infer;
    delete[] indices_cpu;
    delete[] indices;
    delete[] h_y_pred_infer;

    return 0;
}
