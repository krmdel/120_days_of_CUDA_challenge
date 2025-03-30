#include <iostream>
#include <cuda_runtime.h>
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <algorithm>
#include <cstring>

// MLP parameters
const int hidden_size = 5;

// GPU Kernels for MLP

// Forward pass kernel: compute hidden layer and output
__global__ void mlp_forward(float* X, float* W1, float* b1, float* W2, float* b2,
                            float* a1, float* y_pred,
                            int num_samples, int num_features, int hidden_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_samples) {
        // Compute hidden layer activations
        for (int k = 0; k < hidden_size; k++) {
            float z = 0.0f;
            for (int j = 0; j < num_features; j++) {
                z += X[idx * num_features + j] * W1[j * hidden_size + k];
            }
            z += b1[k];
            a1[idx * hidden_size + k] = 1.0f / (1.0f + expf(-z)); // sigmoid
        }
        // Compute output layer
        float z2 = 0.0f;
        for (int k = 0; k < hidden_size; k++) {
            z2 += a1[idx * hidden_size + k] * W2[k];
        }
        z2 += *b2;
        y_pred[idx] = 1.0f / (1.0f + expf(-z2)); // sigmoid
    }
}

// Backward pass kernel: compute gradients for both layers using atomicAdd
__global__ void mlp_backward(float* X, float* y_true, float* a1, float* y_pred,
                             float* grad_W1, float* grad_b1, float* grad_W2, float* grad_b2,
                             float* W2,
                             int num_samples, int num_features, int hidden_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_samples) {
        float error2 = y_pred[idx] - y_true[idx]; // output error
        for (int k = 0; k < hidden_size; k++) {
            atomicAdd(&grad_W2[k], (error2 * a1[idx * hidden_size + k]) / num_samples);
        }
        atomicAdd(grad_b2, error2 / num_samples);
        for (int k = 0; k < hidden_size; k++) {
            float a = a1[idx * hidden_size + k];
            float error1 = error2 * W2[k] * a * (1.0f - a);
            for (int j = 0; j < num_features; j++) {
                atomicAdd(&grad_W1[j * hidden_size + k], (error1 * X[idx * num_features + j]) / num_samples);
            }
            atomicAdd(&grad_b1[k], error1 / num_samples);
        }
    }
}

// Kernel to update model parameters using gradient descent.
__global__ void mlp_update(float* W1, float* grad_W1, float* b1, float* grad_b1,
                           float* W2, float* grad_W2, float* b2, float* grad_b2,
                           float lr, int num_features, int hidden_size) {
    int idx = threadIdx.x;
    int total_W1 = num_features * hidden_size;
    // Update W1 weights
    if (idx < total_W1) {
        W1[idx] -= lr * grad_W1[idx];
    }
    // Update b1 biases
    if (idx < hidden_size) {
        b1[idx] -= lr * grad_b1[idx];
    }
    // Update W2 weights
    if (idx < hidden_size) {
        W2[idx] -= lr * grad_W2[idx];
    }
    // Update b2 (single bias)
    if (idx == 0) {
        *b2 -= lr * (*grad_b2);
    }
}

// CPU functions for MLP

// Sigmoid function
float sigmoid(float z) {
    return 1.0f / (1.0f + std::exp(-z));
}

// CPU forward pass for MLP
void mlp_forward_cpu(float* X, float* W1, float* b1, float* W2, float b2,
                     float* a1, float* y_pred,
                     int num_samples, int num_features, int hidden_size) {
    for (int i = 0; i < num_samples; i++) {
        // Hidden layer
        for (int k = 0; k < hidden_size; k++) {
            float z = 0.0f;
            for (int j = 0; j < num_features; j++) {
                z += X[i * num_features + j] * W1[j * hidden_size + k];
            }
            z += b1[k];
            a1[i * hidden_size + k] = sigmoid(z);
        }
        // Output layer
        float z2 = 0.0f;
        for (int k = 0; k < hidden_size; k++) {
            z2 += a1[i * hidden_size + k] * W2[k];
        }
        z2 += b2;
        y_pred[i] = sigmoid(z2);
    }
}

// Binary cross-entropy loss
float compute_loss_cpu(float* y_true, float* y_pred, int num_samples) {
    float loss = 0.0f;
    float eps = 1e-7f;
    for (int i = 0; i < num_samples; i++) {
        loss += -(y_true[i] * std::log(y_pred[i] + eps) + (1 - y_true[i]) * std::log(1 - y_pred[i] + eps));
    }
    return loss / num_samples;
}

// CPU backward pass: compute gradients for both layers
void mlp_backward_cpu(float* X, float* y_true, float* a1, float* y_pred,
                      float* grad_W1, float* grad_b1, float* grad_W2, float &grad_b2,
                      float* W2,
                      int num_samples, int num_features, int hidden_size) {
    // Zero gradients
    int total_W1 = num_features * hidden_size;
    for (int i = 0; i < total_W1; i++) grad_W1[i] = 0.0f;
    for (int k = 0; k < hidden_size; k++) grad_b1[k] = 0.0f;
    for (int k = 0; k < hidden_size; k++) grad_W2[k] = 0.0f;
    grad_b2 = 0.0f;

    for (int i = 0; i < num_samples; i++) {
        float error2 = y_pred[i] - y_true[i];
        // Gradients for output layer
        for (int k = 0; k < hidden_size; k++) {
            grad_W2[k] += (error2 * a1[i * hidden_size + k]) / num_samples;
        }
        grad_b2 += error2 / num_samples;
        // Backprop to hidden layer
        for (int k = 0; k < hidden_size; k++) {
            float a = a1[i * hidden_size + k];
            float error1 = error2 * W2[k] * a * (1.0f - a);
            for (int j = 0; j < num_features; j++) {
                grad_W1[j * hidden_size + k] += (error1 * X[i * num_features + j]) / num_samples;
            }
            grad_b1[k] += error1 / num_samples;
        }
    }
}

// CPU update of weights
void update_weights_cpu(float* W1, float* grad_W1, float* b1, float* grad_b1,
                        float* W2, float* grad_W2, float &b2, float grad_b2,
                        float lr, int num_features, int hidden_size) {
    int total_W1 = num_features * hidden_size;
    for (int i = 0; i < total_W1; i++) {
        W1[i] -= lr * grad_W1[i];
    }
    for (int k = 0; k < hidden_size; k++) {
        b1[k] -= lr * grad_b1[k];
    }
    for (int k = 0; k < hidden_size; k++) {
        W2[k] -= lr * grad_W2[k];
    }
    b2 -= lr * grad_b2;
}

int main() {
    // Synthetic data parameters
    int num_samples = 1000;
    int num_features = 10;
    int epochs = 10;
    int batch_size = 100;
    float lr = 0.01f;

    std::srand(std::time(0));

    // Allocate host memory for features and binary labels
    float* h_X = new float[num_samples * num_features];
    float* h_y = new float[num_samples];

    // Create a "true" logistic model to generate synthetic data
    float* true_w = new float[num_features];
    float true_b = static_cast<float>(rand()) / RAND_MAX;
    for (int j = 0; j < num_features; j++) {
        true_w[j] = static_cast<float>(rand()) / RAND_MAX;
    }
    // Generate synthetic data
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

    ////////// CPU Training and Inference //////////

    // CPU Training for MLP
    float* h_W1_cpu = new float[num_features * hidden_size];
    float* h_b1_cpu = new float[hidden_size];
    float* h_W2_cpu = new float[hidden_size];
    float h_b2_cpu = 0.0f;
    for (int i = 0; i < num_features * hidden_size; i++)
        h_W1_cpu[i] = static_cast<float>(rand()) / RAND_MAX;
    for (int k = 0; k < hidden_size; k++)
        h_b1_cpu[k] = 0.0f;
    for (int k = 0; k < hidden_size; k++)
        h_W2_cpu[k] = static_cast<float>(rand()) / RAND_MAX;

    int cpu_num_batches = num_samples / batch_size;
    int* indices_cpu = new int[num_samples];
    for (int i = 0; i < num_samples; i++) indices_cpu[i] = i;

    std::cout << "\nStarting CPU Training for MLP...\n";
    double cpu_training_time_total = 0.0;
    for (int epoch = 0; epoch < epochs; epoch++) {
        std::random_shuffle(indices_cpu, indices_cpu + num_samples);
        float epoch_loss = 0.0f;
        auto epoch_start = std::chrono::high_resolution_clock::now();
        for (int b = 0; b < cpu_num_batches; b++) {
            // Allocate mini-batch arrays
            float* batch_X = new float[batch_size * num_features];
            float* batch_y = new float[batch_size];
            float* batch_a1 = new float[batch_size * hidden_size];
            float* batch_y_pred = new float[batch_size];
            // Gradients
            float* batch_grad_W1 = new float[num_features * hidden_size];
            float* batch_grad_b1 = new float[hidden_size];
            float* batch_grad_W2 = new float[hidden_size];
            float batch_grad_b2 = 0.0f;

            // Prepare mini-batch
            for (int i = 0; i < batch_size; i++) {
                int idx = indices_cpu[b * batch_size + i];
                for (int j = 0; j < num_features; j++) {
                    batch_X[i * num_features + j] = h_X[idx * num_features + j];
                }
                batch_y[i] = h_y[idx];
            }

            // Forward pass (CPU)
            mlp_forward_cpu(batch_X, h_W1_cpu, h_b1_cpu, h_W2_cpu, h_b2_cpu,
                            batch_a1, batch_y_pred,
                            batch_size, num_features, hidden_size);
            float batch_loss = compute_loss_cpu(batch_y, batch_y_pred, batch_size);
            epoch_loss += batch_loss;
            // Backward pass (CPU)
            mlp_backward_cpu(batch_X, batch_y, batch_a1, batch_y_pred,
                             batch_grad_W1, batch_grad_b1, batch_grad_W2, batch_grad_b2,
                             h_W2_cpu,
                             batch_size, num_features, hidden_size);
            // Update weights (CPU)
            update_weights_cpu(h_W1_cpu, batch_grad_W1, h_b1_cpu, batch_grad_b1,
                               h_W2_cpu, batch_grad_W2, h_b2_cpu, batch_grad_b2,
                               lr, num_features, hidden_size);

            delete[] batch_X;
            delete[] batch_y;
            delete[] batch_a1;
            delete[] batch_y_pred;
            delete[] batch_grad_W1;
            delete[] batch_grad_b1;
            delete[] batch_grad_W2;
        }
        auto epoch_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> epoch_duration = epoch_end - epoch_start;
        cpu_training_time_total += epoch_duration.count();
        std::cout << "CPU Epoch " << epoch+1 << "/" << epochs << " Loss: "
                  << (epoch_loss / cpu_num_batches) << " Epoch Time (CPU): "
                  << epoch_duration.count() << " ms\n";
    }
    std::cout << "Total CPU Training Time: " << cpu_training_time_total << " ms\n";

    // CPU Inference after training
    float* h_a1_cpu_infer = new float[num_samples * hidden_size];
    float* h_y_pred_cpu_infer = new float[num_samples];
    auto cpu_infer_start = std::chrono::high_resolution_clock::now();
    mlp_forward_cpu(h_X, h_W1_cpu, h_b1_cpu, h_W2_cpu, h_b2_cpu,
                    h_a1_cpu_infer, h_y_pred_cpu_infer,
                    num_samples, num_features, hidden_size);
    auto cpu_infer_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_infer_time = cpu_infer_end - cpu_infer_start;
    std::cout << "\nCPU Inference Time: " << cpu_infer_time.count() << " ms\n";

    ////////// GPU Training and Inference //////////
    // For GPU training, we increase the mini-batch size to improve kernel occupancy.
    int batch_size_gpu = 1000;
    int num_batches_gpu = num_samples / batch_size_gpu;

    // GPU Training for MLP
    float* h_W1 = new float[num_features * hidden_size];
    float* h_b1 = new float[hidden_size];
    float* h_W2 = new float[hidden_size];
    float h_b2 = 0.0f;
    // Copy initial weights from CPU to GPU
    std::memcpy(h_W1, h_W1_cpu, sizeof(float) * num_features * hidden_size);
    std::memcpy(h_b1, h_b1_cpu, sizeof(float) * hidden_size);
    std::memcpy(h_W2, h_W2_cpu, sizeof(float) * hidden_size);

    // Allocate GPU memory for training using batch_size_gpu
    float *d_X, *d_y, *d_W1, *d_b1, *d_W2, *d_b2;
    float *d_a1, *d_y_pred;
    float *d_grad_W1, *d_grad_b1, *d_grad_W2, *d_grad_b2;
    cudaMalloc(&d_X, sizeof(float) * batch_size_gpu * num_features);
    cudaMalloc(&d_y, sizeof(float) * batch_size_gpu);
    cudaMalloc(&d_W1, sizeof(float) * num_features * hidden_size);
    cudaMalloc(&d_b1, sizeof(float) * hidden_size);
    cudaMalloc(&d_W2, sizeof(float) * hidden_size);
    cudaMalloc(&d_b2, sizeof(float));
    cudaMalloc(&d_a1, sizeof(float) * batch_size_gpu * hidden_size);
    cudaMalloc(&d_y_pred, sizeof(float) * batch_size_gpu);
    cudaMalloc(&d_grad_W1, sizeof(float) * num_features * hidden_size);
    cudaMalloc(&d_grad_b1, sizeof(float) * hidden_size);
    cudaMalloc(&d_grad_W2, sizeof(float) * hidden_size);
    cudaMalloc(&d_grad_b2, sizeof(float));

    // Copy initial weights to GPU
    cudaMemcpy(d_W1, h_W1, sizeof(float) * num_features * hidden_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b1, h_b1, sizeof(float) * hidden_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_W2, h_W2, sizeof(float) * hidden_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b2, &h_b2, sizeof(float), cudaMemcpyHostToDevice);

    // CUDA event timing variables
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float time_h2d_total = 0.0f;
    float time_kernel_total = 0.0f;
    float time_d2h_total = 0.0f;

    int* indices = new int[num_samples];
    for (int i = 0; i < num_samples; i++) indices[i] = i;

    std::cout << "\nStarting GPU Training for MLP...\n";
    for (int epoch = 0; epoch < epochs; epoch++) {
        std::random_shuffle(indices, indices + num_samples);
        float epoch_loss = 0.0f;
        auto epoch_start = std::chrono::high_resolution_clock::now();
        for (int b = 0; b < num_batches_gpu; b++) {
            // Prepare mini-batch on host (using batch_size_gpu)
            float* batch_X = new float[batch_size_gpu * num_features];
            float* batch_y = new float[batch_size_gpu];
            for (int i = 0; i < batch_size_gpu; i++) {
                int idx = indices[b * batch_size_gpu + i];
                for (int j = 0; j < num_features; j++) {
                    batch_X[i * num_features + j] = h_X[idx * num_features + j];
                }
                batch_y[i] = h_y[idx];
            }
            // Copy mini-batch to GPU
            cudaEventRecord(start);
            cudaMemcpy(d_X, batch_X, sizeof(float) * batch_size_gpu * num_features, cudaMemcpyHostToDevice);
            cudaMemcpy(d_y, batch_y, sizeof(float) * batch_size_gpu, cudaMemcpyHostToDevice);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            float time_h2d;
            cudaEventElapsedTime(&time_h2d, start, stop);
            time_h2d_total += time_h2d;

            // Launch forward pass kernel
            int blockSize = 256;
            int gridSize = (batch_size_gpu + blockSize - 1) / blockSize;
            cudaEventRecord(start);
            mlp_forward<<<gridSize, blockSize>>>(d_X, d_W1, d_b1, d_W2, d_b2,
                                                   d_a1, d_y_pred,
                                                   batch_size_gpu, num_features, hidden_size);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            float time_kernel;
            cudaEventElapsedTime(&time_kernel, start, stop);
            time_kernel_total += time_kernel;

            // Copy predictions back to host for loss computation
            float* h_y_pred = new float[batch_size_gpu];
            cudaEventRecord(start);
            cudaMemcpy(h_y_pred, d_y_pred, sizeof(float) * batch_size_gpu, cudaMemcpyDeviceToHost);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            float time_d2h;
            cudaEventElapsedTime(&time_d2h, start, stop);
            time_d2h_total += time_d2h;

            float batch_loss = compute_loss_cpu(batch_y, h_y_pred, batch_size_gpu);
            epoch_loss += batch_loss;

            // Zero out gradients on GPU
            cudaMemset(d_grad_W1, 0, sizeof(float) * num_features * hidden_size);
            cudaMemset(d_grad_b1, 0, sizeof(float) * hidden_size);
            cudaMemset(d_grad_W2, 0, sizeof(float) * hidden_size);
            cudaMemset(d_grad_b2, 0, sizeof(float));

            // Launch backward pass kernel
            cudaEventRecord(start);
            mlp_backward<<<gridSize, blockSize>>>(d_X, d_y, d_a1, d_y_pred,
                                                    d_grad_W1, d_grad_b1, d_grad_W2, d_grad_b2,
                                                    d_W2,
                                                    batch_size_gpu, num_features, hidden_size);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            float time_kernel_grad;
            cudaEventElapsedTime(&time_kernel_grad, start, stop);
            time_kernel_total += time_kernel_grad;

            // Launch update kernel
            int total_W1 = num_features * hidden_size;
            int update_block = (total_W1 > hidden_size) ? total_W1 : hidden_size;
            cudaEventRecord(start);
            mlp_update<<<1, update_block>>>(d_W1, d_grad_W1, d_b1, d_grad_b1,
                                            d_W2, d_grad_W2, d_b2, d_grad_b2,
                                            lr, num_features, hidden_size);
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
                  << (epoch_loss / num_batches_gpu) << " Epoch Time (CPU): "
                  << epoch_duration.count() << " ms\n";
    }

    // Print total GPU Training timings separately (Total time = H2D + Kernel + D2H)
    float total_gpu_training_time = time_h2d_total + time_kernel_total + time_d2h_total;
    std::cout << "\nTotal GPU Training Timings:\n";
    std::cout << "Total Host to Device (training): " << time_h2d_total << " ms\n";
    std::cout << "Total Kernel Execution (training): " << time_kernel_total << " ms\n";
    std::cout << "Total Device to Host (training): " << time_d2h_total << " ms\n";
    std::cout << "Total GPU Training Time: " << total_gpu_training_time << " ms\n";

    // Copy final GPU model parameters to host
    cudaMemcpy(h_W1, d_W1, sizeof(float) * num_features * hidden_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_b1, d_b1, sizeof(float) * hidden_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_W2, d_W2, sizeof(float) * hidden_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_b2, d_b2, sizeof(float), cudaMemcpyDeviceToHost);

    // GPU Inference
    std::cout << "\nStarting GPU Inference for MLP...\n";
    int num_test_samples = num_samples;
    float* h_a1_infer = new float[num_test_samples * hidden_size];
    float* h_y_pred_infer_gpu = new float[num_test_samples];

    // Allocate GPU memory for inference (including memory for hidden layer outputs)
    float *d_X_infer, *d_a1_infer, *d_y_pred_infer;
    cudaMalloc(&d_X_infer, sizeof(float) * num_test_samples * num_features);
    cudaMalloc(&d_a1_infer, sizeof(float) * num_test_samples * hidden_size);
    cudaMalloc(&d_y_pred_infer, sizeof(float) * num_test_samples);

    // Copy test data to GPU
    cudaEventRecord(start);
    cudaMemcpy(d_X_infer, h_X, sizeof(float) * num_test_samples * num_features, cudaMemcpyHostToDevice);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float time_h2d_infer;
    cudaEventElapsedTime(&time_h2d_infer, start, stop);

    // Launch inference kernel (reuse mlp_forward)
    int blockSize_infer = 256;
    int gridSize_infer = (num_test_samples + blockSize_infer - 1) / blockSize_infer;
    cudaEventRecord(start);
    mlp_forward<<<gridSize_infer, blockSize_infer>>>(d_X_infer, d_W1, d_b1, d_W2, d_b2,
                                                     d_a1_infer, d_y_pred_infer,
                                                     num_test_samples, num_features, hidden_size);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float time_kernel_infer;
    cudaEventElapsedTime(&time_kernel_infer, start, stop);

    // Copy inference results back to CPU
    cudaEventRecord(start);
    cudaMemcpy(h_y_pred_infer_gpu, d_y_pred_infer, sizeof(float) * num_test_samples, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float time_d2h_infer;
    cudaEventElapsedTime(&time_d2h_infer, start, stop);

    float total_gpu_infer_time = time_h2d_infer + time_kernel_infer + time_d2h_infer;
    std::cout << "\nGPU Inference Timings:\n";
    std::cout << "Host to Device: " << time_h2d_infer << " ms\n";
    std::cout << "Kernel Execution: " << time_kernel_infer << " ms\n";
    std::cout << "Device to Host: " << time_d2h_infer << " ms\n";
    std::cout << "Total GPU Inference Time: " << total_gpu_infer_time << " ms\n";

    // For comparison, perform CPU inference on the same data
    auto cpu_infer_start2 = std::chrono::high_resolution_clock::now();
    mlp_forward_cpu(h_X, h_W1, h_b1, h_W2, h_b2,
                    h_a1_infer, h_y_pred_infer_gpu,
                    num_test_samples, num_features, hidden_size);
    auto cpu_infer_end2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_infer_time2 = cpu_infer_end2 - cpu_infer_start2;
    std::cout << "\nCPU Inference Time (after GPU training): " << cpu_infer_time2.count() << " ms\n";

    // Cleanup GPU memory and events
    cudaFree(d_X);
    cudaFree(d_y);
    cudaFree(d_W1);
    cudaFree(d_b1);
    cudaFree(d_W2);
    cudaFree(d_b2);
    cudaFree(d_a1);
    cudaFree(d_y_pred);
    cudaFree(d_grad_W1);
    cudaFree(d_grad_b1);
    cudaFree(d_grad_W2);
    cudaFree(d_grad_b2);
    cudaFree(d_X_infer);
    cudaFree(d_a1_infer);
    cudaFree(d_y_pred_infer);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Cleanup CPU memory
    delete[] h_X;
    delete[] h_y;
    delete[] true_w;
    delete[] h_W1;
    delete[] h_b1;
    delete[] h_W2;
    delete[] indices_cpu;
    delete[] indices;
    delete[] h_a1_cpu_infer;
    delete[] h_y_pred_cpu_infer;
    delete[] h_a1_infer;
    delete[] h_y_pred_infer_gpu;

    return 0;
}
