#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <float.h>  // for FLT_MAX

// Image Parameters
#define IMAGE_WIDTH    32
#define IMAGE_HEIGHT   32
#define IMAGE_CHANNELS 3

// CNN input parameters
#define IN_WIDTH     IMAGE_WIDTH
#define IN_HEIGHT    IMAGE_HEIGHT
#define IN_CHANNELS  IMAGE_CHANNELS

// Network parameters
#define OUT_CHANNELS 1
#define KERNEL_SIZE  3
#define STRIDE       2

#define OUT_HEIGHT ((IN_HEIGHT - KERNEL_SIZE) / STRIDE + 1)
#define OUT_WIDTH  ((IN_WIDTH  - KERNEL_SIZE) / STRIDE + 1)

// Pooling parameters
#define POOL_SIZE   2
#define POOL_STRIDE 2
#define POOL_OUT_HEIGHT ((OUT_HEIGHT - POOL_SIZE) / POOL_STRIDE + 1)
#define POOL_OUT_WIDTH  ((OUT_WIDTH  - POOL_SIZE) / POOL_STRIDE + 1)

// GPU Kernels:

// Convolution kernel
__global__ void conv_forward(const float* input, const float* weight, float* output,
                             int in_channels, int out_channels,
                             int in_h, int in_w, int kernel_h, int kernel_w, int stride) {
    int ox = blockIdx.x * blockDim.x + threadIdx.x;
    int oy = blockIdx.y * blockDim.y + threadIdx.y;
    int oc = blockIdx.z;
    if (ox < OUT_WIDTH && oy < OUT_HEIGHT) {
        float sum = 0.0f;
        for (int ic = 0; ic < in_channels; ic++) {
            for (int i = 0; i < kernel_h; i++) {
                for (int j = 0; j < kernel_w; j++) {
                    int in_y = oy * stride + i;
                    int in_x = ox * stride + j;
                    int input_idx = ic * in_h * in_w + in_y * in_w + in_x;
                    int weight_idx = oc * in_channels * kernel_h * kernel_w +
                                     ic * kernel_h * kernel_w + i * kernel_w + j;
                    sum += input[input_idx] * weight[weight_idx];
                }
            }
        }
        int output_idx = oc * OUT_HEIGHT * OUT_WIDTH + oy * OUT_WIDTH + ox;
        output[output_idx] = sum;
    }
}

// Kernel to compute gradients for the convolution weights
__global__ void conv_weight_grad(const float* input, const float* d_output, float* d_weight,
                                   int in_channels, int out_channels,
                                   int in_h, int in_w, int kernel_h, int kernel_w, int stride) {
    int oc = blockIdx.x; // output channel index
    int ic = blockIdx.y; // input channel index
    int i = threadIdx.y; // kernel row index
    int j = threadIdx.x; // kernel column index
    if (i < kernel_h && j < kernel_w) {
        float grad = 0.0f;
        for (int oy = 0; oy < OUT_HEIGHT; oy++) {
            for (int ox = 0; ox < OUT_WIDTH; ox++) {
                int in_y = oy * stride + i;
                int in_x = ox * stride + j;
                int input_idx = ic * in_h * in_w + in_y * in_w + in_x;
                int output_idx = oc * OUT_HEIGHT * OUT_WIDTH + oy * OUT_WIDTH + ox;
                grad += input[input_idx] * d_output[output_idx];
            }
        }
        int weight_idx = oc * in_channels * kernel_h * kernel_w +
                         ic * kernel_h * kernel_w + i * kernel_w + j;
        d_weight[weight_idx] = grad;
    }
}

// Kernel to update weights on GPU using gradient descent
__global__ void update_weights(float* weight, const float* d_weight, float learning_rate, int weight_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < weight_size) {
        weight[idx] = weight[idx] - learning_rate * d_weight[idx];
    }
}

// Max Pooling Forward Kernel
__global__ void max_pool_forward(const float* input, float* output, 
                                 int channels, int in_h, int in_w, 
                                 int pool_size, int pool_stride) {
    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    int c  = blockIdx.z;
    if (ow < POOL_OUT_WIDTH && oh < POOL_OUT_HEIGHT) {
        float max_val = -FLT_MAX;
        for (int i = 0; i < pool_size; i++) {
            for (int j = 0; j < pool_size; j++) {
                int in_y = oh * pool_stride + i;
                int in_x = ow * pool_stride + j;
                int index = c * in_h * in_w + in_y * in_w + in_x;
                if (input[index] > max_val)
                    max_val = input[index];
            }
        }
        int out_index = c * POOL_OUT_HEIGHT * POOL_OUT_WIDTH + oh * POOL_OUT_WIDTH + ow;
        output[out_index] = max_val;
    }
}

// Average Pooling Forward Kernel
__global__ void avg_pool_forward(const float* input, float* output, 
                                 int channels, int in_h, int in_w, 
                                 int pool_size, int pool_stride) {
    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    int c  = blockIdx.z;
    if (ow < POOL_OUT_WIDTH && oh < POOL_OUT_HEIGHT) {
        float sum = 0.0f;
        for (int i = 0; i < pool_size; i++) {
            for (int j = 0; j < pool_size; j++) {
                int in_y = oh * pool_stride + i;
                int in_x = ow * pool_stride + j;
                int index = c * in_h * in_w + in_y * in_w + in_x;
                sum += input[index];
            }
        }
        int out_index = c * POOL_OUT_HEIGHT * POOL_OUT_WIDTH + oh * POOL_OUT_WIDTH + ow;
        output[out_index] = sum / (pool_size * pool_size);
    }
}

// Batch Normalization Forward Kernel
__global__ void batch_norm_forward(const float* input, float* output, 
                                   int channels, int size, 
                                   const float* gamma, const float* beta, float epsilon) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c < channels) {
        float sum = 0.0f;
        for (int i = 0; i < size; i++){
            int index = c * size + i;
            sum += input[index];
        }
        float mean = sum / size;
        float var = 0.0f;
        for (int i = 0; i < size; i++){
            int index = c * size + i;
            float diff = input[index] - mean;
            var += diff * diff;
        }
        var /= size;
        float inv_std = 1.0f / sqrtf(var + epsilon);
        for (int i = 0; i < size; i++){
            int index = c * size + i;
            output[index] = gamma[c] * (input[index] - mean) * inv_std + beta[c];
        }
    }
}

// Dropout Forward Kernel
__global__ void dropout_forward(const float* input, float* output, int size, float dropout_prob, unsigned int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        unsigned int x = seed ^ idx;
        x = (x * 1664525u + 1013904223u);
        float rand_val = (float)(x & 0x00FFFFFF) / (float)0x01000000;
        if (rand_val < dropout_prob)
            output[idx] = 0.0f;
        else
            output[idx] = input[idx] / (1.0f - dropout_prob);
    }
}

// CPU Functions:

// Convolution
void conv_forward_cpu(const float* input, const float* weight, float* output,
                      int in_channels, int out_channels,
                      int in_h, int in_w, int kernel_h, int kernel_w, int stride) {
    for (int oc = 0; oc < out_channels; oc++) {
        for (int oy = 0; oy < OUT_HEIGHT; oy++) {
            for (int ox = 0; ox < OUT_WIDTH; ox++) {
                float sum = 0.0f;
                for (int ic = 0; ic < in_channels; ic++) {
                    for (int i = 0; i < kernel_h; i++) {
                        for (int j = 0; j < kernel_w; j++) {
                            int in_y = oy * stride + i;
                            int in_x = ox * stride + j;
                            int input_idx = ic * in_h * in_w + in_y * in_w + in_x;
                            int weight_idx = oc * in_channels * kernel_h * kernel_w +
                                             ic * kernel_h * kernel_w + i * kernel_w + j;
                            sum += input[input_idx] * weight[weight_idx];
                        }
                    }
                }
                int output_idx = oc * OUT_HEIGHT * OUT_WIDTH + oy * OUT_WIDTH + ox;
                output[output_idx] = sum;
            }
        }
    }
}

// Gradient computation for convolution weights
void conv_weight_grad_cpu(const float* input, const float* d_output, float* d_weight,
                          int in_channels, int out_channels,
                          int in_h, int in_w, int kernel_h, int kernel_w, int stride) {
    for (int oc = 0; oc < out_channels; oc++) {
        for (int ic = 0; ic < in_channels; ic++) {
            for (int i = 0; i < kernel_h; i++) {
                for (int j = 0; j < kernel_w; j++) {
                    float grad = 0.0f;
                    for (int oy = 0; oy < OUT_HEIGHT; oy++) {
                        for (int ox = 0; ox < OUT_WIDTH; ox++){
                            int in_y = oy * stride + i;
                            int in_x = ox * stride + j;
                            int input_idx = ic * in_h * in_w + in_y * in_w + in_x;
                            int output_idx = oc * OUT_HEIGHT * OUT_WIDTH + oy * OUT_WIDTH + ox;
                            grad += input[input_idx] * d_output[output_idx];
                        }
                    }
                    int weight_idx = oc * in_channels * kernel_h * kernel_w +
                                     ic * kernel_h * kernel_w + i * kernel_w + j;
                    d_weight[weight_idx] = grad;
                }
            }
        }
    }
}

// Weight update
void update_weights_cpu(float* weight, const float* d_weight, float learning_rate, int weight_size) {
    for (int idx = 0; idx < weight_size; idx++) {
        weight[idx] = weight[idx] - learning_rate * d_weight[idx];
    }
}

// Max Pooling Forward
void max_pool_forward_cpu(const float* input, float* output, int channels, int in_h, int in_w, int pool_size, int pool_stride) {
    for (int c = 0; c < channels; c++){
        for (int oh = 0; oh < POOL_OUT_HEIGHT; oh++){
            for (int ow = 0; ow < POOL_OUT_WIDTH; ow++){
                float max_val = -FLT_MAX;
                for (int i = 0; i < pool_size; i++){
                    for (int j = 0; j < pool_size; j++){
                        int in_y = oh * pool_stride + i;
                        int in_x = ow * pool_stride + j;
                        int index = c * in_h * in_w + in_y * in_w + in_x;
                        if (input[index] > max_val)
                            max_val = input[index];
                    }
                }
                int out_index = c * POOL_OUT_HEIGHT * POOL_OUT_WIDTH + oh * POOL_OUT_WIDTH + ow;
                output[out_index] = max_val;
            }
        }
    }
}

// Average Pooling Forward
void avg_pool_forward_cpu(const float* input, float* output, int channels, int in_h, int in_w, int pool_size, int pool_stride) {
    for (int c = 0; c < channels; c++){
        for (int oh = 0; oh < POOL_OUT_HEIGHT; oh++){
            for (int ow = 0; ow < POOL_OUT_WIDTH; ow++){
                float sum = 0.0f;
                for (int i = 0; i < pool_size; i++){
                    for (int j = 0; j < pool_size; j++){
                        int in_y = oh * pool_stride + i;
                        int in_x = ow * pool_stride + j;
                        int index = c * in_h * in_w + in_y * in_w + in_x;
                        sum += input[index];
                    }
                }
                int out_index = c * POOL_OUT_HEIGHT * POOL_OUT_WIDTH + oh * POOL_OUT_WIDTH + ow;
                output[out_index] = sum / (pool_size * pool_size);
            }
        }
    }
}

// Batch Normalization Forward
void batch_norm_forward_cpu(const float* input, float* output, int channels, int size, const float* gamma, const float* beta, float epsilon) {
    for (int c = 0; c < channels; c++) {
       float sum = 0.0f;
       for (int i = 0; i < size; i++){
           int index = c * size + i;
           sum += input[index];
       }
       float mean = sum / size;
       float var = 0.0f;
       for (int i = 0; i < size; i++){
           int index = c * size + i;
           float diff = input[index] - mean;
           var += diff * diff;
       }
       var /= size;
       float inv_std = 1.0f / sqrtf(var + epsilon);
       for (int i = 0; i < size; i++){
           int index = c * size + i;
           output[index] = gamma[c] * (input[index] - mean) * inv_std + beta[c];
       }
    }
}

// Dropout Forward
void dropout_forward_cpu(const float* input, float* output, int size, float dropout_prob) {
    for (int i = 0; i < size; i++) {
        float rand_val = (float)rand() / RAND_MAX;
        if (rand_val < dropout_prob)
            output[i] = 0.0f;
        else
            output[i] = input[i] / (1.0f - dropout_prob);
    }
}

int main(){
    srand(time(NULL));

    int input_size  = IN_CHANNELS * IN_HEIGHT * IN_WIDTH;
    int weight_size = OUT_CHANNELS * IN_CHANNELS * KERNEL_SIZE * KERNEL_SIZE;
    int conv_output_size = OUT_CHANNELS * OUT_HEIGHT * OUT_WIDTH;
    int pool_output_size = OUT_CHANNELS * POOL_OUT_HEIGHT * POOL_OUT_WIDTH;

    // Allocate host memory for convolution
    float* h_input    = (float*)malloc(input_size * sizeof(float));
    float* h_weight   = (float*)malloc(weight_size * sizeof(float));
    float* h_output   = (float*)malloc(conv_output_size * sizeof(float));

    // Allocate host memory for gradients
    float* h_d_output = (float*)malloc(conv_output_size * sizeof(float));

    // Allocate host memory for new layers (pooling, batch norm, dropout)
    float* h_pool_output   = (float*)malloc(pool_output_size * sizeof(float));
    float* h_bn_output     = (float*)malloc(pool_output_size * sizeof(float));
    float* h_dropout_output= (float*)malloc(pool_output_size * sizeof(float));

    // Initialize input and weights
    for (int i = 0; i < input_size; i++){
        h_input[i] = (float)rand() / RAND_MAX;
    }
    for (int i = 0; i < weight_size; i++){
        h_weight[i] = ((float)rand() / RAND_MAX) - 0.5f;
    }

    // Parameters for batch normalization and dropout
    float dropout_prob = 0.5f;
    float epsilon = 1e-5f;
    float* h_gamma = (float*)malloc(OUT_CHANNELS * sizeof(float));
    float* h_beta  = (float*)malloc(OUT_CHANNELS * sizeof(float));
    for (int i = 0; i < OUT_CHANNELS; i++){
        h_gamma[i] = 1.0f;
        h_beta[i]  = 0.0f;
    }

    // Make a separate copy of initial weights for CPU training
    float* cpu_weight = (float*)malloc(weight_size * sizeof(float));
    memcpy(cpu_weight, h_weight, weight_size * sizeof(float));

    int num_epochs = 10;
    float learning_rate = 0.01f;
    
    printf("Starting CPU Training for CNN (convolution only)...\n");
    clock_t cpu_train_start = clock();
    double total_cpu_epoch_time = 0.0;
    for (int epoch = 0; epoch < num_epochs; epoch++){
        clock_t epoch_start = clock();
        // CPU forward pass
        conv_forward_cpu(h_input, cpu_weight, h_output,
                         IN_CHANNELS, OUT_CHANNELS,
                         IN_HEIGHT, IN_WIDTH, KERNEL_SIZE, KERNEL_SIZE, STRIDE);
        // Compute loss and output gradient (loss computed on conv output)
        float loss = 0.0f;
        for (int i = 0; i < conv_output_size; i++){
            float diff = h_output[i];
            loss += diff * diff;
            h_d_output[i] = 2.0f * diff / conv_output_size;
        }
        loss /= conv_output_size;
        // Compute weight gradient on CPU (for convolution)
        float* cpu_d_weight = (float*)malloc(weight_size * sizeof(float));
        conv_weight_grad_cpu(h_input, h_d_output, cpu_d_weight,
                             IN_CHANNELS, OUT_CHANNELS,
                             IN_HEIGHT, IN_WIDTH, KERNEL_SIZE, KERNEL_SIZE, STRIDE);
        // Update weights on CPU
        update_weights_cpu(cpu_weight, cpu_d_weight, learning_rate, weight_size);
        free(cpu_d_weight);
        clock_t epoch_end = clock();
        double epoch_time = ((double)(epoch_end - epoch_start)) / CLOCKS_PER_SEC * 1000.0;
        total_cpu_epoch_time += epoch_time;
        printf("CPU Epoch %d/%d Loss: %f Epoch Time (CPU): %.4f ms\n", epoch+1, num_epochs, loss, epoch_time);
    }
    clock_t cpu_train_end = clock();
    double total_cpu_train_time = ((double)(cpu_train_end - cpu_train_start)) / CLOCKS_PER_SEC * 1000.0;

    // CPU Inference including new layers (pooling, batch norm, dropout)
    clock_t cpu_inf_start = clock();
    // Convolution forward
    conv_forward_cpu(h_input, cpu_weight, h_output,
                     IN_CHANNELS, OUT_CHANNELS,
                     IN_HEIGHT, IN_WIDTH, KERNEL_SIZE, KERNEL_SIZE, STRIDE);
    // Pooling
    max_pool_forward_cpu(h_output, h_pool_output, OUT_CHANNELS, OUT_HEIGHT, OUT_WIDTH, POOL_SIZE, POOL_STRIDE);
    // Batch Normalization
    batch_norm_forward_cpu(h_pool_output, h_bn_output, OUT_CHANNELS, POOL_OUT_HEIGHT*POOL_OUT_WIDTH, h_gamma, h_beta, epsilon);
    // Dropout
    dropout_forward_cpu(h_bn_output, h_dropout_output, pool_output_size, dropout_prob);
    clock_t cpu_inf_end = clock();
    double cpu_inf_time = ((double)(cpu_inf_end - cpu_inf_start)) / CLOCKS_PER_SEC * 1000.0;

    // GPU Training and inference for convolution (unchanged from your original code)
    float *d_input, *d_weight, *d_output, *d_d_output, *d_d_weight;
    cudaMalloc((void**)&d_input,    input_size * sizeof(float));
    cudaMalloc((void**)&d_weight,   weight_size * sizeof(float));
    cudaMalloc((void**)&d_output,   conv_output_size * sizeof(float));
    cudaMalloc((void**)&d_d_output, conv_output_size * sizeof(float));
    cudaMalloc((void**)&d_d_weight, weight_size * sizeof(float));

    // Copy input and weights to GPU and measure Host->Device copy time
    cudaEvent_t start_copy, stop_copy;
    cudaEventCreate(&start_copy);
    cudaEventCreate(&stop_copy);
    cudaEventRecord(start_copy, 0);
    cudaMemcpy(d_input, h_input, input_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight, h_weight, weight_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    cudaEventRecord(stop_copy, 0);
    cudaEventSynchronize(stop_copy);
    float copy_h2d_time;
    cudaEventElapsedTime(&copy_h2d_time, start_copy, stop_copy);
    cudaEventDestroy(start_copy);
    cudaEventDestroy(stop_copy);

    // GPU kernel parameters for convolution
    dim3 blockSize(16, 16, 1);
    dim3 gridSize((OUT_WIDTH + blockSize.x - 1) / blockSize.x,
                  (OUT_HEIGHT + blockSize.y - 1) / blockSize.y,
                  OUT_CHANNELS);
    dim3 gridSizeGrad(OUT_CHANNELS, IN_CHANNELS, 1);
    dim3 blockSizeGrad(KERNEL_SIZE, KERNEL_SIZE, 1);
    int threadsPerBlock = 256;
    int blocksPerGrid = (weight_size + threadsPerBlock - 1) / threadsPerBlock;

    // GPU training:
    float total_gpu_kernel_time = 0.0f;
    float total_gpu_h2d_time = 0.0f;
    float total_gpu_d2h_time = 0.0f;

    cudaEvent_t gpu_train_start, gpu_train_end;
    cudaEventCreate(&gpu_train_start);
    cudaEventCreate(&gpu_train_end);
    cudaEventRecord(gpu_train_start, 0);

    printf("\nStarting GPU Training for CNN (convolution only)...\n");
    for (int epoch = 0; epoch < num_epochs; epoch++){
        clock_t gpu_epoch_start = clock();

        // Forward pass kernel
        cudaEvent_t start_k, end_k;
        cudaEventCreate(&start_k);
        cudaEventCreate(&end_k);
        cudaEventRecord(start_k, 0);
        conv_forward<<<gridSize, blockSize>>>(d_input, d_weight, d_output,
                                              IN_CHANNELS, OUT_CHANNELS,
                                              IN_HEIGHT, IN_WIDTH, KERNEL_SIZE, KERNEL_SIZE, STRIDE);
        cudaDeviceSynchronize();
        cudaEventRecord(end_k, 0);
        cudaEventSynchronize(end_k);
        float t_conv = 0.0f;
        cudaEventElapsedTime(&t_conv, start_k, end_k);
        total_gpu_kernel_time += t_conv;
        cudaEventDestroy(start_k);
        cudaEventDestroy(end_k);

        // Copy convolution output from GPU (Device -> Host) for loss computation
        cudaEvent_t start_d2h, end_d2h;
        cudaEventCreate(&start_d2h);
        cudaEventCreate(&end_d2h);
        cudaEventRecord(start_d2h, 0);
        cudaMemcpy(h_output, d_output, conv_output_size * sizeof(float), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        cudaEventRecord(end_d2h, 0);
        cudaEventSynchronize(end_d2h);
        float t_d2h = 0.0f;
        cudaEventElapsedTime(&t_d2h, start_d2h, end_d2h);
        total_gpu_d2h_time += t_d2h;
        cudaEventDestroy(start_d2h);
        cudaEventDestroy(end_d2h);

        // Loss and gradient computation on CPU (for convolution output)
        float loss = 0.0f;
        for (int i = 0; i < conv_output_size; i++){
            float diff = h_output[i];
            loss += diff * diff;
            h_d_output[i] = 2.0f * diff / conv_output_size;
        }
        loss /= conv_output_size;

        // Copy gradient to GPU (Host -> Device)
        cudaEvent_t start_h2d, end_h2d;
        cudaEventCreate(&start_h2d);
        cudaEventCreate(&end_h2d);
        cudaEventRecord(start_h2d, 0);
        cudaMemcpy(d_d_output, h_d_output, conv_output_size * sizeof(float), cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();
        cudaEventRecord(end_h2d, 0);
        cudaEventSynchronize(end_h2d);
        float t_h2d = 0.0f;
        cudaEventElapsedTime(&t_h2d, start_h2d, end_h2d);
        total_gpu_h2d_time += t_h2d;
        cudaEventDestroy(start_h2d);
        cudaEventDestroy(end_h2d);

        // Backward pass: compute weight gradient
        cudaEventCreate(&start_k);
        cudaEventCreate(&end_k);
        cudaEventRecord(start_k, 0);
        conv_weight_grad<<<gridSizeGrad, blockSizeGrad>>>(d_input, d_d_output, d_d_weight,
                                                          IN_CHANNELS, OUT_CHANNELS,
                                                          IN_HEIGHT, IN_WIDTH, KERNEL_SIZE, KERNEL_SIZE, STRIDE);
        cudaDeviceSynchronize();
        cudaEventRecord(end_k, 0);
        cudaEventSynchronize(end_k);
        float t_grad = 0.0f;
        cudaEventElapsedTime(&t_grad, start_k, end_k);
        total_gpu_kernel_time += t_grad;
        cudaEventDestroy(start_k);
        cudaEventDestroy(end_k);

        // Update weights kernel
        cudaEventCreate(&start_k);
        cudaEventCreate(&end_k);
        cudaEventRecord(start_k, 0);
        update_weights<<<blocksPerGrid, threadsPerBlock>>>(d_weight, d_d_weight, learning_rate, weight_size);
        cudaDeviceSynchronize();
        cudaEventRecord(end_k, 0);
        cudaEventSynchronize(end_k);
        float t_update = 0.0f;
        cudaEventElapsedTime(&t_update, start_k, end_k);
        total_gpu_kernel_time += t_update;
        cudaEventDestroy(start_k);
        cudaEventDestroy(end_k);

        clock_t gpu_epoch_end = clock();
        double epoch_cpu_time = ((double)(gpu_epoch_end - gpu_epoch_start)) / CLOCKS_PER_SEC * 1000.0;
        printf("GPU Epoch %d/%d Loss: %f Epoch Time (CPU): %.4f ms\n", epoch+1, num_epochs, loss, epoch_cpu_time);
    }
    cudaEventRecord(gpu_train_end, 0);
    cudaEventSynchronize(gpu_train_end);
    float total_gpu_train_time = 0.0f;
    cudaEventElapsedTime(&total_gpu_train_time, gpu_train_start, gpu_train_end);
    cudaEventDestroy(gpu_train_start);
    cudaEventDestroy(gpu_train_end);

    // GPU Inference including new layers (Pooling, Batch Norm, Dropout)

    // Allocate device memory for new layers
    float *d_pool_output, *d_bn_output, *d_dropout_output;
    cudaMalloc((void**)&d_pool_output, pool_output_size * sizeof(float));
    cudaMalloc((void**)&d_bn_output, pool_output_size * sizeof(float));
    cudaMalloc((void**)&d_dropout_output, pool_output_size * sizeof(float));

    // Allocate and copy batch norm parameters to device
    float *d_gamma, *d_beta;
    cudaMalloc((void**)&d_gamma, OUT_CHANNELS * sizeof(float));
    cudaMalloc((void**)&d_beta,  OUT_CHANNELS * sizeof(float));
    cudaMemcpy(d_gamma, h_gamma, OUT_CHANNELS * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_beta, h_beta, OUT_CHANNELS * sizeof(float), cudaMemcpyHostToDevice);

    unsigned int dropout_seed = time(NULL);

    // GPU Inference for convolution
    cudaEvent_t inf_start_h2d, inf_end_h2d;
    cudaEventCreate(&inf_start_h2d);
    cudaEventCreate(&inf_end_h2d);
    cudaEventRecord(inf_start_h2d, 0);
    cudaMemcpy(d_input, h_input, input_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    cudaEventRecord(inf_end_h2d, 0);
    cudaEventSynchronize(inf_end_h2d);
    float inf_h2d_time = 0.0f;
    cudaEventElapsedTime(&inf_h2d_time, inf_start_h2d, inf_end_h2d);
    cudaEventDestroy(inf_start_h2d);
    cudaEventDestroy(inf_end_h2d);

    // Convolution inference kernel execution
    cudaEvent_t inf_start_kernel, inf_end_kernel;
    cudaEventCreate(&inf_start_kernel);
    cudaEventCreate(&inf_end_kernel);
    cudaEventRecord(inf_start_kernel, 0);
    conv_forward<<<gridSize, blockSize>>>(d_input, d_weight, d_output,
                                          IN_CHANNELS, OUT_CHANNELS,
                                          IN_HEIGHT, IN_WIDTH, KERNEL_SIZE, KERNEL_SIZE, STRIDE);
    cudaDeviceSynchronize();
    cudaEventRecord(inf_end_kernel, 0);
    cudaEventSynchronize(inf_end_kernel);
    float inf_kernel_time = 0.0f;
    cudaEventElapsedTime(&inf_kernel_time, inf_start_kernel, inf_end_kernel);
    cudaEventDestroy(inf_start_kernel);
    cudaEventDestroy(inf_end_kernel);

    // GPU: Pooling, Batch Norm and Dropout
    dim3 poolBlockSize(16, 16, 1);
    dim3 poolGridSize((POOL_OUT_WIDTH + poolBlockSize.x - 1) / poolBlockSize.x,
                      (POOL_OUT_HEIGHT + poolBlockSize.y - 1) / poolBlockSize.y,
                      OUT_CHANNELS);
    // Max Pooling kernel
    cudaEvent_t pool_start, pool_end;
    cudaEventCreate(&pool_start);
    cudaEventCreate(&pool_end);
    cudaEventRecord(pool_start, 0);
    max_pool_forward<<<poolGridSize, poolBlockSize>>>(d_output, d_pool_output,
                                                      OUT_CHANNELS, OUT_HEIGHT, OUT_WIDTH, POOL_SIZE, POOL_STRIDE);
    cudaDeviceSynchronize();
    cudaEventRecord(pool_end, 0);
    cudaEventSynchronize(pool_end);
    float t_pool = 0.0f;
    cudaEventElapsedTime(&t_pool, pool_start, pool_end);
    cudaEventDestroy(pool_start);
    cudaEventDestroy(pool_end);

    // Batch Norm kernel execution
    int bn_size = POOL_OUT_HEIGHT * POOL_OUT_WIDTH;
    int bnThreads = 32;
    int bnBlocks = (OUT_CHANNELS + bnThreads - 1) / bnThreads;
    cudaEvent_t bn_start, bn_end;
    cudaEventCreate(&bn_start);
    cudaEventCreate(&bn_end);
    cudaEventRecord(bn_start, 0);
    batch_norm_forward<<<bnBlocks, bnThreads>>>(d_pool_output, d_bn_output, OUT_CHANNELS, bn_size, d_gamma, d_beta, epsilon);
    cudaDeviceSynchronize();
    cudaEventRecord(bn_end, 0);
    cudaEventSynchronize(bn_end);
    float t_bn = 0.0f;
    cudaEventElapsedTime(&t_bn, bn_start, bn_end);
    cudaEventDestroy(bn_start);
    cudaEventDestroy(bn_end);

    // Dropout kernel execution
    int dropoutThreads = 256;
    int dropoutBlocks = (pool_output_size + dropoutThreads - 1) / dropoutThreads;
    cudaEvent_t dropout_start, dropout_end;
    cudaEventCreate(&dropout_start);
    cudaEventCreate(&dropout_end);
    cudaEventRecord(dropout_start, 0);
    dropout_forward<<<dropoutBlocks, dropoutThreads>>>(d_bn_output, d_dropout_output, pool_output_size, dropout_prob, dropout_seed);
    cudaDeviceSynchronize();
    cudaEventRecord(dropout_end, 0);
    cudaEventSynchronize(dropout_end);
    float t_dropout = 0.0f;
    cudaEventElapsedTime(&t_dropout, dropout_start, dropout_end);
    cudaEventDestroy(dropout_start);
    cudaEventDestroy(dropout_end);

    // Copy inference output from convolution and new layers (Device -> Host)
    float* h_gpu_dropout_output = (float*)malloc(pool_output_size * sizeof(float));
    cudaEvent_t inf_start_d2h, inf_end_d2h;
    cudaEventCreate(&inf_start_d2h);
    cudaEventCreate(&inf_end_d2h);
    cudaEventRecord(inf_start_d2h, 0);
    cudaMemcpy(h_gpu_dropout_output, d_dropout_output, pool_output_size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    cudaEventRecord(inf_end_d2h, 0);
    cudaEventSynchronize(inf_end_d2h);
    float inf_d2h_time = 0.0f;
    cudaEventElapsedTime(&inf_d2h_time, inf_start_d2h, inf_end_d2h);
    cudaEventDestroy(inf_start_d2h);
    cudaEventDestroy(inf_end_d2h);

    float total_gpu_inf_time = inf_h2d_time + inf_kernel_time + t_pool + t_bn + t_dropout + inf_d2h_time;

    printf("\n--- Timing Results ---\n");
    printf("CPU Training: Total Epoch Time (sum): %.4f ms\n", total_cpu_epoch_time);
    printf("CPU Training: Wall Clock Time: %f ms\n", total_cpu_train_time);
    printf("CPU Inference Time (including new layers): %.4f ms\n", cpu_inf_time);
    
    printf("\n--- GPU Training Timings (convolution only) ---\n");
    printf("Total Host to Device (training): %f ms\n", total_gpu_h2d_time);
    printf("Total Kernel Execution (training): %f ms\n", total_gpu_kernel_time);
    printf("Total Device to Host (training): %f ms\n", total_gpu_d2h_time);
    printf("Total GPU Training Time: %f ms\n", total_gpu_train_time);
    
    printf("\n--- GPU Inference Timings ---\n");
    printf("Host to Device: %f ms\n", inf_h2d_time);
    printf("Convolution Kernel Execution: %f ms\n", inf_kernel_time);
    printf("Pooling Kernel: %f ms\n", t_pool);
    printf("Batch Norm Kernel: %f ms\n", t_bn);
    printf("Dropout Kernel: %f ms\n", t_dropout);
    printf("Device to Host: %f ms\n", inf_d2h_time);
    printf("Total GPU Inference Time (with new layers): %f ms\n", total_gpu_inf_time);

    // Clean up host memory
    free(h_input);
    free(h_weight);
    free(h_output);
    free(h_d_output);
    free(cpu_weight);
    free(h_pool_output);
    free(h_bn_output);
    free(h_dropout_output);
    free(h_gamma);
    free(h_beta);
    free(h_gpu_dropout_output);

    // Clean up device memory
    cudaFree(d_input);
    cudaFree(d_weight);
    cudaFree(d_output);
    cudaFree(d_d_output);
    cudaFree(d_d_weight);
    cudaFree(d_pool_output);
    cudaFree(d_bn_output);
    cudaFree(d_dropout_output);
    cudaFree(d_gamma);
    cudaFree(d_beta);

    return 0;
}
