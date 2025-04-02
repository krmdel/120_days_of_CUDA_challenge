#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <float.h>

// Parameters
#define IMAGE_WIDTH         32
#define IMAGE_HEIGHT        32
#define IMAGE_CHANNELS      3

//Input (3 channels) projected into 64 channels.
#define INIT_OUT_CHANNELS   64

// 3x3 convolution with stride=1 and padding=1
#define KERNEL_SIZE         3
#define STRIDE              1
#define PADDING             1

// Output spatial dimensions
#define RES_WIDTH           IMAGE_WIDTH
#define RES_HEIGHT          IMAGE_HEIGHT
#define RES_FEATURE_SIZE    (INIT_OUT_CHANNELS * RES_WIDTH * RES_HEIGHT)

// Training parameters
#define NUM_EPOCHS          10
#define LEARNING_RATE       0.01f

// CPU Functions

// Convolution forward pass
void conv_forward_res_cpu(const float* input, const float* weight, float* output,
                          int in_channels, int out_channels,
                          int in_h, int in_w, int kernel_h, int kernel_w,
                          int stride, int padding) {
    for (int oc = 0; oc < out_channels; oc++) {
        for (int y = 0; y < in_h; y++) {
            for (int x = 0; x < in_w; x++) {
                float sum = 0.0f;
                for (int ic = 0; ic < in_channels; ic++) {
                    for (int i = 0; i < kernel_h; i++) {
                        for (int j = 0; j < kernel_w; j++) {
                            int in_y = y * stride + i - padding;
                            int in_x = x * stride + j - padding;
                            if (in_y >= 0 && in_y < in_h && in_x >= 0 && in_x < in_w) {
                                int input_idx = ic * in_h * in_w + in_y * in_w + in_x;
                                int weight_idx = oc * in_channels * kernel_h * kernel_w +
                                                 ic * kernel_h * kernel_w + i * kernel_w + j;
                                sum += input[input_idx] * weight[weight_idx];
                            }
                        }
                    }
                }
                int output_idx = oc * in_h * in_w + y * in_w + x;
                output[output_idx] = sum;
            }
        }
    }
}

// Batch normalization
void batch_norm_forward_cpu(const float* input, float* output, int channels, int size,
                            const float* gamma, const float* beta, float epsilon) {
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

// ReLU activation
void relu_forward_cpu(const float* input, float* output, int size) {
    for (int i = 0; i < size; i++){
        output[i] = (input[i] > 0.0f) ? input[i] : 0.0f;
    }
}

// Elementwise addition
void elementwise_add_cpu(const float* a, const float* b, float* out, int size) {
    for (int i = 0; i < size; i++){
        out[i] = a[i] + b[i];
    }
}

// Compute convolution weight gradients
void conv_weight_grad_res_cpu(const float* input, const float* d_output, float* d_weight,
                               int in_channels, int out_channels,
                               int in_h, int in_w, int kernel_h, int kernel_w,
                               int stride, int padding) {
    for (int oc = 0; oc < out_channels; oc++){
        for (int ic = 0; ic < in_channels; ic++){
            for (int i = 0; i < kernel_h; i++){
                for (int j = 0; j < kernel_w; j++){
                    float grad = 0.0f;
                    for (int y = 0; y < in_h; y++){
                        for (int x = 0; x < in_w; x++){
                            int in_y = y * stride + i - padding;
                            int in_x = x * stride + j - padding;
                            if (in_y >= 0 && in_y < in_h && in_x >= 0 && in_x < in_w) {
                                int input_idx = ic * in_h * in_w + in_y * in_w + in_x;
                                int output_idx = oc * in_h * in_w + y * in_w + x;
                                grad += input[input_idx] * d_output[output_idx];
                            }
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

// Weight updates
void update_weights_cpu(float* weight, const float* d_weight, float learning_rate, int weight_size) {
    for (int idx = 0; idx < weight_size; idx++) {
        weight[idx] = weight[idx] - learning_rate * d_weight[idx];
    }
}

// GPU Kernels

// Convolution forward pass
__global__ void conv_forward_res(const float* input, const float* weight, float* output,
                                  int in_channels, int out_channels,
                                  int in_h, int in_w, int kernel_h, int kernel_w,
                                  int stride, int padding) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;  // output x index
    int y = blockIdx.y * blockDim.y + threadIdx.y;  // output y index
    int oc = blockIdx.z;                            // output channel index
    if (x < in_w && y < in_h) {
        float sum = 0.0f;
        for (int ic = 0; ic < in_channels; ic++) {
            for (int i = 0; i < kernel_h; i++) {
                for (int j = 0; j < kernel_w; j++) {
                    int in_y = y * stride + i - padding;
                    int in_x = x * stride + j - padding;
                    if (in_y >= 0 && in_y < in_h && in_x >= 0 && in_x < in_w) {
                        int input_idx = ic * in_h * in_w + in_y * in_w + in_x;
                        int weight_idx = oc * in_channels * kernel_h * kernel_w +
                                         ic * kernel_h * kernel_w + i * kernel_w + j;
                        sum += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
        int output_idx = oc * in_h * in_w + y * in_w + x;
        output[output_idx] = sum;
    }
}

// Batch normalization
__global__ void batch_norm_forward_gpu(const float* input, float* output, 
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

// ReLU activation
__global__ void relu_forward_gpu(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < size) {
        output[idx] = (input[idx] > 0.0f) ? input[idx] : 0.0f;
    }
}

// Elementwise addition
__global__ void elementwise_add_gpu(const float* a, const float* b, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < size) {
        out[idx] = a[idx] + b[idx];
    }
}

// Convolution weight gradient
__global__ void conv_weight_grad_res(const float* input, const float* d_output, float* d_weight,
                                       int in_channels, int out_channels,
                                       int in_h, int in_w, int kernel_h, int kernel_w,
                                       int stride, int padding) {
    int oc = blockIdx.x;
    int ic = blockIdx.y;
    int i = threadIdx.y;
    int j = threadIdx.x;
    if (i < kernel_h && j < kernel_w) {
        float grad = 0.0f;
        for (int y = 0; y < in_h; y++){
            for (int x = 0; x < in_w; x++){
                int in_y = y * stride + i - padding;
                int in_x = x * stride + j - padding;
                if(in_y >= 0 && in_y < in_h && in_x >= 0 && in_x < in_w) {
                    int input_idx = ic * in_h * in_w + in_y * in_w + in_x;
                    int output_idx = oc * in_h * in_w + y * in_w + x;
                    grad += input[input_idx] * d_output[output_idx];
                }
            }
        }
        int weight_idx = oc * in_channels * kernel_h * kernel_w +
                         ic * kernel_h * kernel_w + i * kernel_w + j;
        d_weight[weight_idx] = grad;
    }
}

// Dummy gradient computation on GPU: grad = 2 * input / size
__global__ void compute_dummy_grad(const float* input, float* grad, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < size) {
        grad[idx] = 2.0f * input[idx] / size;
    }
}

// Weight updates
__global__ void update_weights(float* weight, const float* d_weight, float learning_rate, int weight_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < weight_size) {
        weight[idx] = weight[idx] - learning_rate * d_weight[idx];
    }
}

// Main: CPU Training, CPU Inference, GPU Training, and GPU Inference
int main(){
    srand(time(NULL));
    float epsilon = 1e-5f;

    // Host memory allocation 
    int input_size = IMAGE_CHANNELS * IMAGE_WIDTH * IMAGE_HEIGHT;
    int weight_init_size = INIT_OUT_CHANNELS * IMAGE_CHANNELS * KERNEL_SIZE * KERNEL_SIZE;
    int weight_res1_size = INIT_OUT_CHANNELS * INIT_OUT_CHANNELS * KERNEL_SIZE * KERNEL_SIZE;
    int weight_res2_size = weight_res1_size;
    
    // Allocate buffers for CPU training/inference
    float *h_input         = (float*)malloc(input_size * sizeof(float));
    float *h_weight_init   = (float*)malloc(weight_init_size * sizeof(float));
    float *h_weight_res1   = (float*)malloc(weight_res1_size * sizeof(float));
    float *h_weight_res2   = (float*)malloc(weight_res2_size * sizeof(float));
    
    // Batch norm parameters
    float *h_gamma1 = (float*)malloc(INIT_OUT_CHANNELS * sizeof(float));
    float *h_beta1  = (float*)malloc(INIT_OUT_CHANNELS * sizeof(float));
    float *h_gamma2 = (float*)malloc(INIT_OUT_CHANNELS * sizeof(float));
    float *h_beta2  = (float*)malloc(INIT_OUT_CHANNELS * sizeof(float));
    
    // Initialize input and weights
    for (int i = 0; i < input_size; i++)
        h_input[i] = (float)rand() / RAND_MAX;
    for (int i = 0; i < weight_init_size; i++)
        h_weight_init[i] = ((float)rand() / RAND_MAX) - 0.5f;
    for (int i = 0; i < weight_res1_size; i++)
        h_weight_res1[i] = ((float)rand() / RAND_MAX) - 0.5f;
    for (int i = 0; i < weight_res2_size; i++)
        h_weight_res2[i] = ((float)rand() / RAND_MAX) - 0.5f;
    for (int i = 0; i < INIT_OUT_CHANNELS; i++){
        h_gamma1[i] = 1.0f; h_beta1[i] = 0.0f;
        h_gamma2[i] = 1.0f; h_beta2[i] = 0.0f;
    }
    
    // Allocate intermediate buffers
    float *h_init_out     = (float*)malloc(RES_FEATURE_SIZE * sizeof(float)); // initial conv output
    float *h_res1_out     = (float*)malloc(RES_FEATURE_SIZE * sizeof(float)); // conv1 output
    float *h_bn1_out      = (float*)malloc(RES_FEATURE_SIZE * sizeof(float)); // BN after conv1
    float *h_relu1_out    = (float*)malloc(RES_FEATURE_SIZE * sizeof(float)); // ReLU after BN1
    float *h_res2_out     = (float*)malloc(RES_FEATURE_SIZE * sizeof(float)); // conv2 output
    float *h_bn2_out      = (float*)malloc(RES_FEATURE_SIZE * sizeof(float)); // BN after conv2
    float *h_skip_out     = (float*)malloc(RES_FEATURE_SIZE * sizeof(float)); // skip addition output
    float *h_resblock_out = (float*)malloc(RES_FEATURE_SIZE * sizeof(float)); // final output after ReLU
    
    // CPU Training
    printf("Starting CPU Training for ResNet Block...\n");
    clock_t cpu_train_start = clock();
    for (int epoch = 0; epoch < NUM_EPOCHS; epoch++){
        // Forward pass
        conv_forward_res_cpu(h_input, h_weight_init, h_init_out,
                             IMAGE_CHANNELS, INIT_OUT_CHANNELS,
                             IMAGE_HEIGHT, IMAGE_WIDTH, KERNEL_SIZE, KERNEL_SIZE,
                             STRIDE, PADDING);
        conv_forward_res_cpu(h_init_out, h_weight_res1, h_res1_out,
                             INIT_OUT_CHANNELS, INIT_OUT_CHANNELS,
                             RES_HEIGHT, RES_WIDTH, KERNEL_SIZE, KERNEL_SIZE,
                             STRIDE, PADDING);
        batch_norm_forward_cpu(h_res1_out, h_bn1_out, INIT_OUT_CHANNELS, RES_WIDTH*RES_HEIGHT,
                               h_gamma1, h_beta1, epsilon);
        relu_forward_cpu(h_bn1_out, h_relu1_out, RES_FEATURE_SIZE);
        conv_forward_res_cpu(h_relu1_out, h_weight_res2, h_res2_out,
                             INIT_OUT_CHANNELS, INIT_OUT_CHANNELS,
                             RES_HEIGHT, RES_WIDTH, KERNEL_SIZE, KERNEL_SIZE,
                             STRIDE, PADDING);
        batch_norm_forward_cpu(h_res2_out, h_bn2_out, INIT_OUT_CHANNELS, RES_WIDTH*RES_HEIGHT,
                               h_gamma2, h_beta2, epsilon);
        elementwise_add_cpu(h_bn2_out, h_init_out, h_skip_out, RES_FEATURE_SIZE);
        relu_forward_cpu(h_skip_out, h_resblock_out, RES_FEATURE_SIZE);
        
        // Compute loss (mean squared error)
        float loss = 0.0f;
        for (int i = 0; i < RES_FEATURE_SIZE; i++){
            loss += h_resblock_out[i] * h_resblock_out[i];
        }
        loss /= RES_FEATURE_SIZE;
        printf("CPU Epoch %d Loss: %f\n", epoch+1, loss);
        
        // Dummy Backpropagation: compute dummy gradients and update convolution weights
        float *d_grad_init = (float*)malloc(RES_FEATURE_SIZE * sizeof(float));
        float *d_grad_res1 = (float*)malloc(RES_FEATURE_SIZE * sizeof(float));
        float *d_grad_res2 = (float*)malloc(RES_FEATURE_SIZE * sizeof(float));
        for (int i = 0; i < RES_FEATURE_SIZE; i++){
            d_grad_init[i] = 2.0f * h_init_out[i] / RES_FEATURE_SIZE;
            d_grad_res1[i] = 2.0f * h_res1_out[i] / RES_FEATURE_SIZE;
            d_grad_res2[i] = 2.0f * h_res2_out[i] / RES_FEATURE_SIZE;
        }
        float *dW_init = (float*)malloc(weight_init_size * sizeof(float));
        float *dW_res1 = (float*)malloc(weight_res1_size * sizeof(float));
        float *dW_res2 = (float*)malloc(weight_res2_size * sizeof(float));
        
        conv_weight_grad_res_cpu(h_input, d_grad_init, dW_init,
                                   IMAGE_CHANNELS, INIT_OUT_CHANNELS,
                                   IMAGE_HEIGHT, IMAGE_WIDTH, KERNEL_SIZE, KERNEL_SIZE,
                                   STRIDE, PADDING);
        conv_weight_grad_res_cpu(h_init_out, d_grad_res1, dW_res1,
                                   INIT_OUT_CHANNELS, INIT_OUT_CHANNELS,
                                   RES_HEIGHT, RES_WIDTH, KERNEL_SIZE, KERNEL_SIZE,
                                   STRIDE, PADDING);
        conv_weight_grad_res_cpu(h_relu1_out, d_grad_res2, dW_res2,
                                   INIT_OUT_CHANNELS, INIT_OUT_CHANNELS,
                                   RES_HEIGHT, RES_WIDTH, KERNEL_SIZE, KERNEL_SIZE,
                                   STRIDE, PADDING);
        update_weights_cpu(h_weight_init, dW_init, LEARNING_RATE, weight_init_size);
        update_weights_cpu(h_weight_res1, dW_res1, LEARNING_RATE, weight_res1_size);
        update_weights_cpu(h_weight_res2, dW_res2, LEARNING_RATE, weight_res2_size);
        
        free(d_grad_init); free(d_grad_res1); free(d_grad_res2);
        free(dW_init); free(dW_res1); free(dW_res2);
    }
    clock_t cpu_train_end = clock();
    double cpu_train_time = ((double)(cpu_train_end - cpu_train_start)) / CLOCKS_PER_SEC * 1000.0;
    printf("Total CPU Training Time: %f ms\n", cpu_train_time);
    
    // CPU Inference
    clock_t cpu_inf_start = clock();
    conv_forward_res_cpu(h_input, h_weight_init, h_init_out,
                         IMAGE_CHANNELS, INIT_OUT_CHANNELS,
                         IMAGE_HEIGHT, IMAGE_WIDTH, KERNEL_SIZE, KERNEL_SIZE,
                         STRIDE, PADDING);
    conv_forward_res_cpu(h_init_out, h_weight_res1, h_res1_out,
                         INIT_OUT_CHANNELS, INIT_OUT_CHANNELS,
                         RES_HEIGHT, RES_WIDTH, KERNEL_SIZE, KERNEL_SIZE,
                         STRIDE, PADDING);
    batch_norm_forward_cpu(h_res1_out, h_bn1_out, INIT_OUT_CHANNELS, RES_WIDTH*RES_HEIGHT,
                           h_gamma1, h_beta1, epsilon);
    relu_forward_cpu(h_bn1_out, h_relu1_out, RES_FEATURE_SIZE);
    conv_forward_res_cpu(h_relu1_out, h_weight_res2, h_res2_out,
                         INIT_OUT_CHANNELS, INIT_OUT_CHANNELS,
                         RES_HEIGHT, RES_WIDTH, KERNEL_SIZE, KERNEL_SIZE,
                         STRIDE, PADDING);
    batch_norm_forward_cpu(h_res2_out, h_bn2_out, INIT_OUT_CHANNELS, RES_WIDTH*RES_HEIGHT,
                           h_gamma2, h_beta2, epsilon);
    elementwise_add_cpu(h_bn2_out, h_init_out, h_skip_out, RES_FEATURE_SIZE);
    relu_forward_cpu(h_skip_out, h_resblock_out, RES_FEATURE_SIZE);
    clock_t cpu_inf_end = clock();
    double cpu_inf_time = ((double)(cpu_inf_end - cpu_inf_start)) / CLOCKS_PER_SEC * 1000.0;
    printf("CPU Inference Time: %f ms\n", cpu_inf_time);
    
    // GPU Memory Allocation
    printf("\nStarting GPU Training for ResNet Block...\n");
    float *d_input, *d_weight_init, *d_weight_res1, *d_weight_res2;
    float *d_init_out, *d_res1_out, *d_bn1_out, *d_relu1_out;
    float *d_res2_out, *d_bn2_out, *d_skip_out, *d_resblock_out;
    cudaMalloc((void**)&d_input,         input_size * sizeof(float));
    cudaMalloc((void**)&d_weight_init,   weight_init_size * sizeof(float));
    cudaMalloc((void**)&d_weight_res1,   weight_res1_size * sizeof(float));
    cudaMalloc((void**)&d_weight_res2,   weight_res2_size * sizeof(float));
    cudaMalloc((void**)&d_init_out,      RES_FEATURE_SIZE * sizeof(float));
    cudaMalloc((void**)&d_res1_out,      RES_FEATURE_SIZE * sizeof(float));
    cudaMalloc((void**)&d_bn1_out,       RES_FEATURE_SIZE * sizeof(float));
    cudaMalloc((void**)&d_relu1_out,     RES_FEATURE_SIZE * sizeof(float));
    cudaMalloc((void**)&d_res2_out,      RES_FEATURE_SIZE * sizeof(float));
    cudaMalloc((void**)&d_bn2_out,       RES_FEATURE_SIZE * sizeof(float));
    cudaMalloc((void**)&d_skip_out,      RES_FEATURE_SIZE * sizeof(float));
    cudaMalloc((void**)&d_resblock_out,  RES_FEATURE_SIZE * sizeof(float));
    
    // Allocate and copy batch norm parameters to device
    float *d_gamma1, *d_beta1, *d_gamma2, *d_beta2;
    cudaMalloc((void**)&d_gamma1, INIT_OUT_CHANNELS * sizeof(float));
    cudaMalloc((void**)&d_beta1,  INIT_OUT_CHANNELS * sizeof(float));
    cudaMalloc((void**)&d_gamma2, INIT_OUT_CHANNELS * sizeof(float));
    cudaMalloc((void**)&d_beta2,  INIT_OUT_CHANNELS * sizeof(float));
    cudaMemcpy(d_gamma1, h_gamma1, INIT_OUT_CHANNELS * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_beta1,  h_beta1,  INIT_OUT_CHANNELS * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_gamma2, h_gamma2, INIT_OUT_CHANNELS * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_beta2,  h_beta2,  INIT_OUT_CHANNELS * sizeof(float), cudaMemcpyHostToDevice);
    
    // Copy input and initial weights to device
    cudaMemcpy(d_input, h_input, input_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight_init, h_weight_init, weight_init_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight_res1, h_weight_res1, weight_res1_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight_res2, h_weight_res2, weight_res2_size * sizeof(float), cudaMemcpyHostToDevice);
    
    // Set up grid and block sizes for convolution.
    dim3 blockSizeRes(16, 16, 1);
    dim3 gridSizeRes((IMAGE_WIDTH + blockSizeRes.x - 1) / blockSizeRes.x,
                     (IMAGE_HEIGHT + blockSizeRes.y - 1) / blockSizeRes.y,
                     INIT_OUT_CHANNELS);
    
    // For dummy gradient and weight update kernels
    int numElements = RES_FEATURE_SIZE;
    int threads1D = 256;
    int blocks1D = (numElements + threads1D - 1) / threads1D;
    
    // Allocate device buffers for dummy gradients and weight gradients
    float *d_grad_init, *d_grad_res1, *d_grad_res2;
    float *dW_init, *dW_res1, *dW_res2;
    cudaMalloc((void**)&d_grad_init, RES_FEATURE_SIZE * sizeof(float));
    cudaMalloc((void**)&d_grad_res1, RES_FEATURE_SIZE * sizeof(float));
    cudaMalloc((void**)&d_grad_res2, RES_FEATURE_SIZE * sizeof(float));
    cudaMalloc((void**)&dW_init, weight_init_size * sizeof(float));
    cudaMalloc((void**)&dW_res1, weight_res1_size * sizeof(float));
    cudaMalloc((void**)&dW_res2, weight_res2_size * sizeof(float));
    
    // GPU Training
    cudaEvent_t gpu_train_start, gpu_train_end;
    cudaEventCreate(&gpu_train_start);
    cudaEventCreate(&gpu_train_end);
    cudaEventRecord(gpu_train_start, 0);
    
    for (int epoch = 0; epoch < NUM_EPOCHS; epoch++){
        // Forward Pass
        conv_forward_res<<<gridSizeRes, blockSizeRes>>>(d_input, d_weight_init, d_init_out,
                                                        IMAGE_CHANNELS, INIT_OUT_CHANNELS,
                                                        IMAGE_HEIGHT, IMAGE_WIDTH, KERNEL_SIZE, KERNEL_SIZE,
                                                        STRIDE, PADDING);
        cudaDeviceSynchronize();
        conv_forward_res<<<gridSizeRes, blockSizeRes>>>(d_init_out, d_weight_res1, d_res1_out,
                                                        INIT_OUT_CHANNELS, INIT_OUT_CHANNELS,
                                                        IMAGE_HEIGHT, IMAGE_WIDTH, KERNEL_SIZE, KERNEL_SIZE,
                                                        STRIDE, PADDING);
        cudaDeviceSynchronize();
        batch_norm_forward_gpu<<< (INIT_OUT_CHANNELS + 31)/32, 32>>>(d_res1_out, d_bn1_out,
                                                                      INIT_OUT_CHANNELS, IMAGE_WIDTH*IMAGE_HEIGHT,
                                                                      d_gamma1, d_beta1, epsilon);
        cudaDeviceSynchronize();
        relu_forward_gpu<<<blocks1D, threads1D>>>(d_bn1_out, d_relu1_out, RES_FEATURE_SIZE);
        cudaDeviceSynchronize();
        conv_forward_res<<<gridSizeRes, blockSizeRes>>>(d_relu1_out, d_weight_res2, d_res2_out,
                                                        INIT_OUT_CHANNELS, INIT_OUT_CHANNELS,
                                                        IMAGE_HEIGHT, IMAGE_WIDTH, KERNEL_SIZE, KERNEL_SIZE,
                                                        STRIDE, PADDING);
        cudaDeviceSynchronize();
        batch_norm_forward_gpu<<< (INIT_OUT_CHANNELS + 31)/32, 32>>>(d_res2_out, d_bn2_out,
                                                                      INIT_OUT_CHANNELS, IMAGE_WIDTH*IMAGE_HEIGHT,
                                                                      d_gamma2, d_beta2, epsilon);
        cudaDeviceSynchronize();
        int addElements = RES_FEATURE_SIZE;
        int addBlocks = (addElements + threads1D - 1) / threads1D;
        elementwise_add_gpu<<<addBlocks, threads1D>>>(d_bn2_out, d_init_out, d_skip_out, addElements);
        cudaDeviceSynchronize();
        relu_forward_gpu<<<blocks1D, threads1D>>>(d_skip_out, d_resblock_out, RES_FEATURE_SIZE);
        cudaDeviceSynchronize();
        
        // Compute loss: copy final output to host and compute MSE
        float *h_gpu_out = (float*)malloc(RES_FEATURE_SIZE * sizeof(float));
        cudaMemcpy(h_gpu_out, d_resblock_out, RES_FEATURE_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
        float gpu_loss = 0.0f;
        for (int i = 0; i < RES_FEATURE_SIZE; i++){
            gpu_loss += h_gpu_out[i] * h_gpu_out[i];
        }
        gpu_loss /= RES_FEATURE_SIZE;
        printf("GPU Epoch %d Loss: %f\n", epoch+1, gpu_loss);
        free(h_gpu_out);
        
        // Dummy Backpropagation: compute dummy gradients
        compute_dummy_grad<<<blocks1D, threads1D>>>(d_init_out, d_grad_init, RES_FEATURE_SIZE);
        cudaDeviceSynchronize();
        compute_dummy_grad<<<blocks1D, threads1D>>>(d_res1_out, d_grad_res1, RES_FEATURE_SIZE);
        cudaDeviceSynchronize();
        compute_dummy_grad<<<blocks1D, threads1D>>>(d_res2_out, d_grad_res2, RES_FEATURE_SIZE);
        cudaDeviceSynchronize();
        
        // Weight gradient computation
        dim3 gridGrad(INIT_OUT_CHANNELS, IMAGE_CHANNELS, 1);
        dim3 blockGrad(KERNEL_SIZE, KERNEL_SIZE, 1);
        conv_weight_grad_res<<<gridGrad, blockGrad>>>(d_input, d_grad_init, dW_init,
                                                        IMAGE_CHANNELS, INIT_OUT_CHANNELS,
                                                        IMAGE_HEIGHT, IMAGE_WIDTH, KERNEL_SIZE, KERNEL_SIZE,
                                                        STRIDE, PADDING);
        cudaDeviceSynchronize();
        dim3 gridGrad1(INIT_OUT_CHANNELS, INIT_OUT_CHANNELS, 1);
        conv_weight_grad_res<<<gridGrad1, blockGrad>>>(d_init_out, d_grad_res1, dW_res1,
                                                         INIT_OUT_CHANNELS, INIT_OUT_CHANNELS,
                                                         IMAGE_HEIGHT, IMAGE_WIDTH, KERNEL_SIZE, KERNEL_SIZE,
                                                         STRIDE, PADDING);
        cudaDeviceSynchronize();
        conv_weight_grad_res<<<gridGrad1, blockGrad>>>(d_relu1_out, d_grad_res2, dW_res2,
                                                         INIT_OUT_CHANNELS, INIT_OUT_CHANNELS,
                                                         IMAGE_HEIGHT, IMAGE_WIDTH, KERNEL_SIZE, KERNEL_SIZE,
                                                         STRIDE, PADDING);
        cudaDeviceSynchronize();
        
        // Update weights
        int blocksW_init = (weight_init_size + threads1D - 1) / threads1D;
        update_weights<<<blocksW_init, threads1D>>>(d_weight_init, dW_init, LEARNING_RATE, weight_init_size);
        cudaDeviceSynchronize();
        int blocksW_res1 = (weight_res1_size + threads1D - 1) / threads1D;
        update_weights<<<blocksW_res1, threads1D>>>(d_weight_res1, dW_res1, LEARNING_RATE, weight_res1_size);
        cudaDeviceSynchronize();
        int blocksW_res2 = (weight_res2_size + threads1D - 1) / threads1D;
        update_weights<<<blocksW_res2, threads1D>>>(d_weight_res2, dW_res2, LEARNING_RATE, weight_res2_size);
        cudaDeviceSynchronize();
    }
    cudaEventRecord(gpu_train_end, 0);
    cudaEventSynchronize(gpu_train_end);
    float gpu_train_time = 0.0f;
    cudaEventElapsedTime(&gpu_train_time, gpu_train_start, gpu_train_end);
    printf("Total GPU Training Time: %f ms\n", gpu_train_time);
    
    // GPU Inference
    cudaEvent_t inf_start, inf_end;
    cudaEventCreate(&inf_start);
    cudaEventCreate(&inf_end);
    cudaEventRecord(inf_start, 0);
    conv_forward_res<<<gridSizeRes, blockSizeRes>>>(d_input, d_weight_init, d_init_out,
                                                    IMAGE_CHANNELS, INIT_OUT_CHANNELS,
                                                    IMAGE_HEIGHT, IMAGE_WIDTH, KERNEL_SIZE, KERNEL_SIZE,
                                                    STRIDE, PADDING);
    cudaDeviceSynchronize();
    conv_forward_res<<<gridSizeRes, blockSizeRes>>>(d_init_out, d_weight_res1, d_res1_out,
                                                    INIT_OUT_CHANNELS, INIT_OUT_CHANNELS,
                                                    IMAGE_HEIGHT, IMAGE_WIDTH, KERNEL_SIZE, KERNEL_SIZE,
                                                    STRIDE, PADDING);
    cudaDeviceSynchronize();
    batch_norm_forward_gpu<<< (INIT_OUT_CHANNELS + 31)/32, 32>>>(d_res1_out, d_bn1_out,
                                                                  INIT_OUT_CHANNELS, IMAGE_WIDTH*IMAGE_HEIGHT,
                                                                  d_gamma1, d_beta1, epsilon);
    cudaDeviceSynchronize();
    relu_forward_gpu<<<blocks1D, threads1D>>>(d_bn1_out, d_relu1_out, RES_FEATURE_SIZE);
    cudaDeviceSynchronize();
    conv_forward_res<<<gridSizeRes, blockSizeRes>>>(d_relu1_out, d_weight_res2, d_res2_out,
                                                    INIT_OUT_CHANNELS, INIT_OUT_CHANNELS,
                                                    IMAGE_HEIGHT, IMAGE_WIDTH, KERNEL_SIZE, KERNEL_SIZE,
                                                    STRIDE, PADDING);
    cudaDeviceSynchronize();
    batch_norm_forward_gpu<<< (INIT_OUT_CHANNELS + 31)/32, 32>>>(d_res2_out, d_bn2_out,
                                                                  INIT_OUT_CHANNELS, IMAGE_WIDTH*IMAGE_HEIGHT,
                                                                  d_gamma2, d_beta2, epsilon);
    cudaDeviceSynchronize();
    int addBlocks = (RES_FEATURE_SIZE + threads1D - 1) / threads1D;
    elementwise_add_gpu<<<addBlocks, threads1D>>>(d_bn2_out, d_init_out, d_skip_out, RES_FEATURE_SIZE);
    cudaDeviceSynchronize();
    relu_forward_gpu<<<blocks1D, threads1D>>>(d_skip_out, d_resblock_out, RES_FEATURE_SIZE);
    cudaDeviceSynchronize();
    cudaEventRecord(inf_end, 0);
    cudaEventSynchronize(inf_end);
    float gpu_inf_time = 0.0f;
    cudaEventElapsedTime(&gpu_inf_time, inf_start, inf_end);
    printf("GPU Inference Time: %f ms\n", gpu_inf_time);
    
    // Clean up host and device memory
    free(h_input);
    free(h_weight_init);
    free(h_weight_res1);
    free(h_weight_res2);
    free(h_gamma1); free(h_beta1);
    free(h_gamma2); free(h_beta2);
    free(h_init_out);
    free(h_res1_out);
    free(h_bn1_out);
    free(h_relu1_out);
    free(h_res2_out);
    free(h_bn2_out);
    free(h_skip_out);
    free(h_resblock_out);
    
    cudaFree(d_input);
    cudaFree(d_weight_init);
    cudaFree(d_weight_res1);
    cudaFree(d_weight_res2);
    cudaFree(d_init_out);
    cudaFree(d_res1_out);
    cudaFree(d_bn1_out);
    cudaFree(d_relu1_out);
    cudaFree(d_res2_out);
    cudaFree(d_bn2_out);
    cudaFree(d_skip_out);
    cudaFree(d_resblock_out);
    cudaFree(d_gamma1); cudaFree(d_beta1);
    cudaFree(d_gamma2); cudaFree(d_beta2);
    cudaFree(d_grad_init); cudaFree(d_grad_res1); cudaFree(d_grad_res2);
    cudaFree(dW_init); cudaFree(dW_res1); cudaFree(dW_res2);
    
    return 0;
}
