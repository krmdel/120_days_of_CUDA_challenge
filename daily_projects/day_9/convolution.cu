// convolution.cu
#include <stdio.h>
#include <cuda_runtime.h>

#define TILE_WIDTH 32
#define KERNEL_WIDTH 3
#define KERNEL_RADIUS (KERNEL_WIDTH / 2)

__constant__ float d_kernel_const[KERNEL_WIDTH * KERNEL_WIDTH];

// CPU Convolution
void convolutionCPU(float *input, float *output, float *kernel, int width, int height) {
    for (int y = KERNEL_RADIUS; y < height - KERNEL_RADIUS; y++) {
        for (int x = KERNEL_RADIUS; x < width - KERNEL_RADIUS; x++) {
            float sum = 0.0f;
            for (int ky = -KERNEL_RADIUS; ky <= KERNEL_RADIUS; ky++) {
                for (int kx = -KERNEL_RADIUS; kx <= KERNEL_RADIUS; kx++) {
                    sum += input[(y + ky) * width + (x + kx)] * kernel[(ky + KERNEL_RADIUS) * KERNEL_WIDTH + (kx + KERNEL_RADIUS)];
                }
            }
            output[y * width + x] = sum;
        }
    }
}

// GPU Convolution without constant memory
__global__ void convolutionGPU(float *input, float *output, float *kernel, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= KERNEL_RADIUS && x < (width - KERNEL_RADIUS) && y >= KERNEL_RADIUS && y < (height - KERNEL_RADIUS)) {
        float sum = 0.0f;
        for (int ky = -KERNEL_RADIUS; ky <= KERNEL_RADIUS; ky++) {
            for (int kx = -KERNEL_RADIUS; kx <= KERNEL_RADIUS; kx++) {
                sum += input[(y + ky) * width + (x + kx)] * kernel[(ky + KERNEL_RADIUS) * KERNEL_WIDTH + (kx + KERNEL_RADIUS)];
            }
        }
        output[y * width + x] = sum;
    }
}

// GPU Convolution with constant memory
__global__ void convolutionGPUConstant(float *input, float *output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= KERNEL_RADIUS && x < (width - KERNEL_RADIUS) && y >= KERNEL_RADIUS && y < (height - KERNEL_RADIUS)) {
        float sum = 0.0f;
        for (int ky = -KERNEL_RADIUS; ky <= KERNEL_RADIUS; ky++) {
            for (int kx = -KERNEL_RADIUS; kx <= KERNEL_RADIUS; kx++) {
                sum += input[(y + ky) * width + (x + kx)] * d_kernel_const[(ky + KERNEL_RADIUS) * KERNEL_WIDTH + (kx + KERNEL_RADIUS)];
            }
        }
        output[y * width + x] = sum;
    }
}

int main() {
    const int width = 1024;
    const int height = 1024;

    size_t bytes = width * height * sizeof(float);

    float *h_input = (float*)malloc(bytes);
    float *h_output_cpu = (float*)malloc(bytes);
    float *h_output_gpu = (float*)malloc(bytes);
    float *kernel = (float*)malloc(KERNEL_WIDTH * KERNEL_WIDTH * sizeof(float));

    // Initialize input and kernel
    for (int i = 0; i < width * height; i++)
        h_input[i] = 1.0f;
    for (int i = 0; i < KERNEL_WIDTH * KERNEL_WIDTH; i++)
        kernel[i] = 1.0f / (KERNEL_WIDTH * KERNEL_WIDTH);

    // CPU convolution
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    convolutionCPU(h_input, h_output_cpu, kernel, width, height);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float cpu_time;
    cudaEventElapsedTime(&cpu_time, start, stop);

    // Allocate GPU memory
    float *d_input, *d_output, *d_kernel;
    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, bytes);
    cudaMalloc(&d_kernel, KERNEL_WIDTH * KERNEL_WIDTH * sizeof(float));

    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel, KERNEL_WIDTH * KERNEL_WIDTH * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockDim(TILE_WIDTH, TILE_WIDTH);
    dim3 gridDim((width + TILE_WIDTH - 1) / TILE_WIDTH, (height + TILE_WIDTH - 1) / TILE_WIDTH);

    // GPU convolution without constant memory
    cudaEventRecord(start);
    convolutionGPU<<<gridDim, blockDim>>>(d_input, d_output, d_kernel, width, height);
    cudaMemcpy(h_output_gpu, d_output, bytes, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float gpu_time;
    cudaEventElapsedTime(&gpu_time, start, stop);

    // GPU convolution with constant memory
    cudaMemcpyToSymbol(d_kernel_const, kernel, KERNEL_WIDTH * KERNEL_WIDTH * sizeof(float));
    cudaEventRecord(start);
    convolutionGPUConstant<<<gridDim, blockDim>>>(d_input, d_output, width, height);
    cudaMemcpy(h_output_gpu, d_output, bytes, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float gpu_const_time;
    cudaEventElapsedTime(&gpu_const_time, start, stop);

    printf("CPU Time: %f ms\n", cpu_time);
    printf("GPU Time without constant memory: %f ms\n", gpu_time);
    printf("GPU Time with constant memory: %f ms\n", gpu_const_time);

    free(h_input);
    free(h_output_cpu);
    free(h_output_gpu);
    free(kernel);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_kernel);

    return 0;
}