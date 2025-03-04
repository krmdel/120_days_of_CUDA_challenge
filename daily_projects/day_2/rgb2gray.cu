#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void rgb2gray_kernel(unsigned char* red, unsigned char* green, unsigned char* blue, unsigned char* gray, unsigned int width, unsigned int height) {
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < height && col < width) {
        unsigned int i = row * width + col;
        gray[i] = 3/10 * red[i] + 6/10 * green[i] + 1/10 * blue[i];
    }
}

void rgb2gray_gpu(unsigned char* red, unsigned char* green, unsigned char* blue, unsigned char* gray, unsigned int width, unsigned int height) {
    
    // Create CUDA events for timing different stages
    cudaEvent_t start_alloc, stop_alloc;
    cudaEvent_t start_h2d, stop_h2d;
    cudaEvent_t start_kernel, stop_kernel;
    cudaEvent_t start_d2h, stop_d2h;
    float time_alloc, time_h2d, time_kernel, time_d2h;

    cudaEventCreate(&start_alloc);
    cudaEventCreate(&stop_alloc);
    cudaEventCreate(&start_h2d);
    cudaEventCreate(&stop_h2d);
    cudaEventCreate(&start_kernel);
    cudaEventCreate(&stop_kernel);
    cudaEventCreate(&start_d2h);
    cudaEventCreate(&stop_d2h);

    int size = width * height;
    unsigned char *red_d, *green_d, *blue_d, *gray_d;

    // Allocating memory on the device
    cudaEventRecord(start_alloc, 0);
    cudaMalloc((void**)&red_d, size * sizeof(unsigned char));
    cudaMalloc((void**)&green_d, size * sizeof(unsigned char));
    cudaMalloc((void**)&blue_d, size * sizeof(unsigned char));
    cudaMalloc((void**)&gray_d, size * sizeof(unsigned char));
    cudaDeviceSynchronize();
    cudaEventRecord(stop_alloc, 0);
    cudaEventSynchronize(stop_alloc);
    cudaEventElapsedTime(&time_alloc, start_alloc, stop_alloc);
    printf("Time for memory allocation: %f ms\n", time_alloc);

    // Copying data from CPU to GPU
    cudaEventRecord(start_h2d, 0);
    cudaMemcpy(red_d, red, size * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(green_d, green, size * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(blue_d, blue, size * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    cudaEventRecord(stop_h2d, 0);
    cudaEventSynchronize(stop_h2d);
    cudaEventElapsedTime(&time_h2d, start_h2d, stop_h2d);
    printf("Time for host-to-device copy: %f ms\n", time_h2d);

    // Calling the kernel and performing the operation
    cudaEventRecord(start_kernel, 0);
    dim3 numThreadsPerBlock(32, 32);
    dim3 numBlocks((width + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x,
                   (height + numThreadsPerBlock.y - 1) / numThreadsPerBlock.y);
    rgb2gray_kernel<<<numBlocks, numThreadsPerBlock>>>(red_d, green_d, blue_d, gray_d, width, height);
    cudaDeviceSynchronize();
    cudaEventRecord(stop_kernel, 0);
    cudaEventSynchronize(stop_kernel);
    cudaEventElapsedTime(&time_kernel, start_kernel, stop_kernel);
    printf("Time for kernel execution: %f ms\n", time_kernel);

    // Copying data from GPU to CPU
    cudaEventRecord(start_d2h, 0);
    cudaMemcpy(gray, gray_d, size * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    cudaEventRecord(stop_d2h, 0);
    cudaEventSynchronize(stop_d2h);
    cudaEventElapsedTime(&time_d2h, start_d2h, stop_d2h);
    printf("Time for device-to-host copy: %f ms\n", time_d2h);

    // Free GPU memory
    cudaFree(red_d);
    cudaFree(green_d);
    cudaFree(blue_d);
    cudaFree(gray_d);
    cudaDeviceSynchronize();

    // Clean up events
    cudaEventDestroy(start_alloc);
    cudaEventDestroy(stop_alloc);
    cudaEventDestroy(start_h2d);
    cudaEventDestroy(stop_h2d);
    cudaEventDestroy(start_kernel);
    cudaEventDestroy(stop_kernel);
    cudaEventDestroy(start_d2h);
    cudaEventDestroy(stop_d2h);
}

int main() {

    unsigned int width = 1024;
    unsigned int height = 1024;
    int size = width * height;

    // Allocate memory on CPU
    unsigned char* red   = (unsigned char*)malloc(size * sizeof(unsigned char));
    unsigned char* green = (unsigned char*)malloc(size * sizeof(unsigned char));
    unsigned char* blue  = (unsigned char*)malloc(size * sizeof(unsigned char));
    unsigned char* gray  = (unsigned char*)malloc(size * sizeof(unsigned char));

    // Initialize host arrays
    for (int i = 0; i < size; i++) {
        red[i]   = 255;  // maximum intensity for red
        green[i] = 255;  // medium intensity for green
        blue[i] = 255;   // low intensity for blue
    }

    // Call the function to convert RGB to grayscale
    rgb2gray_gpu(red, green, blue, gray, width, height);

    // Free CPU memory
    free(red);
    free(green);
    free(blue);
    free(gray);

    return 0;
}