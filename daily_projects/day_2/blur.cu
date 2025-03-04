#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define BLUR_SIZE 1

__global__ void blur_kernel(unsigned char* image, unsigned char* blurred, unsigned int width, unsigned int height) {
    int outrow = blockIdx.y * blockDim.y + threadIdx.y;
    int outcol = blockIdx.x * blockDim.x + threadIdx.x;

    if (outrow < height && outcol < width) {
        unsigned int average = 0;
        for (int inrow = outrow - BLUR_SIZE; inrow < outrow + BLUR_SIZE + 1; inrow++) {
            for (int incol = outcol - BLUR_SIZE; incol < outcol + BLUR_SIZE + 1; incol++) {
                if (inrow >= 0 && inrow < height && incol >= 0 && incol < width) {
                    average += image[inrow * width + incol];
                }
            }
        }
        blurred[outrow * width + outcol] = average / ((2 * BLUR_SIZE + 1) * (2 * BLUR_SIZE + 1));
    }
}

void blur_gpu(unsigned char* image, unsigned char* blurred, unsigned int width, unsigned int height) {

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
    unsigned char *image_d, *blurred_d;

    // Allocating memory on the device
    cudaEventRecord(start_alloc, 0);
    cudaMalloc((void**)&image_d, size * sizeof(unsigned char));
    cudaMalloc((void**)&blurred_d, size * sizeof(unsigned char));
    cudaDeviceSynchronize();
    cudaEventRecord(stop_alloc, 0);
    cudaEventSynchronize(stop_alloc);
    cudaEventElapsedTime(&time_alloc, start_alloc, stop_alloc);
    printf("Time for memory allocation: %f ms\n", time_alloc);

    // Copying data from CPU to GPU
    cudaEventRecord(start_h2d, 0);
    cudaMemcpy(image_d, image, size * sizeof(unsigned char), cudaMemcpyHostToDevice);
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
    blur_kernel<<<numBlocks, numThreadsPerBlock>>>(image_d, blurred_d, width, height);
    cudaDeviceSynchronize();
    cudaEventRecord(stop_kernel, 0);
    cudaEventSynchronize(stop_kernel);
    cudaEventElapsedTime(&time_kernel, start_kernel, stop_kernel);
    printf("Time for kernel execution: %f ms\n", time_kernel);

    // Copying data from GPU to CPU
    cudaEventRecord(start_d2h, 0);
    cudaMemcpy(blurred, blurred_d, size * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    cudaEventRecord(stop_d2h, 0);
    cudaEventSynchronize(stop_d2h);
    cudaEventElapsedTime(&time_d2h, start_d2h, stop_d2h);
    printf("Time for device-to-host copy: %f ms\n", time_d2h);

    // Free GPU memory
    cudaFree(image_d);
    cudaFree(blurred_d);
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
    unsigned char* image = (unsigned char*)malloc(size * sizeof(unsigned char));
    unsigned char* blurred = (unsigned char*)malloc(size * sizeof(unsigned char));

    // Initialize host arrays
    for (int i = 0; i < size; i++) {
        image[i] = 255;  // Example: white image.
    }

    // Call the function to blur image
    blur_gpu(image, blurred, width, height);

    // Free CPU memory
    free(image);
    free(blurred);

    return 0;
}