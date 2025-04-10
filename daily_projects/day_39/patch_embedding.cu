#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include <cstdlib>

// Error checking macro for CUDA calls
#define cudaCheck(error)                                                   \
    if(error != cudaSuccess) {                                             \
        std::cerr << "CUDA Error: " << cudaGetErrorString(error)           \
                  << " in file " << __FILE__ << " at line " << __LINE__      \
                  << std::endl;                                            \
        exit(EXIT_FAILURE);                                                \
    }

// Image/patch parameters
const int IMAGE_HEIGHT = 1024;
const int IMAGE_WIDTH  = 1024;
const int PATCH_SIZE   = 32;  // image dimensions are divisible by PATCH_SIZE.
const int NUM_PATCHES_H = IMAGE_HEIGHT / PATCH_SIZE;
const int NUM_PATCHES_W = IMAGE_WIDTH / PATCH_SIZE;
const int NUM_PATCHES   = NUM_PATCHES_H * NUM_PATCHES_W;
const int PATCH_VEC_SIZE = PATCH_SIZE * PATCH_SIZE;
const int EMBEDDING_DIM = 128;

// CPU implementation
void cpu_patch_embedding(const float* image, const float* weight, float* output)
{
    for (int patchIdx = 0; patchIdx < NUM_PATCHES; ++patchIdx) {
        int patch_row = patchIdx / NUM_PATCHES_W;
        int patch_col = patchIdx % NUM_PATCHES_W;
        int start_row = patch_row * PATCH_SIZE;
        int start_col = patch_col * PATCH_SIZE;
        
        for (int d = 0; d < EMBEDDING_DIM; ++d) {
            float sum = 0.0f;
            for (int i = 0; i < PATCH_SIZE; ++i) {
                for (int j = 0; j < PATCH_SIZE; ++j) {
                    int imgRow = start_row + i;
                    int imgCol = start_col + j;
                    float pix = image[imgRow * IMAGE_WIDTH + imgCol];
                    int weightIdx = ((i * PATCH_SIZE) + j) * EMBEDDING_DIM + d;
                    sum += pix * weight[weightIdx];
                }
            }
            output[patchIdx * EMBEDDING_DIM + d] = sum;
        }
    }
}

// CUDA kernels using shared memory tiling
__global__ void patch_embedding_kernel_tiled(const float* image, const float* weight, float* output,
                                               int imageWidth, int patchSize, int embeddingDim)
{
    // Each block handles one patch
    int patchIdx = blockIdx.x;
    if (patchIdx >= NUM_PATCHES) return;
    
    // Calculate the starting row and column for the patch
    int numPatches_w = imageWidth / patchSize;
    int patch_row = patchIdx / numPatches_w;
    int patch_col = patchIdx % numPatches_w;
    int start_row = patch_row * patchSize;
    int start_col = patch_col * patchSize;
    
    // Allocate shared memory for the patch
    __shared__ float s_patch[PATCH_VEC_SIZE];
    
    // Each block should have at least PATCH_VEC_SIZE threads (assuming blockDim.x >= EMBEDDING_DIM and EMBEDDING_DIM >= PATCH_VEC_SIZE)
    int tid = threadIdx.x;
    
    // Let the first 64 threads load the patch pixels into shared memory
    if (tid < PATCH_VEC_SIZE) {
         int i = tid / patchSize;
         int j = tid % patchSize;
         int imgRow = start_row + i;
         int imgCol = start_col + j;
         s_patch[tid] = image[imgRow * imageWidth + imgCol];
    }
    __syncthreads();
    
    // Each thread computes the dot product for one embedding dimension
    if (tid < embeddingDim) {
         float sum = 0.0f;
         // Unroll the loop over the patch vector
         #pragma unroll
         for (int k = 0; k < PATCH_VEC_SIZE; ++k) {
              // Get the weight corresponding to the kth pixel for this embedding dimension.
              int weightIdx = k * embeddingDim + tid;
              sum += s_patch[k] * weight[weightIdx];
         }
         output[patchIdx * embeddingDim + tid] = sum;
    }
}

int main()
{
    // Allocate host memory
    const int imageSize = IMAGE_HEIGHT * IMAGE_WIDTH;
    const int weightSize = PATCH_VEC_SIZE * EMBEDDING_DIM;
    const int outputSize = NUM_PATCHES * EMBEDDING_DIM;
    
    float* h_image  = new float[imageSize];
    float* h_weight = new float[weightSize];
    float* h_output_cpu = new float[outputSize];
    float* h_output_gpu = new float[outputSize];
    
    // Initialize image and weight with random values
    for (int i = 0; i < imageSize; ++i)
        h_image[i] = static_cast<float>(rand()) / RAND_MAX;
    for (int i = 0; i < weightSize; ++i)
        h_weight[i] = static_cast<float>(rand()) / RAND_MAX;

    // CPU Implementation
    auto cpu_start = std::chrono::high_resolution_clock::now();
    cpu_patch_embedding(h_image, h_weight, h_output_cpu);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_duration = cpu_end - cpu_start;
    std::cout << "CPU patch embedding time: " << cpu_duration.count() << " ms" << std::endl;
    
    // GPU Implementation
    float *d_image, *d_weight, *d_output;
    cudaCheck(cudaMalloc((void**)&d_image,  imageSize * sizeof(float)));
    cudaCheck(cudaMalloc((void**)&d_weight, weightSize * sizeof(float)));
    cudaCheck(cudaMalloc((void**)&d_output, outputSize * sizeof(float)));
    
    // Set up CUDA events for timing
    cudaEvent_t start_total, stop_total;
    cudaEvent_t start_H2D, stop_H2D;
    cudaEvent_t start_kernel, stop_kernel;
    cudaEvent_t start_D2H, stop_D2H;
    
    cudaCheck(cudaEventCreate(&start_total));
    cudaCheck(cudaEventCreate(&stop_total));
    cudaCheck(cudaEventCreate(&start_H2D));
    cudaCheck(cudaEventCreate(&stop_H2D));
    cudaCheck(cudaEventCreate(&start_kernel));
    cudaCheck(cudaEventCreate(&stop_kernel));
    cudaCheck(cudaEventCreate(&start_D2H));
    cudaCheck(cudaEventCreate(&stop_D2H));
    
    // Record the overall GPU start time
    cudaCheck(cudaEventRecord(start_total, 0));
    
    // Copy data from CPU to GPU.
    cudaCheck(cudaEventRecord(start_H2D, 0));
    cudaCheck(cudaMemcpy(d_image, h_image, imageSize * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_weight, h_weight, weightSize * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaEventRecord(stop_H2D, 0));
    cudaCheck(cudaEventSynchronize(stop_H2D));
    float time_H2D;
    cudaCheck(cudaEventElapsedTime(&time_H2D, start_H2D, stop_H2D));
    
    // Execute the kernel
    // Launch one block per patch with EMBEDDING_DIM threads per block
    dim3 blockDim(EMBEDDING_DIM);  // 128 threads per block
    dim3 gridDim(NUM_PATCHES);     // One block per patch
    
    cudaCheck(cudaEventRecord(start_kernel, 0));
    patch_embedding_kernel_tiled<<<gridDim, blockDim>>>(d_image, d_weight, d_output,
                                                        IMAGE_WIDTH, PATCH_SIZE, EMBEDDING_DIM);
    cudaCheck(cudaEventRecord(stop_kernel, 0));
    cudaCheck(cudaEventSynchronize(stop_kernel));
    float time_kernel;
    cudaCheck(cudaEventElapsedTime(&time_kernel, start_kernel, stop_kernel));
    
    // Copy the results back from GPU to CPU
    cudaCheck(cudaEventRecord(start_D2H, 0));
    cudaCheck(cudaMemcpy(h_output_gpu, d_output, outputSize * sizeof(float), cudaMemcpyDeviceToHost));
    cudaCheck(cudaEventRecord(stop_D2H, 0));
    cudaCheck(cudaEventSynchronize(stop_D2H));
    float time_D2H;
    cudaCheck(cudaEventElapsedTime(&time_D2H, start_D2H, stop_D2H));
    
    // Record overall GPU end time
    cudaCheck(cudaEventRecord(stop_total, 0));
    cudaCheck(cudaEventSynchronize(stop_total));
    float time_total;
    cudaCheck(cudaEventElapsedTime(&time_total, start_total, stop_total));
    
    // Print GPU timing details
    std::cout << "GPU timings (in ms):" << std::endl;
    std::cout << "  Host-to-Device copy time: " << time_H2D << std::endl;
    std::cout << "  Kernel execution time:   " << time_kernel << std::endl;
    std::cout << "  Device-to-Host copy time: " << time_D2H << std::endl;
    std::cout << "  Total GPU time:          " << time_total << std::endl;
        
    // Cleanup
    delete[] h_image;
    delete[] h_weight;
    delete[] h_output_cpu;
    delete[] h_output_gpu;
    cudaFree(d_image);
    cudaFree(d_weight);
    cudaFree(d_output);
    cudaEventDestroy(start_total);
    cudaEventDestroy(stop_total);
    cudaEventDestroy(start_H2D);
    cudaEventDestroy(stop_H2D);
    cudaEventDestroy(start_kernel);
    cudaEventDestroy(stop_kernel);
    cudaEventDestroy(start_D2H);
    cudaEventDestroy(stop_D2H);
    
    return 0;
}
