#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include <cstdlib>
#include <cmath>

// Error checking macro for CUDA calls
#define cudaCheck(error)                                                   \
    if (error != cudaSuccess) {                                            \
        std::cerr << "CUDA Error: " << cudaGetErrorString(error)           \
                  << " in file " << __FILE__ << " at line " << __LINE__      \
                  << std::endl;                                            \
        exit(EXIT_FAILURE);                                                \
    }

// Image/patch parameters
const int IMAGE_HEIGHT  = 1024;
const int IMAGE_WIDTH   = 1024;
const int PATCH_SIZE    = 8;  // dimensions should be divisible by PATCH_SIZE.
const int NUM_PATCHES_H = IMAGE_HEIGHT / PATCH_SIZE;
const int NUM_PATCHES_W = IMAGE_WIDTH  / PATCH_SIZE;
const int NUM_PATCHES   = NUM_PATCHES_H * NUM_PATCHES_W;  // e.g., 128 * 128 = 16384
const int EMBEDDING_DIM = 128;

// CPU Implementation: Add positional embeddings to patch embeddings
void add_positional_embeddings_cpu(float* embeddings, int num_patches, int embedding_dim)
{
    for (int p = 0; p < num_patches; p++) {
        for (int d = 0; d < embedding_dim; d++) {
            float exponent = (2 * (d / 2)) / (float)embedding_dim;
            float angle    = p / pow(10000.0f, exponent);
            float pos_val  = ((d % 2) == 0) ? sin(angle) : cos(angle);
            embeddings[p * embedding_dim + d] += pos_val;
        }
    }
}

// GPU kernels using tiling with shared memory
__global__ void add_positional_embeddings_tiled(float* embeddings, int num_patches, int embedding_dim)
{
    int patch = blockIdx.x;
    if (patch >= num_patches) return;

    // Declare shared memory for a tile of embedding elements
    extern __shared__ float s_tile[];  // shared memory size will be provided during launch

    // Process the embedding vector in tiles
    for (int tile_start = 0; tile_start < embedding_dim; tile_start += blockDim.x) {
        int idx = tile_start + threadIdx.x;
        if (idx < embedding_dim) {
            int global_idx = patch * embedding_dim + idx;
            // Load current embedding element into shared memory
            s_tile[threadIdx.x] = embeddings[global_idx];

            // Compute positional encoding value
            float exponent = (2 * (idx / 2)) / (float)embedding_dim;
            float angle    = patch / powf(10000.0f, exponent);
            float pos_val  = ((idx % 2) == 0) ? sinf(angle) : cosf(angle);

            // Add the positional encoding
            s_tile[threadIdx.x] += pos_val;
        }
        __syncthreads();

        if (idx < embedding_dim) {
            int global_idx = patch * embedding_dim + idx;
            // Write back the result from shared memory to global memory
            embeddings[global_idx] = s_tile[threadIdx.x];
        }
        __syncthreads();
    }
}

int main()
{
    // Allocate host memory for patch embeddings
    const int total_embeddings = NUM_PATCHES * EMBEDDING_DIM;
    float* h_embeddings_cpu = new float[total_embeddings];
    float* h_embeddings_gpu = new float[total_embeddings];

    // Initialize patch embeddings with random values
    for (int i = 0; i < total_embeddings; i++) {
        h_embeddings_cpu[i] = static_cast<float>(rand()) / RAND_MAX;
        h_embeddings_gpu[i] = h_embeddings_cpu[i]; // same initial values for both versions
    }

    
    // CPU Timing: Add positional embeddings on CPU
    auto cpu_start = std::chrono::high_resolution_clock::now();
    add_positional_embeddings_cpu(h_embeddings_cpu, NUM_PATCHES, EMBEDDING_DIM);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_duration = cpu_end - cpu_start;
    std::cout << "CPU positional embeddings addition time: " << cpu_duration.count() << " ms" << std::endl;

    // GPU Timing: Add positional embeddings using CUDA
    float *d_embeddings;
    cudaCheck(cudaMalloc((void**)&d_embeddings, total_embeddings * sizeof(float)));

    // Create CUDA events to measure GPU timings
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

    // Record overall GPU start time
    cudaCheck(cudaEventRecord(start_total, 0));

    // Copy embeddings from CPU to GPU
    cudaCheck(cudaEventRecord(start_H2D, 0));
    cudaCheck(cudaMemcpy(d_embeddings, h_embeddings_gpu, total_embeddings * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaEventRecord(stop_H2D, 0));
    cudaCheck(cudaEventSynchronize(stop_H2D));
    float time_H2D;
    cudaCheck(cudaEventElapsedTime(&time_H2D, start_H2D, stop_H2D));

    // Launch the GPU kernel
    // Use one block per patch. Choose a tile size (number of threads per block) for processing the embedding vector.
    int tile_size = 32;  // For example, process 32 elements at a time
    dim3 blockDim(tile_size);
    dim3 gridDim(NUM_PATCHES);  // one block per patch
    size_t sharedMemSize = tile_size * sizeof(float);
    cudaCheck(cudaEventRecord(start_kernel, 0));
    add_positional_embeddings_tiled<<<gridDim, blockDim, sharedMemSize>>>(d_embeddings, NUM_PATCHES, EMBEDDING_DIM);
    cudaCheck(cudaEventRecord(stop_kernel, 0));
    cudaCheck(cudaEventSynchronize(stop_kernel));
    float time_kernel;
    cudaCheck(cudaEventElapsedTime(&time_kernel, start_kernel, stop_kernel));

    // Copy the results back from GPU to CPU
    cudaCheck(cudaEventRecord(start_D2H, 0));
    cudaCheck(cudaMemcpy(h_embeddings_gpu, d_embeddings, total_embeddings * sizeof(float), cudaMemcpyDeviceToHost));
    cudaCheck(cudaEventRecord(stop_D2H, 0));
    cudaCheck(cudaEventSynchronize(stop_D2H));
    float time_D2H;
    cudaCheck(cudaEventElapsedTime(&time_D2H, start_D2H, stop_D2H));

    // Record overall GPU stop time
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
    delete[] h_embeddings_cpu;
    delete[] h_embeddings_gpu;
    cudaFree(d_embeddings);
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
