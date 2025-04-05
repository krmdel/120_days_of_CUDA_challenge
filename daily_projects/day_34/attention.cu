#include <iostream>
#include <cmath>
#include <chrono>
#include <cuda_runtime.h>

#define TILE_WIDTH 16

// CUDA error checking macro.
#define CHECK_CUDA(call) {                                              \
    cudaError_t err = call;                                             \
    if(err != cudaSuccess) {                                            \
        std::cerr << "CUDA error: " << cudaGetErrorString(err)          \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl;  \
        exit(1);                                                        \
    }                                                                   \
}

// Kernel to compute scaled dot-product attention between Q and K matrices using shared memory
// C[i,j] = (1/sqrt(d)) * sum_{k=0}^{d-1} Q[i,k] * K[j,k]
__global__ void scaled_dot_product_kernel(const float *Q, const float *K, float *C, int M, int N, int d) {
    // Allocate shared memory for tiles of Q and K
    __shared__ float ds_Q[TILE_WIDTH][TILE_WIDTH];
    __shared__ float ds_K[TILE_WIDTH][TILE_WIDTH];

    // Calculate row index for Q and column index for K (which becomes row index in K since we compute K^T)
    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;

    float value = 0.0f;

    // Loop over the tiles along the shared dimension
    for (int t = 0; t < (d + TILE_WIDTH - 1) / TILE_WIDTH; t++) {
        // Load tile from Q
        if (row < M && (t * TILE_WIDTH + threadIdx.x) < d)
            ds_Q[threadIdx.y][threadIdx.x] = Q[row * d + t * TILE_WIDTH + threadIdx.x];
        else
            ds_Q[threadIdx.y][threadIdx.x] = 0.0f;

        // Load tile from K
        if (col < N && (t * TILE_WIDTH + threadIdx.y) < d)
            ds_K[threadIdx.y][threadIdx.x] = K[col * d + t * TILE_WIDTH + threadIdx.y];
        else
            ds_K[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        // Perform multiplication on the tile
        for (int i = 0; i < TILE_WIDTH; i++) {
            value += ds_Q[threadIdx.y][i] * ds_K[i][threadIdx.x];
        }
        __syncthreads();
    }

    // Output matrix, scaling by 1/sqrt(d)
    if (row < M && col < N) {
        C[row * N + col] = value / sqrtf((float)d);
    }
}

// Scaled dot-product attention on CPU
void cpu_scaled_dot_product(const float *Q, const float *K, float *C, int M, int N, int d) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < d; k++) {
                sum += Q[i * d + k] * K[j * d + k];
            }
            C[i * N + j] = sum / sqrtf((float)d);
        }
    }
}

int main() {
    const int M = 1000; // number of rows in Q
    const int N = 1000; // number of rows in K (i.e. number of columns in QK^T)
    const int d = 512; // feature dimension (d_k)

    // Allocate host memory for Q, K, and output C
    float *h_Q = new float[M * d];
    float *h_K = new float[N * d];
    float *h_C = new float[M * N];
    float *cpu_C = new float[M * N];

    // Initialize Q and K with sample values
    for (int i = 0; i < M * d; i++) {
        h_Q[i] = static_cast<float>(i + 1);
    }
    for (int i = 0; i < N * d; i++) {
        h_K[i] = static_cast<float>(i + 1);
    }

    // Attention computation on CPU
    auto cpu_start = std::chrono::high_resolution_clock::now();
    cpu_scaled_dot_product(h_Q, h_K, cpu_C, M, N, d);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    double cpu_time = std::chrono::duration_cast<std::chrono::milliseconds>(cpu_end - cpu_start).count();

    // Attention computation on GPU
    cudaEvent_t start_total, stop_total, start_h2d, stop_h2d, start_kernel, stop_kernel, start_d2h, stop_d2h;
    CHECK_CUDA(cudaEventCreate(&start_total));
    CHECK_CUDA(cudaEventCreate(&stop_total));
    CHECK_CUDA(cudaEventCreate(&start_h2d));
    CHECK_CUDA(cudaEventCreate(&stop_h2d));
    CHECK_CUDA(cudaEventCreate(&start_kernel));
    CHECK_CUDA(cudaEventCreate(&stop_kernel));
    CHECK_CUDA(cudaEventCreate(&start_d2h));
    CHECK_CUDA(cudaEventCreate(&stop_d2h));

    // Allocate device memory
    float *d_Q, *d_K, *d_C;
    CHECK_CUDA(cudaMalloc(&d_Q, M * d * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_K, N * d * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_C, M * N * sizeof(float)));

    CHECK_CUDA(cudaEventRecord(start_total, 0));

    // Host-to-Device Copy Timing
    CHECK_CUDA(cudaEventRecord(start_h2d, 0));
    CHECK_CUDA(cudaMemcpy(d_Q, h_Q, M * d * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_K, h_K, N * d * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaEventRecord(stop_h2d, 0));
    CHECK_CUDA(cudaEventSynchronize(stop_h2d));
    float h2d_time;
    CHECK_CUDA(cudaEventElapsedTime(&h2d_time, start_h2d, stop_h2d));

    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid((N + TILE_WIDTH - 1) / TILE_WIDTH, (M + TILE_WIDTH - 1) / TILE_WIDTH);
    
    CHECK_CUDA(cudaEventRecord(start_kernel, 0));
    scaled_dot_product_kernel<<<dimGrid, dimBlock>>>(d_Q, d_K, d_C, M, N, d);
    CHECK_CUDA(cudaEventRecord(stop_kernel, 0));
    CHECK_CUDA(cudaEventSynchronize(stop_kernel));
    float kernel_time;
    CHECK_CUDA(cudaEventElapsedTime(&kernel_time, start_kernel, stop_kernel));

    // Device-to-Host Copy Timing
    CHECK_CUDA(cudaEventRecord(start_d2h, 0));
    CHECK_CUDA(cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaEventRecord(stop_d2h, 0));
    CHECK_CUDA(cudaEventSynchronize(stop_d2h));
    float d2h_time;
    CHECK_CUDA(cudaEventElapsedTime(&d2h_time, start_d2h, stop_d2h));

    CHECK_CUDA(cudaEventRecord(stop_total, 0));
    CHECK_CUDA(cudaEventSynchronize(stop_total));
    float total_gpu_time;
    CHECK_CUDA(cudaEventElapsedTime(&total_gpu_time, start_total, stop_total));

    std::cout << "CPU Inference Time: " << cpu_time << " ms\n";
    std::cout << "GPU Host-to-Device Copy Time: " << h2d_time << " ms\n";
    std::cout << "GPU Kernel Execution Time: " << kernel_time << " ms\n";
    std::cout << "GPU Device-to-Host Copy Time: " << d2h_time << " ms\n";
    std::cout << "Total GPU Inference Time: " << total_gpu_time << " ms\n";

    // Cleanup
    delete[] h_Q;
    delete[] h_K;
    delete[] h_C;
    delete[] cpu_C;
    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_C);

    // Destroy CUDA events
    cudaEventDestroy(start_total);
    cudaEventDestroy(stop_total);
    cudaEventDestroy(start_h2d);
    cudaEventDestroy(stop_h2d);
    cudaEventDestroy(start_kernel);
    cudaEventDestroy(stop_kernel);
    cudaEventDestroy(start_d2h);
    cudaEventDestroy(stop_d2h);

    return 0;
}
