#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define N 128
#define RADIUS 1
#define BLOCK_DIM 8
#define IN_TILE_DIM BLOCK_DIM
#define OUT_TILE_DIM (IN_TILE_DIM - 2)

void cpuStencil3D(const float* input, float* output, int n) {
    for (int z = RADIUS; z < n - RADIUS; ++z)
        for (int y = RADIUS; y < n - RADIUS; ++y)
            for (int x = RADIUS; x < n - RADIUS; ++x) {
                float sum = 0;
                for (int dz = -RADIUS; dz <= RADIUS; ++dz)
                    for (int dy = -RADIUS; dy <= RADIUS; ++dy)
                        for (int dx = -RADIUS; dx <= RADIUS; ++dx)
                            sum += input[(z + dz)*n*n + (y + dy)*n + (x + dx)];
                output[z*n*n + y*n + x] = sum;
            }
}

__global__ void gpuStencil3D(const float* input, float* output, int n) {
    int x = blockIdx.x * blockDim.x + threadIdx.x + RADIUS;
    int y = blockIdx.y * blockDim.y + threadIdx.y + RADIUS;
    int z = blockIdx.z * blockDim.z + threadIdx.z + RADIUS;

    if (x < n - RADIUS && y < n - RADIUS && z < n - RADIUS) {
        float sum = 0;
        for (int dz = -RADIUS; dz <= RADIUS; ++dz)
            for (int dy = -RADIUS; dy <= RADIUS; ++dy)
                for (int dx = -RADIUS; dx <= RADIUS; ++dx)
                    sum += input[(z + dz)*n*n + (y + dy)*n + (x + dx)];
        output[z*n*n + y*n + x] = sum;
    }
}

__global__ void gpuStencilTiled3D(const float* input, float* output, int n) {
    __shared__ float tile[IN_TILE_DIM][IN_TILE_DIM][IN_TILE_DIM];

    int x = blockIdx.x * OUT_TILE_DIM + threadIdx.x - RADIUS;
    int y = blockIdx.y * OUT_TILE_DIM + threadIdx.y - RADIUS;
    int z = blockIdx.z * OUT_TILE_DIM + threadIdx.z - RADIUS;

    if (x >= 0 && x < n && y >= 0 && y < n && z >= 0 && z < n)
        tile[threadIdx.z][threadIdx.y][threadIdx.x] = input[z*n*n + y*n + x];

    __syncthreads();

    if (threadIdx.x >= RADIUS && threadIdx.x < IN_TILE_DIM - RADIUS &&
        threadIdx.y >= RADIUS && threadIdx.y < IN_TILE_DIM - RADIUS &&
        threadIdx.z >= RADIUS && threadIdx.z < IN_TILE_DIM - RADIUS &&
        x < n - RADIUS && y < n - RADIUS && z < n - RADIUS) {
        float sum = 0;
        for (int dz = -RADIUS; dz <= RADIUS; ++dz)
            for (int dy = -RADIUS; dy <= RADIUS; ++dy)
                for (int dx = -RADIUS; dx <= RADIUS; ++dx)
                    sum += tile[threadIdx.z + dz][threadIdx.y + dy][threadIdx.x + dx];
        output[z*n*n + y*n + x] = sum;
    }
}

__global__ void gpuStencilCoarse3D(const float* input, float* output, int n) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 2 + RADIUS;
    int y = blockIdx.y * blockDim.y + threadIdx.y + RADIUS;
    int z = blockIdx.z * blockDim.z + threadIdx.z + RADIUS;

    for (int j = 0; j < 2; ++j) {
        int x = idx + j;
        if (x < n - RADIUS && y < n - RADIUS && z < n - RADIUS) {
            float sum = 0;
            for (int dz = -RADIUS; dz <= RADIUS; ++dz)
                for (int dy = -RADIUS; dy <= RADIUS; ++dy)
                    for (int dx = -RADIUS; dx <= RADIUS; ++dx)
                        sum += input[(z + dz)*n*n + (y + dy)*n + (x + dx)];
            output[z*n*n + y*n + x] = sum;
        }
    }
}

int main() {
    size_t size = N * N * N * sizeof(float);
    float *h_input, *h_output;
    cudaMallocHost(&h_input, size);
    cudaMallocHost(&h_output, size);

    for (int i = 0; i < N * N * N; ++i)
        h_input[i] = 1.0f;

    float *d_input, *d_output;
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);

    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

    dim3 blockDim(BLOCK_DIM, BLOCK_DIM, BLOCK_DIM);
    dim3 gridDim((N + BLOCK_DIM - 1) / BLOCK_DIM,
                 (N + BLOCK_DIM - 1) / BLOCK_DIM,
                 (N + BLOCK_DIM - 1) / BLOCK_DIM);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // CPU stencil
    cudaEventRecord(start);
    cpuStencil3D(h_input, h_output, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    printf("CPU stencil 3D: %f ms\n", ms);

    // GPU stencil
    cudaEventRecord(start);
    gpuStencil3D<<<gridDim, blockDim>>>(d_input, d_output, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf("GPU stencil 3D: %f ms\n", ms);

    // GPU stencil with tiling
    dim3 blockDimTiled(IN_TILE_DIM, IN_TILE_DIM, IN_TILE_DIM);
    dim3 gridDimTiled((N + OUT_TILE_DIM - 1) / OUT_TILE_DIM,
                      (N + OUT_TILE_DIM - 1) / OUT_TILE_DIM,
                      (N + OUT_TILE_DIM - 1) / OUT_TILE_DIM);

    cudaEventRecord(start);
    gpuStencilTiled3D<<<gridDimTiled, blockDimTiled>>>(d_input, d_output, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf("GPU tiled stencil 3D: %f ms\n", ms);

    // GPU stencil with thread coarsening
    dim3 blockDimCoarse(BLOCK_DIM / 2, BLOCK_DIM, BLOCK_DIM);
    dim3 gridDimCoarse((N / 2 + BLOCK_DIM / 2 - 1) / (BLOCK_DIM / 2),
                       (N + BLOCK_DIM - 1) / BLOCK_DIM,
                       (N + BLOCK_DIM - 1) / BLOCK_DIM);

    cudaEventRecord(start);
    gpuStencilCoarse3D<<<gridDimCoarse, blockDimCoarse>>>(d_input, d_output, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf("GPU coarse stencil 3D: %f ms\n", ms);

    cudaFreeHost(h_input);
    cudaFreeHost(h_output);
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
