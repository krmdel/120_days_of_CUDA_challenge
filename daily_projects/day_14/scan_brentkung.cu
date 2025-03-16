#include <stdio.h>
#include <cuda.h>

#define BLOCK_DIM 1024

__global__ void brent_kung_scan(float *input, float *output, unsigned int N)
{
    __shared__ float temp[BLOCK_DIM];
    
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    float val = 0.0f;
    if (i < N) {
        val = input[i];
    }
    temp[threadIdx.x] = val;
    __syncthreads();

    for (int offset = 1; offset < blockDim.x; offset <<= 1) {
        int index = (threadIdx.x + 1) * (offset << 1) - 1;
        if (index < blockDim.x) {
            temp[index] += temp[index - offset];
        }
        __syncthreads();
    }

    if (threadIdx.x == blockDim.x - 1) {
        temp[threadIdx.x] = 0.0f;
    }
    __syncthreads();

    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        int index = (threadIdx.x + 1) * (offset << 1) - 1;
        if (index + offset < blockDim.x) {
            float t = temp[index];
            temp[index]        += temp[index + offset];
            temp[index + offset] = t;
        }
        __syncthreads();
    }

    float scanVal = temp[threadIdx.x] + val;
    __syncthreads();
    temp[threadIdx.x] = scanVal;
    __syncthreads();

    if (i < N) {
        output[i] = temp[threadIdx.x];
    }
}

int main()
{
    const unsigned int N = 1024;
    size_t size = N * sizeof(float);

    unsigned int numThreadsPerBlock   = BLOCK_DIM;
    unsigned int numElementsPerBlock  = numThreadsPerBlock;
    unsigned int numBlocks = (N + numElementsPerBlock - 1) / numElementsPerBlock;

    float h_input[N];
    for (unsigned int i = 0; i < N; i++) {
        h_input[i] = 1.0f;  // Fill with ones for testing
    }

    // Allocate device memory
    float *d_input, *d_output;
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);

    // Copy from CPU to GPU
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Perform Brent-Kung Scan
    cudaEventRecord(start);
    brent_kung_scan<<<numBlocks, numThreadsPerBlock>>>(d_input, d_output, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float elapsed;
    cudaEventElapsedTime(&elapsed, start, stop);
    printf("Brent-Kung Scan Kernel Time: %.3f ms\n", elapsed);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Copy from GPU to CPU
    float h_output[N];
    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
    return 0;
}
