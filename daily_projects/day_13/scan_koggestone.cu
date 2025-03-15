#include <stdio.h>
#include <cuda.h>

#define BLOCK_DIM 1024

// Baseline GPU Scan Kernel
__global__ void baseline_scan(float *input, float *output, unsigned int N) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    output[i] = input[i];
    __syncthreads();

    for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
        float v = 0;
        if (threadIdx.x >= stride)
            v = output[i - stride];
        __syncthreads();
        if (threadIdx.x >= stride)
            output[i] += v;
        __syncthreads();
    }
}

// Shared Memory Single Buffer Scan Kernel
__global__ void shared_single_buffer_scan(float *input, float *output, unsigned int N) {
    __shared__ float buffer[BLOCK_DIM];
    int tid = threadIdx.x;
    buffer[tid] = input[blockIdx.x * blockDim.x + tid];
    __syncthreads();

    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        float val = 0;
        if (tid >= stride)
            val = buffer[tid - stride];
        __syncthreads();
        if (tid >= stride)
            buffer[tid] += val;
        __syncthreads();
    }

    output[blockIdx.x * blockDim.x + tid] = buffer[tid];
}

// Shared Memory Double Buffer Scan Kernel
__global__ void shared_double_buffer_scan(float *input, float *output, unsigned int N) {
    __shared__ float buffer[2][BLOCK_DIM];
    int tid = threadIdx.x;
    buffer[0][tid] = input[blockIdx.x * blockDim.x + tid];
    __syncthreads();

    int ping = 0, pong = 1;
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        buffer[pong][tid] = buffer[ping][tid];
        if (tid >= stride)
            buffer[pong][tid] += buffer[ping][tid - stride];
        __syncthreads();
        int tmp = ping; ping = pong; pong = tmp;
    }

    output[blockIdx.x * blockDim.x + tid] = buffer[ping][tid];
}

int main() {
    const unsigned int N = 1024;
    size_t size = N * sizeof(float);

    unsigned int numThreadsPerBlock = BLOCK_DIM;
    unsigned int numElementsPerBlock = numThreadsPerBlock;
    unsigned int numBlocks = (N + numElementsPerBlock - 1) / numElementsPerBlock;

    float h_input[N];
    for (unsigned int i = 0; i < N; i++) h_input[i] = 1.0f;

    float *d_input, *d_output;
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);

    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

    float scenarios_kernel_time[3];
    const char *scenario_names[3] = {"Baseline", "Shared Single Buffer", "Shared Double Buffer"};

    void (*kernels[3])(float*, float*, unsigned int) = {baseline_scan, shared_single_buffer_scan, shared_double_buffer_scan};

    for (int k = 0; k < 3; k++) {
        cudaEvent_t start, stop;
        cudaEventCreate(&start); cudaEventCreate(&stop);

        cudaEventRecord(start);
        kernels[k]<<<numBlocks, numThreadsPerBlock>>>(d_input, d_output, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        cudaEventElapsedTime(&scenarios_kernel_time[k], start, stop);

        printf("%s GPU Kernel Time: %.3f ms\n", scenario_names[k], scenarios_kernel_time[k]);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    cudaFree(d_input); cudaFree(d_output);
    return 0;
}
