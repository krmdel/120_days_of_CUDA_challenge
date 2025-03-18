#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>
#include <chrono>

__device__ void co_rank(const int k, const int *A, const int m,
                          const int *B, const int n,
                          int &i, int &j)
{
    int low = (k - n > 0) ? k - n : 0;
    int high = (k < m) ? k : m;
    while (low < high) {
        int mid = (low + high) / 2;
        int bj = k - mid; // since k = mid + bj
        if (mid > 0 && bj < n && A[mid - 1] > B[bj]) {
            high = mid;
        } else if (bj > 0 && mid < m && B[bj - 1] > A[mid]) {
            low = mid + 1;
        } else {
            low = mid;
            break;
        }
    }
    i = low;
    j = k - low;
}

__global__ void mergeKernel_baseline(const int *A, int m,
                                     const int *B, int n,
                                     int *C)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = m + n;
    int numThreads = gridDim.x * blockDim.x;
    int perThread = (total + numThreads - 1) / numThreads;
    
    int k_start = tid * perThread;
    int k_end   = (k_start + perThread < total) ? k_start + perThread : total;
    
    int i_start, j_start;
    co_rank(k_start, A, m, B, n, i_start, j_start);
    int i_end, j_end;
    co_rank(k_end, A, m, B, n, i_end, j_end);
    
    int i = i_start;
    int j = j_start;
    for (int k = k_start; k < k_end; k++) {
        if (i < i_end && (j >= j_end || A[i] <= B[j])) {
            C[k] = A[i++];
        } else {
            C[k] = B[j++];
        }
    }
}

__global__ void mergeKernel_optimized(const int *A, int m,
                                      const int *B, int n,
                                      int *C)
{
    int total = m + n;
    int threadsPerBlock = blockDim.x;
    int blockId = blockIdx.x;
    int numBlocks = gridDim.x;
    
    int block_output_start = (total * blockId) / numBlocks;
    int block_output_end   = (total * (blockId + 1)) / numBlocks;
    
    __shared__ int A_start_index, B_start_index, A_end_index, B_end_index;
    if (threadIdx.x == 0) {
        co_rank(block_output_start, A, m, B, n, A_start_index, B_start_index);
        co_rank(block_output_end, A, m, B, n, A_end_index, B_end_index);
    }
    __syncthreads();
    
    int sizeA = A_end_index - A_start_index;
    int sizeB = B_end_index - B_start_index;
    
    // Use dynamic shared memory:
    // s_data holds both sA and sB.
    extern __shared__ int s_data[];
    int *sA = s_data;
    int *sB = s_data + sizeA;
    
    // Load A's segment into shared memory
    for (int i = threadIdx.x; i < sizeA; i += threadsPerBlock) {
        sA[i] = A[A_start_index + i];
    }
    // Load B's segment into shared memory
    for (int j = threadIdx.x; j < sizeB; j += threadsPerBlock) {
        sB[j] = B[B_start_index + j];
    }
    __syncthreads();
    
    int block_total = sizeA + sizeB;
    int perThread = (block_total + threadsPerBlock - 1) / threadsPerBlock;
    int local_start = threadIdx.x * perThread;
    int local_end   = (local_start + perThread < block_total) ? local_start + perThread : block_total;
    
    int local_i_start, local_j_start;
    co_rank(local_start, sA, sizeA, sB, sizeB, local_i_start, local_j_start);
    int local_i_end, local_j_end;
    co_rank(local_end, sA, sizeA, sB, sizeB, local_i_end, local_j_end);
    
    int i = local_i_start;
    int j = local_j_start;
    int global_offset = block_output_start;
    
    for (int k = local_start; k < local_end; k++) {
        if (i < local_i_end && (j >= local_j_end || sA[i] <= sB[j])) {
            C[global_offset + k] = sA[i++];
        } else {
            C[global_offset + k] = sB[j++];
        }
    }
}

int main() {
    const int m = 10;
    const int n = 10;
    const int total = m + n;
    int h_A[m] = {1, 3, 5, 7, 9, 11, 13, 15, 17, 19};
    int h_B[n] = {2, 4, 6, 8, 10, 12, 14, 16, 18, 20};
    
    // Allocate memory for results on CPU
    int *h_C = new int[total];
    int *h_C_cpu = new int[total];
    
    // Allocate device memory
    int *d_A = nullptr;
    int *d_B = nullptr;
    int *d_C = nullptr;
    cudaMalloc(&d_A, m * sizeof(int));
    cudaMalloc(&d_B, n * sizeof(int));
    cudaMalloc(&d_C, total * sizeof(int));
    
      std::cout << "==== Baseline Kernel ====\n";
    cudaEvent_t start, stop;
    float timeH2D = 0, timeKernel = 0, timeD2H = 0;
    
    // Copy input arrays from CPU to GPU
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    cudaMemcpy(d_A, h_A, m * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&timeH2D, start, stop);
    
    // Launch kernel
    int threadsPerBlock = 256;
    int numBlocks = (total + threadsPerBlock - 1) / threadsPerBlock;
    cudaEventRecord(start);
    mergeKernel_baseline<<<numBlocks, threadsPerBlock>>>(d_A, m, d_B, n, d_C);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&timeKernel, start, stop);
    
    // Copy result back from GPU to CPU
    cudaEventRecord(start);
    cudaMemcpy(h_C, d_C, total * sizeof(int), cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&timeD2H, start, stop);
    
    float totalGPUTime = timeH2D + timeKernel + timeD2H;
    std::cout << "CPU->GPU Copy Time: " << timeH2D << " ms\n";
    std::cout << "Kernel Execution Time: " << timeKernel << " ms\n";
    std::cout << "GPU->CPU Copy Time: " << timeD2H << " ms\n";
    std::cout << "Total GPU Baseline Merge Time: " << totalGPUTime << " ms\n";
    
    std::cout << "==== Optimized Kernel ====\n";
    int threadsPerBlockOpt = 256;
    int numBlocksOpt = 1; 
    size_t sharedMemSize = (m + n) * sizeof(int);
    
    // Copy input arrays
    cudaEventRecord(start);
    cudaMemcpy(d_A, h_A, m * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float timeH2D_opt = 0;
    cudaEventElapsedTime(&timeH2D_opt, start, stop);
    
    // Launch optimized kernel
    cudaEventRecord(start);
    mergeKernel_optimized<<<numBlocksOpt, threadsPerBlockOpt, sharedMemSize>>>(d_A, m, d_B, n, d_C);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float timeKernel_opt = 0;
    cudaEventElapsedTime(&timeKernel_opt, start, stop);
    
    // Copy result back from GPU
    cudaEventRecord(start);
    cudaMemcpy(h_C, d_C, total * sizeof(int), cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float timeD2H_opt = 0;
    cudaEventElapsedTime(&timeD2H_opt, start, stop);
    
    float totalGPUTime_opt = timeH2D_opt + timeKernel_opt + timeD2H_opt;
    std::cout << "CPU->GPU Copy Time: " << timeH2D_opt << " ms\n";
    std::cout << "Kernel Execution Time: " << timeKernel_opt << " ms\n";
    std::cout << "GPU->CPU Copy Time: " << timeD2H_opt << " ms\n";
    std::cout << "Total GPU Optimized Merge Time: " << totalGPUTime_opt << " ms\n";
            
    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    delete[] h_C;
    delete[] h_C_cpu;
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}
