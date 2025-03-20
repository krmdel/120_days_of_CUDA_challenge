#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// Error checking macro
#define CUDA_CHECK(call) {                                               \
    cudaError_t err = call;                                              \
    if(err != cudaSuccess) {                                             \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,  \
                cudaGetErrorString(err));                                \
        exit(err);                                                       \
    }                                                                    \
}

__global__ void spmv_coo_kernel(int nnz, const int *cooRow, const int *cooCol, 
                                const float *cooVal, const float *x, float *y) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nnz) {
        // Each thread computes its product and atomically adds to the correct output element.
        atomicAdd(&y[cooRow[idx]], cooVal[idx] * x[cooCol[idx]]);
    }
}

__global__ void spmv_csr_kernel(int num_rows, const int *csrRowPtr, const int *csrCol, 
                                const float *csrVal, const float *x, float *y) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < num_rows) {
        float dot = 0.0f;
        int row_start = csrRowPtr[row];
        int row_end   = csrRowPtr[row+1];
        for (int i = row_start; i < row_end; i++) {
            dot += csrVal[i] * x[csrCol[i]];
        }
        y[row] = dot;
    }
}

int main() {
    const int num_rows = 4;
    const int num_cols = 4;
    const int nnz = 6;

    int h_cooRow[] = {0, 0, 1, 2, 3, 3};
    int h_cooCol[] = {0, 1, 1, 2, 0, 3};
    float h_cooVal[] = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f};

    int h_csrRowPtr[] = {0, 2, 3, 4, 6}; // row i goes from h_csrRowPtr[i] to h_csrRowPtr[i+1]-1
    int h_csrCol[] = {0, 1, 1, 2, 0, 3};
    float h_csrVal[] = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f};

    float h_x[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float h_y_coo[4] = {0};
    float h_y_csr[4] = {0};

    cudaEvent_t start, stop;
    float elapsedTime = 0.0f;

    int *d_cooRow, *d_cooCol;
    float *d_cooVal, *d_x, *d_y;
    CUDA_CHECK(cudaMalloc((void**)&d_cooRow, nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_cooCol, nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_cooVal, nnz * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_x, num_cols * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_y, num_rows * sizeof(float)));

    CUDA_CHECK(cudaMemset(d_y, 0, num_rows * sizeof(float)));

    // Copy data from CPU to GPU
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    cudaEventRecord(start, 0);
    CUDA_CHECK(cudaMemcpy(d_cooRow, h_cooRow, nnz * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cooCol, h_cooCol, nnz * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cooVal, h_cooVal, nnz * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x, h_x, num_cols * sizeof(float), cudaMemcpyHostToDevice));
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("COO copy time: %f ms\n", elapsedTime);
    float cooHtoD = elapsedTime;

    // Launch COO kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (nnz + threadsPerBlock - 1) / threadsPerBlock;
    cudaEventRecord(start, 0);
    spmv_coo_kernel<<<blocksPerGrid, threadsPerBlock>>>(nnz, d_cooRow, d_cooCol, d_cooVal, d_x, d_y);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("COO kernel time: %f ms\n", elapsedTime);
    float cooKernel = elapsedTime;

    // Copy data back to CPU
    cudaEventRecord(start, 0);
    CUDA_CHECK(cudaMemcpy(h_y_coo, d_y, num_rows * sizeof(float), cudaMemcpyDeviceToHost));
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("COO copy time: %f ms\n", elapsedTime);
    float cooDtoH = elapsedTime;

    float cooTotal = cooHtoD + cooKernel + cooDtoH;
    printf("COO total time: %f ms\n\n", cooTotal);

    CUDA_CHECK(cudaFree(d_cooRow));
    CUDA_CHECK(cudaFree(d_cooCol));
    CUDA_CHECK(cudaFree(d_cooVal));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));

    int *d_csrRowPtr, *d_csrCol;
    float *d_csrVal, *d_x2, *d_y2;
    CUDA_CHECK(cudaMalloc((void**)&d_csrRowPtr, (num_rows + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_csrCol, nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_csrVal, nnz * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_x2, num_cols * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_y2, num_rows * sizeof(float)));

    CUDA_CHECK(cudaMemset(d_y2, 0, num_rows * sizeof(float)));

    // Copy data from CPU to GPU
    cudaEventRecord(start, 0);
    CUDA_CHECK(cudaMemcpy(d_csrRowPtr, h_csrRowPtr, (num_rows + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_csrCol, h_csrCol, nnz * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_csrVal, h_csrVal, nnz * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x2, h_x, num_cols * sizeof(float), cudaMemcpyHostToDevice));
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("CSR copy time: %f ms\n", elapsedTime);
    float csrHtoD = elapsedTime;

    // Launch CSR kernel
    threadsPerBlock = 256;
    blocksPerGrid = (num_rows + threadsPerBlock - 1) / threadsPerBlock;
    cudaEventRecord(start, 0);
    spmv_csr_kernel<<<blocksPerGrid, threadsPerBlock>>>(num_rows, d_csrRowPtr, d_csrCol, d_csrVal, d_x2, d_y2);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("CSR kernel time: %f ms\n", elapsedTime);
    float csrKernel = elapsedTime;

    // Copy data back to CPU
    cudaEventRecord(start, 0);
    CUDA_CHECK(cudaMemcpy(h_y_csr, d_y2, num_rows * sizeof(float), cudaMemcpyDeviceToHost));
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("CSR copy time: %f ms\n", elapsedTime);
    float csrDtoH = elapsedTime;

    float csrTotal = csrHtoD + csrKernel + csrDtoH;
    printf("CSR total time: %f ms\n\n", csrTotal);

    // Clean up
    CUDA_CHECK(cudaFree(d_csrRowPtr));
    CUDA_CHECK(cudaFree(d_csrCol));
    CUDA_CHECK(cudaFree(d_csrVal));
    CUDA_CHECK(cudaFree(d_x2));
    CUDA_CHECK(cudaFree(d_y2));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return 0;
}
