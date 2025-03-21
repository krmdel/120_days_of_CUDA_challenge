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

__global__ void spmv_ell_kernel(int num_rows, int max_nnz_per_row, 
                                const int *ellCol, const float *ellVal, 
                                const float *x, float *y) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < num_rows) {
        float dot = 0.0f;
        // Loop over the fixed number of nonzeros per row
        for (int j = 0; j < max_nnz_per_row; j++) {
            int idx = row * max_nnz_per_row + j;
            dot += ellVal[idx] * x[ellCol[idx]];
        }
        y[row] = dot;
    }
}

__global__ void spmv_jds_kernel(int num_rows, int num_diags, 
                                const int *jdsDiagPtr, const int *jdsDiagLengths, 
                                const int *jdsCol, const float *jdsVal, 
                                const int *jdsPerm, const float *x, float *y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_rows) {
        float dot = 0.0f;
        for (int d = 0; d < num_diags; d++) {
            if (i < jdsDiagLengths[d]) {
                int idx = jdsDiagPtr[d] + i;
                dot += jdsVal[idx] * x[jdsCol[idx]];
            }
        }
        y[jdsPerm[i]] = dot;
    }
}

int main() {
   
    const int num_rows = 4;
    const int num_cols = 4;
    const int max_nnz_per_row = 2;

    int h_ellCol[] = { 0, 1,   1, 0,   2, 0,   0, 3 };
    float h_ellVal[] = { 10.0f, 20.0f,   30.0f, 0.0f,   40.0f, 0.0f,   50.0f, 60.0f };

    int h_jdsPerm[] = { 0, 3, 1, 2 };
    int h_jdsDiagPtr[] = { 0, 4 }; 
    int h_jdsDiagLengths[] = { 4, 2 };
    float h_jdsVal[] = { 10.0f, 50.0f, 30.0f, 40.0f, 20.0f, 60.0f };
    int h_jdsCol[] = { 0, 0, 1, 2, 1, 3 };

    float h_x[] = { 1.0f, 2.0f, 3.0f, 4.0f };
    float h_y_ell[4] = { 0 };
    float h_y_jds[4] = { 0 };

    cudaEvent_t start, stop;
    float elapsedTime = 0.0f;

    int *d_ellCol;
    float *d_ellVal, *d_x, *d_y;
    CUDA_CHECK(cudaMalloc((void**)&d_ellCol, num_rows * max_nnz_per_row * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_ellVal, num_rows * max_nnz_per_row * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_x, num_cols * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_y, num_rows * sizeof(float)));

    CUDA_CHECK(cudaMemset(d_y, 0, num_rows * sizeof(float)));

    // Copy data from CPU to GPU
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    cudaEventRecord(start, 0);
    CUDA_CHECK(cudaMemcpy(d_ellCol, h_ellCol, num_rows * max_nnz_per_row * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ellVal, h_ellVal, num_rows * max_nnz_per_row * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x, h_x, num_cols * sizeof(float), cudaMemcpyHostToDevice));
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("ELL copy time: %f ms\n", elapsedTime);
    float ellHtoD = elapsedTime;

    // Launch ELL kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (num_rows + threadsPerBlock - 1) / threadsPerBlock;
    cudaEventRecord(start, 0);
    spmv_ell_kernel<<<blocksPerGrid, threadsPerBlock>>>(num_rows, max_nnz_per_row, d_ellCol, d_ellVal, d_x, d_y);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("ELL kernel time: %f ms\n", elapsedTime);
    float ellKernel = elapsedTime;

    // Copy data back from GPU to CPU
    cudaEventRecord(start, 0);
    CUDA_CHECK(cudaMemcpy(h_y_ell, d_y, num_rows * sizeof(float), cudaMemcpyDeviceToHost));
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("ELL copy back time: %f ms\n", elapsedTime);
    float ellDtoH = elapsedTime;

    float ellTotal = ellHtoD + ellKernel + ellDtoH;
    printf("ELL total time: %f ms\n\n", ellTotal);

    CUDA_CHECK(cudaFree(d_ellCol));
    CUDA_CHECK(cudaFree(d_ellVal));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));

    int *d_jdsDiagPtr, *d_jdsDiagLengths, *d_jdsCol, *d_jdsPerm;
    float *d_jdsVal, *d_x2, *d_y2;

    CUDA_CHECK(cudaMalloc((void**)&d_jdsDiagPtr, 2 * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_jdsDiagLengths, 2 * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_jdsCol, 6 * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_jdsVal, 6 * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_jdsPerm, num_rows * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_x2, num_cols * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_y2, num_rows * sizeof(float)));

    CUDA_CHECK(cudaMemset(d_y2, 0, num_rows * sizeof(float)));

    // Copy data from CPU to GPU for JDS
    cudaEventRecord(start, 0);
    CUDA_CHECK(cudaMemcpy(d_jdsDiagPtr, h_jdsDiagPtr, 2 * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_jdsDiagLengths, h_jdsDiagLengths, 2 * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_jdsCol, h_jdsCol, 6 * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_jdsVal, h_jdsVal, 6 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_jdsPerm, h_jdsPerm, num_rows * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x2, h_x, num_cols * sizeof(float), cudaMemcpyHostToDevice));
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("JDS copy time: %f ms\n", elapsedTime);
    float jdsHtoD = elapsedTime;

    // Launch JDS kernel
    threadsPerBlock = 256;
    blocksPerGrid = (num_rows + threadsPerBlock - 1) / threadsPerBlock;
    cudaEventRecord(start, 0);
    spmv_jds_kernel<<<blocksPerGrid, threadsPerBlock>>>(num_rows, 2, d_jdsDiagPtr, d_jdsDiagLengths,
                                                         d_jdsCol, d_jdsVal, d_jdsPerm, d_x2, d_y2);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("JDS kernel time: %f ms\n", elapsedTime);
    float jdsKernel = elapsedTime;

    // Copy data back from GPU to CPU
    cudaEventRecord(start, 0);
    CUDA_CHECK(cudaMemcpy(h_y_jds, d_y2, num_rows * sizeof(float), cudaMemcpyDeviceToHost));
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("JDS copy back time: %f ms\n", elapsedTime);
    float jdsDtoH = elapsedTime;

    float jdsTotal = jdsHtoD + jdsKernel + jdsDtoH;
    printf("JDS total time: %f ms\n\n", jdsTotal);

    CUDA_CHECK(cudaFree(d_jdsDiagPtr));
    CUDA_CHECK(cudaFree(d_jdsDiagLengths));
    CUDA_CHECK(cudaFree(d_jdsCol));
    CUDA_CHECK(cudaFree(d_jdsVal));
    CUDA_CHECK(cudaFree(d_jdsPerm));
    CUDA_CHECK(cudaFree(d_x2));
    CUDA_CHECK(cudaFree(d_y2));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return 0;
}
