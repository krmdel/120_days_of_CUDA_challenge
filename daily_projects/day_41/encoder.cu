#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <chrono>

// Error-checking macro
#define CHECK_CUDA(call) {                                                         \
    cudaError_t err = call;                                                        \
    if (err != cudaSuccess) {                                                      \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err)                     \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl;           \
        exit(err);                                                                 \
    }                                                                              \
}

// Tunable parameters
#define TILE_WIDTH 16
#define SEQ_LEN 256       // e.g., number of patches
#define MODEL_DIM 512     // model dimension

// GPU implementation

// GPU tiled matrix multiplication kernel, vomputes C = A * B
__global__ void tiledMatMulKernel(const float* A, const float* B, float* C,
                                  int M, int K, int N)
{
    __shared__ float tileA[TILE_WIDTH][TILE_WIDTH];
    __shared__ float tileB[TILE_WIDTH][TILE_WIDTH];

    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;
    float value = 0.0f;
    // Loop over tiles
    for (int t = 0; t < (K + TILE_WIDTH - 1) / TILE_WIDTH; t++) {
        // Load tile from A
        if (row < M && (t * TILE_WIDTH + threadIdx.x) < K)
            tileA[threadIdx.y][threadIdx.x] = A[row * K + t * TILE_WIDTH + threadIdx.x];
        else
            tileA[threadIdx.y][threadIdx.x] = 0.0f;
        // Load tile from B
        if (col < N && (t * TILE_WIDTH + threadIdx.y) < K)
            tileB[threadIdx.y][threadIdx.x] = B[(t * TILE_WIDTH + threadIdx.y) * N + col];
        else
            tileB[threadIdx.y][threadIdx.x] = 0.0f;
        __syncthreads();

        // Multiply the two tiles
        for (int i = 0; i < TILE_WIDTH; i++) {
            value += tileA[threadIdx.y][i] * tileB[i][threadIdx.x];
        }
        __syncthreads();
    }
    if(row < M && col < N)
        C[row * N + col] = value;
}

// GPU kernel to transpose a matrix
__global__ void transposeKernel(const float* in, float* out, int rows, int cols)
{
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if(r < rows && c < cols)
        out[c * rows + r] = in[r * cols + c];
}

// GPU kernel to scale an entire matrix by a constant factor
__global__ void scaleKernel(float* matrix, int size, float factor)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < size)
        matrix[idx] *= factor;
}

// GPU kernel to apply softmax row-wise on a matrix
__global__ void softmaxKernel(float* matrix, int rows, int cols)
{
    int row = blockIdx.x;  // one block per row
    if(row < rows) {
        // Compute maximum value in the row for numerical stability
        float maxVal = matrix[row * cols];
        for (int j = 1; j < cols; j++) {
            float val = matrix[row * cols + j];
            if(val > maxVal)
                maxVal = val;
        }
        // Compute exponential sum
        float sum = 0.0f;
        for (int j = 0; j < cols; j++) {
            float exp_val = expf(matrix[row * cols + j] - maxVal);
            matrix[row * cols + j] = exp_val;
            sum += exp_val;
        }
        // Normalize
        for (int j = 0; j < cols; j++) {
            matrix[row * cols + j] /= sum;
        }
    }
}

// CPU implementation

// Matrix multiplication
void cpuMatMul(const float* A, const float* B, float* C,
               int M, int K, int N)
{
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i*K + k] * B[k*N + j];
            }
            C[i*N + j] = sum;
        }
    }
}

// Transpose
void cpuTranspose(const float* in, float* out, int rows, int cols)
{
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            out[c * rows + r] = in[r * cols + c];
        }
    }
}

// Softmax (row-wise)
void cpuSoftmax(float* matrix, int rows, int cols)
{
    for (int r = 0; r < rows; r++) {
        float maxVal = matrix[r * cols];
        for (int j = 1; j < cols; j++) {
            float val = matrix[r * cols + j];
            if(val > maxVal) maxVal = val;
        }
        float sum = 0.0f;
        for (int j = 0; j < cols; j++) {
            matrix[r * cols + j] = exp(matrix[r * cols + j] - maxVal);
            sum += matrix[r * cols + j];
        }
        for (int j = 0; j < cols; j++) {
            matrix[r * cols + j] /= sum;
        }
    }
}

//  GPU encoder implementation
//   Q = X * Wq, K = X * Wk, V = X * Wv,
//   scores = Q * (K)ᵀ, scaling of scores,
//   softmax on scores, then attention = scores * V,
//   and finally output = attention * Wo.
void gpu_encoder(const float* h_X, const float* h_Wq, const float* h_Wk,
                 const float* h_Wv, const float* h_Wo, float* h_out)
{
    int size_X = SEQ_LEN * MODEL_DIM;
    int size_W = MODEL_DIM * MODEL_DIM;
    int size_QKV = size_X;
    int size_scores = SEQ_LEN * SEQ_LEN;

    // Device pointers
    float *d_X, *d_Wq, *d_Wk, *d_Wv, *d_Wo;
    float *d_Q, *d_K, *d_V;
    float *d_Kt, *d_scores, *d_attention, *d_out;

    // Allocate device memory
    CHECK_CUDA(cudaMalloc(&d_X, size_X * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_Wq, size_W * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_Wk, size_W * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_Wv, size_W * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_Wo, size_W * sizeof(float)));

    CHECK_CUDA(cudaMalloc(&d_Q, size_QKV * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_K, size_QKV * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_V, size_QKV * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_Kt, size_QKV * sizeof(float)));  // Transposed K
    CHECK_CUDA(cudaMalloc(&d_scores, size_scores * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_attention, size_QKV * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_out, size_QKV * sizeof(float)));

    cudaEvent_t copyH2D_start, copyH2D_stop, kernel_start, kernel_stop, copyD2H_start, copyD2H_stop;
    CHECK_CUDA(cudaEventCreate(&copyH2D_start));
    CHECK_CUDA(cudaEventCreate(&copyH2D_stop));
    CHECK_CUDA(cudaEventCreate(&kernel_start));
    CHECK_CUDA(cudaEventCreate(&kernel_stop));
    CHECK_CUDA(cudaEventCreate(&copyD2H_start));
    CHECK_CUDA(cudaEventCreate(&copyD2H_stop));

    // Record host-to-device copy time
    CHECK_CUDA(cudaEventRecord(copyH2D_start));
    CHECK_CUDA(cudaMemcpy(d_X, h_X, size_X * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_Wq, h_Wq, size_W * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_Wk, h_Wk, size_W * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_Wv, h_Wv, size_W * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_Wo, h_Wo, size_W * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaEventRecord(copyH2D_stop));
    CHECK_CUDA(cudaEventSynchronize(copyH2D_stop));
    float time_H2D;
    CHECK_CUDA(cudaEventElapsedTime(&time_H2D, copyH2D_start, copyH2D_stop));

    CHECK_CUDA(cudaEventRecord(kernel_start));

    // Grid and block settings
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid_Q((MODEL_DIM + TILE_WIDTH - 1) / TILE_WIDTH, (SEQ_LEN + TILE_WIDTH - 1) / TILE_WIDTH);
    
    // Q = X * Wq
    tiledMatMulKernel<<<dimGrid_Q, dimBlock>>>(d_X, d_Wq, d_Q, SEQ_LEN, MODEL_DIM, MODEL_DIM);
    // K = X * Wk
    tiledMatMulKernel<<<dimGrid_Q, dimBlock>>>(d_X, d_Wk, d_K, SEQ_LEN, MODEL_DIM, MODEL_DIM);
    // V = X * Wv
    tiledMatMulKernel<<<dimGrid_Q, dimBlock>>>(d_X, d_Wv, d_V, SEQ_LEN, MODEL_DIM, MODEL_DIM);

    // Transpose K to get Kᵀ
    dim3 dimBlockT(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGridT((SEQ_LEN + TILE_WIDTH - 1) / TILE_WIDTH, (MODEL_DIM + TILE_WIDTH - 1) / TILE_WIDTH);
    transposeKernel<<<dimGridT, dimBlockT>>>(d_K, d_Kt, SEQ_LEN, MODEL_DIM);

    // scores = Q * (Kᵀ)   [dimensions: (SEQ_LEN x MODEL_DIM) * (MODEL_DIM x SEQ_LEN) -> (SEQ_LEN x SEQ_LEN)]
    dim3 dimGrid_scores((SEQ_LEN + TILE_WIDTH - 1) / TILE_WIDTH, (SEQ_LEN + TILE_WIDTH - 1) / TILE_WIDTH);
    tiledMatMulKernel<<<dimGrid_scores, dimBlock>>>(d_Q, d_Kt, d_scores, SEQ_LEN, MODEL_DIM, SEQ_LEN);

    // Scale scores: divide each element by sqrt(MODEL_DIM)
    float scale_factor = 1.0f / sqrtf((float)MODEL_DIM);
    int total_scores = SEQ_LEN * SEQ_LEN;
    int threadsPerBlock = 256;
    int blocks = (total_scores + threadsPerBlock - 1) / threadsPerBlock;
    scaleKernel<<<blocks, threadsPerBlock>>>(d_scores, total_scores, scale_factor);

    // Apply softmax on each row of scores
    // Launch one block per row
    softmaxKernel<<<SEQ_LEN, 1>>>(d_scores, SEQ_LEN, SEQ_LEN);

    // Compute attention = scores * V  [ (SEQ_LEN x SEQ_LEN) * (SEQ_LEN x MODEL_DIM) -> (SEQ_LEN x MODEL_DIM)]
    tiledMatMulKernel<<<dimGrid_Q, dimBlock>>>(d_scores, d_V, d_attention, SEQ_LEN, SEQ_LEN, MODEL_DIM);

    // Output = attention * Wo  [ (SEQ_LEN x MODEL_DIM) * (MODEL_DIM x MODEL_DIM) -> (SEQ_LEN x MODEL_DIM)]
    tiledMatMulKernel<<<dimGrid_Q, dimBlock>>>(d_attention, d_Wo, d_out, SEQ_LEN, MODEL_DIM, MODEL_DIM);

    CHECK_CUDA(cudaEventRecord(kernel_stop));
    CHECK_CUDA(cudaEventSynchronize(kernel_stop));
    float time_kernel;
    CHECK_CUDA(cudaEventElapsedTime(&time_kernel, kernel_start, kernel_stop));

    // Copy the result back to host
    CHECK_CUDA(cudaEventRecord(copyD2H_start));
    CHECK_CUDA(cudaMemcpy(h_out, d_out, size_QKV * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaEventRecord(copyD2H_stop));
    CHECK_CUDA(cudaEventSynchronize(copyD2H_stop));
    float time_D2H;
    CHECK_CUDA(cudaEventElapsedTime(&time_D2H, copyD2H_start, copyD2H_stop));

    // Clean up device memory and events
    CHECK_CUDA(cudaFree(d_X));
    CHECK_CUDA(cudaFree(d_Wq));
    CHECK_CUDA(cudaFree(d_Wk));
    CHECK_CUDA(cudaFree(d_Wv));
    CHECK_CUDA(cudaFree(d_Wo));
    CHECK_CUDA(cudaFree(d_Q));
    CHECK_CUDA(cudaFree(d_K));
    CHECK_CUDA(cudaFree(d_V));
    CHECK_CUDA(cudaFree(d_Kt));
    CHECK_CUDA(cudaFree(d_scores));
    CHECK_CUDA(cudaFree(d_attention));
    CHECK_CUDA(cudaFree(d_out));
    
    CHECK_CUDA(cudaEventDestroy(copyH2D_start));
    CHECK_CUDA(cudaEventDestroy(copyH2D_stop));
    CHECK_CUDA(cudaEventDestroy(kernel_start));
    CHECK_CUDA(cudaEventDestroy(kernel_stop));
    CHECK_CUDA(cudaEventDestroy(copyD2H_start));
    CHECK_CUDA(cudaEventDestroy(copyD2H_stop));

    // Print out GPU timing results (in milliseconds)
    std::cout << "GPU Timings (ms):\n";
    std::cout << "  Host -> Device copy: " << time_H2D << "\n";
    std::cout << "  Kernel execution:    " << time_kernel << "\n";
    std::cout << "  Device -> Host copy: " << time_D2H << "\n";
    std::cout << "  Total GPU time:      " << (time_H2D + time_kernel + time_D2H) << "\n";
}

// CPU encoder implementation
void cpu_encoder(const float* h_X, const float* h_Wq, const float* h_Wk,
                 const float* h_Wv, const float* h_Wo, float* h_out)
{
    // Allocate temporary buffers
    float* Q  = new float[SEQ_LEN * MODEL_DIM];
    float* K  = new float[SEQ_LEN * MODEL_DIM];
    float* V  = new float[SEQ_LEN * MODEL_DIM];
    float* Kt = new float[SEQ_LEN * MODEL_DIM];
    float* scores    = new float[SEQ_LEN * SEQ_LEN];
    float* attention = new float[SEQ_LEN * MODEL_DIM];

    // Q = X * Wq
    cpuMatMul(h_X, h_Wq, Q, SEQ_LEN, MODEL_DIM, MODEL_DIM);
    // K = X * Wk
    cpuMatMul(h_X, h_Wk, K, SEQ_LEN, MODEL_DIM, MODEL_DIM);
    // V = X * Wv
    cpuMatMul(h_X, h_Wv, V, SEQ_LEN, MODEL_DIM, MODEL_DIM);

    // Transpose K to get Kᵀ
    cpuTranspose(K, Kt, SEQ_LEN, MODEL_DIM);

    // scores = Q * Kᵀ  => [SEQ_LEN x SEQ_LEN]
    cpuMatMul(Q, Kt, scores, SEQ_LEN, MODEL_DIM, SEQ_LEN);

    // Scale scores by 1/sqrt(MODEL_DIM)
    float scale_factor = 1.0f / sqrtf((float)MODEL_DIM);
    for (int i = 0; i < SEQ_LEN * SEQ_LEN; i++)
        scores[i] *= scale_factor;

    // Apply softmax row-wise on scores
    cpuSoftmax(scores, SEQ_LEN, SEQ_LEN);

    // attention = scores * V  => [SEQ_LEN x MODEL_DIM]
    cpuMatMul(scores, V, attention, SEQ_LEN, SEQ_LEN, MODEL_DIM);

    // Output = attention * Wo  => [SEQ_LEN x MODEL_DIM]
    cpuMatMul(attention, h_Wo, h_out, SEQ_LEN, MODEL_DIM, MODEL_DIM);

    // Clean up temporary buffers
    delete[] Q;
    delete[] K;
    delete[] V;
    delete[] Kt;
    delete[] scores;
    delete[] attention;
}

int main()
{
    // Allocate host memory for input and weight matrices
    int size_X = SEQ_LEN * MODEL_DIM;
    int size_W = MODEL_DIM * MODEL_DIM;
    float *h_X   = new float[size_X];
    float *h_Wq  = new float[size_W];
    float *h_Wk  = new float[size_W];
    float *h_Wv  = new float[size_W];
    float *h_Wo  = new float[size_W];
    float *h_out_gpu = new float[size_X];  // result from GPU encoder
    float *h_out_cpu = new float[size_X];  // result from CPU encoder

    // Initialize input and weights with some random values.
    for (int i = 0; i < size_X; i++) {
        h_X[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    for (int i = 0; i < size_W; i++) {
        h_Wq[i] = static_cast<float>(rand()) / RAND_MAX;
        h_Wk[i] = static_cast<float>(rand()) / RAND_MAX;
        h_Wv[i] = static_cast<float>(rand()) / RAND_MAX;
        h_Wo[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // Launch encoder kernel
    cudaEvent_t gpu_total_start, gpu_total_stop;
    CHECK_CUDA(cudaEventCreate(&gpu_total_start));
    CHECK_CUDA(cudaEventCreate(&gpu_total_stop));
    CHECK_CUDA(cudaEventRecord(gpu_total_start));
    gpu_encoder(h_X, h_Wq, h_Wk, h_Wv, h_Wo, h_out_gpu);
    CHECK_CUDA(cudaEventRecord(gpu_total_stop));
    CHECK_CUDA(cudaEventSynchronize(gpu_total_stop));
    float gpu_total_time;
    CHECK_CUDA(cudaEventElapsedTime(&gpu_total_time, gpu_total_start, gpu_total_stop));
    std::cout << "Total GPU Inference Time (ms): " << gpu_total_time << "\n";
    CHECK_CUDA(cudaEventDestroy(gpu_total_start));
    CHECK_CUDA(cudaEventDestroy(gpu_total_stop));

    // CPU encoder
    auto cpu_start = std::chrono::high_resolution_clock::now();
    cpu_encoder(h_X, h_Wq, h_Wk, h_Wv, h_Wo, h_out_cpu);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_duration = cpu_end - cpu_start;
    std::cout << "Total CPU Inference Time (ms): " << cpu_duration.count() << "\n";

    // Clean up host memory
    delete[] h_X;
    delete[] h_Wq;
    delete[] h_Wk;
    delete[] h_Wv;
    delete[] h_Wo;
    delete[] h_out_gpu;
    delete[] h_out_cpu;

    return 0;
}
