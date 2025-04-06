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

// GPU Matrix Multiplication Kernel (for linear projections)
__global__
void matmul_kernel(const float *A, const float *B, float *C, int M, int K, int N)
{
    __shared__ float ds_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float ds_B[TILE_WIDTH][TILE_WIDTH];

    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;
    float sum = 0.0f;

    for (int t = 0; t < (K + TILE_WIDTH - 1) / TILE_WIDTH; t++) {
        if (row < M && t * TILE_WIDTH + threadIdx.x < K)
            ds_A[threadIdx.y][threadIdx.x] = A[row * K + t * TILE_WIDTH + threadIdx.x];
        else
            ds_A[threadIdx.y][threadIdx.x] = 0.0f;

        if (col < N && t * TILE_WIDTH + threadIdx.y < K)
            ds_B[threadIdx.y][threadIdx.x] = B[(t * TILE_WIDTH + threadIdx.y) * N + col];
        else
            ds_B[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();
        for (int i = 0; i < TILE_WIDTH; i++) {
            sum += ds_A[threadIdx.y][i] * ds_B[i][threadIdx.x];
        }
        __syncthreads();
    }
    if (row < M && col < N)
        C[row * N + col] = sum;
}

// CPU version of matrix multiplication
void cpu_matmul(const float* A, const float* B, float* C, int M, int K, int N) {
    for (int i = 0; i < M; i++){
        for (int j = 0; j < N; j++){
            float sum = 0.0f;
            for (int k = 0; k < K; k++){
                sum += A[i*K + k] * B[k*N + j];
            }
            C[i*N + j] = sum;
        }
    }
}

// GPU Kernels for Multi-Head Attention (Q, K, V already projected)
__global__
void compute_scores_kernel(const float *Q, const float *K, float *d_scores,
                           int M, int N, int H, int d_head)
{
    int h = blockIdx.z; // head index
    int row = blockIdx.y * TILE_WIDTH + threadIdx.y; // query index
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x; // key index

    __shared__ float sQ[TILE_WIDTH][TILE_WIDTH];
    __shared__ float sK[TILE_WIDTH][TILE_WIDTH];

    float val = 0.0f;
    for (int t = 0; t < (d_head + TILE_WIDTH - 1) / TILE_WIDTH; t++) {
        int kQ = t * TILE_WIDTH + threadIdx.x;
        int kK = t * TILE_WIDTH + threadIdx.y;

        if (row < M && kQ < d_head)
            sQ[threadIdx.y][threadIdx.x] = Q[row * (H * d_head) + h * d_head + kQ];
        else
            sQ[threadIdx.y][threadIdx.x] = 0.0f;

        if (col < N && kK < d_head)
            sK[threadIdx.y][threadIdx.x] = K[col * (H * d_head) + h * d_head + kK];
        else
            sK[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        for (int i = 0; i < TILE_WIDTH; i++){
            val += sQ[threadIdx.y][i] * sK[i][threadIdx.x];
        }
        __syncthreads();
    }
    val = val / sqrtf((float)d_head);
    if (row < M && col < N)
        d_scores[((row * N) + col) * H + h] = val;
}


// Softmax kernel: for each (query, head) pair, compute softmax over the key dimension.
__global__
void softmax_kernel(float *d_scores, int M, int N, int H)
{
    int idx = blockIdx.x; // block for each (i, h) pair; total blocks = M*H
    int i = idx / H;
    int h = idx % H;
    int t = threadIdx.x;
    extern __shared__ float shmem[];

    float thread_max = -1e20f;
    for (int j = t; j < N; j += blockDim.x) {
        float val = d_scores[((i * N) + j) * H + h];
        if (val > thread_max) thread_max = val;
    }
    shmem[t] = thread_max;
    __syncthreads();
    for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
        if (t < stride && (t + stride) < blockDim.x)
            shmem[t] = fmaxf(shmem[t], shmem[t+stride]);
        __syncthreads();
    }
    float max_val = shmem[0];
    __syncthreads();

    float sum_exp = 0.0f;
    for (int j = t; j < N; j += blockDim.x) {
        float val = d_scores[((i * N) + j) * H + h];
        sum_exp += expf(val - max_val);
    }
    shmem[t] = sum_exp;
    __syncthreads();
    for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
        if (t < stride)
            shmem[t] += shmem[t+stride];
        __syncthreads();
    }
    float total_exp = shmem[0];
    __syncthreads();
    for (int j = t; j < N; j += blockDim.x) {
        float val = d_scores[((i * N) + j) * H + h];
        d_scores[((i * N) + j) * H + h] = expf(val - max_val) / total_exp;
    }
}

// Weighted sum kernel:
__global__
void weighted_sum_kernel(const float *d_scores, const float *V, float *O,
                         int M, int N, int H, int d_head)
{
    int h = blockIdx.z;
    int i = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int k = blockIdx.x * TILE_WIDTH + threadIdx.x;

    if (i < M && k < d_head) {
        float sum_val = 0.0f;
        for (int j = 0; j < N; j++){
            float score = d_scores[((i * N) + j) * H + h];
            float v_val = V[j * (H * d_head) + h * d_head + k];
            sum_val += score * v_val;
        }
        O[i * (H * d_head) + h * d_head + k] = sum_val;
    }
}

// CPU Implementation of Multi-Head Attention (for verification)
void cpu_multi_head_attention(const float *Q, const float *K, const float *V,
                              float *O, int M, int N, int H, int d_head)
{
    for (int h = 0; h < H; h++){
        for (int i = 0; i < M; i++){
            float *scores = new float[N];
            float max_val = -1e20f;
            for (int j = 0; j < N; j++){
                float dot = 0.0f;
                for (int k = 0; k < d_head; k++){
                    float q_val = Q[i * (H*d_head) + h*d_head + k];
                    float k_val = K[j * (H*d_head) + h*d_head + k];
                    dot += q_val * k_val;
                }
                dot /= sqrtf((float)d_head);
                scores[j] = dot;
                if (dot > max_val) max_val = dot;
            }
            float sum_exp = 0.0f;
            for (int j = 0; j < N; j++){
                scores[j] = expf(scores[j] - max_val);
                sum_exp += scores[j];
            }
            for (int j = 0; j < N; j++){
                scores[j] /= sum_exp;
            }
            for (int k = 0; k < d_head; k++){
                float val = 0.0f;
                for (int j = 0; j < N; j++){
                    float v_val = V[j * (H*d_head) + h*d_head + k];
                    val += scores[j] * v_val;
                }
                O[i * (H*d_head) + h*d_head + k] = val;
            }
            delete[] scores;
        }
    }
}

int main()
{
    // Dimensions.
    const int M = 512;         // number of queries (and keys/values in self-attention)
    const int d_total = 256;   // total hidden dimension (d_model)
    const int H = 4;           // number of heads
    const int d_head = d_total / H;
    const int N = M;           // number of keys/values

    // Host allocations for input and weights
    float *h_X   = new float[M * d_total];      // Input X: [M, d_total]
    float *h_Wq  = new float[d_total * d_total];  // Weight for Q
    float *h_Wk  = new float[d_total * d_total];  // Weight for K
    float *h_Wv  = new float[d_total * d_total];  // Weight for V
    float *h_Wo  = new float[d_total * d_total];  // Final projection weight

    // Host allocations for projected Q, K, V
    float *h_Q = new float[M * d_total];
    float *h_K = new float[M * d_total];  // For self-attention, same M as input.
    float *h_V = new float[M * d_total];
    // Output of multi-head attention (before final projection)
    float *h_O = new float[M * d_total];
    // Final output
    float *h_final = new float[M * d_total];

    // Initialize input and weights with sample values
    for (int i = 0; i < M * d_total; i++){
        h_X[i] = static_cast<float>((i + 1) % 100) / 100.0f;
    }
    for (int i = 0; i < d_total * d_total; i++){
        h_Wq[i] = static_cast<float>((i + 2) % 100) / 100.0f;
        h_Wk[i] = static_cast<float>((i + 3) % 100) / 100.0f;
        h_Wv[i] = static_cast<float>((i + 4) % 100) / 100.0f;
        h_Wo[i] = static_cast<float>((i + 5) % 100) / 100.0f;
    }

    // CPU Implementation
    auto cpu_start = std::chrono::high_resolution_clock::now();

    // CPU linear projections: Q = X*Wq, K = X*Wk, V = X*Wv
    cpu_matmul(h_X, h_Wq, h_Q, M, d_total, d_total);
    cpu_matmul(h_X, h_Wk, h_K, M, d_total, d_total);
    cpu_matmul(h_X, h_Wv, h_V, M, d_total, d_total);

    // CPU Multi-Head Attention
    cpu_multi_head_attention(h_Q, h_K, h_V, h_O, M, N, H, d_head);

    // CPU final projection: Out = O * Wo
    cpu_matmul(h_O, h_Wo, h_final, M, d_total, d_total);

    auto cpu_end = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration_cast<std::chrono::milliseconds>(cpu_end - cpu_start).count();

    // GPU Implementation

    // Allocate device memory
    float *d_X, *d_Wq, *d_Wk, *d_Wv, *d_Wo;
    float *d_Q, *d_K, *d_V;
    float *d_scores, *d_O;

    size_t input_size  = M * d_total * sizeof(float);
    size_t weight_size = d_total * d_total * sizeof(float);
    size_t proj_size   = M * d_total * sizeof(float);
    size_t scores_size = M * N * H * sizeof(float); // [M, N, H]
    
    CHECK_CUDA(cudaMalloc(&d_X, input_size));
    CHECK_CUDA(cudaMalloc(&d_Wq, weight_size));
    CHECK_CUDA(cudaMalloc(&d_Wk, weight_size));
    CHECK_CUDA(cudaMalloc(&d_Wv, weight_size));
    CHECK_CUDA(cudaMalloc(&d_Wo, weight_size));
    CHECK_CUDA(cudaMalloc(&d_Q, proj_size));
    CHECK_CUDA(cudaMalloc(&d_K, proj_size));
    CHECK_CUDA(cudaMalloc(&d_V, proj_size));
    CHECK_CUDA(cudaMalloc(&d_scores, scores_size));
    CHECK_CUDA(cudaMalloc(&d_O, proj_size));

    cudaEvent_t start_total, stop_total;
    cudaEvent_t start_h2d, stop_h2d;
    cudaEvent_t start_proj, stop_proj;
    cudaEvent_t start_attn, stop_attn;
    cudaEvent_t start_final, stop_final;
    cudaEvent_t start_d2h, stop_d2h;

    CHECK_CUDA(cudaEventCreate(&start_total));
    CHECK_CUDA(cudaEventCreate(&stop_total));
    CHECK_CUDA(cudaEventCreate(&start_h2d));
    CHECK_CUDA(cudaEventCreate(&stop_h2d));
    CHECK_CUDA(cudaEventCreate(&start_proj));
    CHECK_CUDA(cudaEventCreate(&stop_proj));
    CHECK_CUDA(cudaEventCreate(&start_attn));
    CHECK_CUDA(cudaEventCreate(&stop_attn));
    CHECK_CUDA(cudaEventCreate(&start_final));
    CHECK_CUDA(cudaEventCreate(&stop_final));
    CHECK_CUDA(cudaEventCreate(&start_d2h));
    CHECK_CUDA(cudaEventCreate(&stop_d2h));

    CHECK_CUDA(cudaEventRecord(start_total, 0));

    // Host-to-Device copy
    CHECK_CUDA(cudaEventRecord(start_h2d, 0));
    CHECK_CUDA(cudaMemcpy(d_X, h_X, input_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_Wq, h_Wq, weight_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_Wk, h_Wk, weight_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_Wv, h_Wv, weight_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_Wo, h_Wo, weight_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaEventRecord(stop_h2d, 0));
    CHECK_CUDA(cudaEventSynchronize(stop_h2d));
    float h2d_ms = 0;
    CHECK_CUDA(cudaEventElapsedTime(&h2d_ms, start_h2d, stop_h2d));

    // GPU Linear Projections: Q = X*Wq, K = X*Wk, V = X*Wv
    dim3 block(TILE_WIDTH, TILE_WIDTH);
    dim3 grid((d_total + TILE_WIDTH - 1) / TILE_WIDTH, (M + TILE_WIDTH - 1) / TILE_WIDTH);
    CHECK_CUDA(cudaEventRecord(start_proj, 0));
    // Q = X * Wq.
    matmul_kernel<<<grid, block>>>(d_X, d_Wq, d_Q, M, d_total, d_total);
    // K = X * Wk.
    matmul_kernel<<<grid, block>>>(d_X, d_Wk, d_K, M, d_total, d_total);
    // V = X * Wv.
    matmul_kernel<<<grid, block>>>(d_X, d_Wv, d_V, M, d_total, d_total);
    CHECK_CUDA(cudaEventRecord(stop_proj, 0));
    CHECK_CUDA(cudaEventSynchronize(stop_proj));
    float proj_ms = 0;
    CHECK_CUDA(cudaEventElapsedTime(&proj_ms, start_proj, stop_proj));

    // GPU Multi-Head Attention
    CHECK_CUDA(cudaEventRecord(start_attn, 0));
    // Kernel: Compute scaled dot-product scores
    dim3 block1(TILE_WIDTH, TILE_WIDTH);
    dim3 grid1((N + TILE_WIDTH - 1) / TILE_WIDTH,
               (M + TILE_WIDTH - 1) / TILE_WIDTH,
               H);
    compute_scores_kernel<<<grid1, block1>>>(d_Q, d_K, d_scores, M, N, H, d_head);

    // Kernel: Softmax over keys
    int threads = 256;
    int blocks = M * H;
    size_t shared_size = threads * sizeof(float);
    softmax_kernel<<<blocks, threads, shared_size>>>(d_scores, M, N, H);

    // Kernel: Weighted sum to get output O
    dim3 block3(TILE_WIDTH, TILE_WIDTH);
    dim3 grid3((d_head + TILE_WIDTH - 1) / TILE_WIDTH,
               (M + TILE_WIDTH - 1) / TILE_WIDTH,
               H);
    weighted_sum_kernel<<<grid3, block3>>>(d_scores, d_V, d_O, M, N, H, d_head);
    CHECK_CUDA(cudaEventRecord(stop_attn, 0));
    CHECK_CUDA(cudaEventSynchronize(stop_attn));
    float attn_ms = 0;
    CHECK_CUDA(cudaEventElapsedTime(&attn_ms, start_attn, stop_attn));

    // GPU Final Projection: Out = O * Wo
    dim3 grid_final((d_total + TILE_WIDTH - 1) / TILE_WIDTH, (M + TILE_WIDTH - 1) / TILE_WIDTH);
    CHECK_CUDA(cudaEventRecord(start_final, 0));
    matmul_kernel<<<grid_final, block>>>(d_O, d_Wo, d_X, M, d_total, d_total);

    CHECK_CUDA(cudaEventRecord(stop_final, 0));
    CHECK_CUDA(cudaEventSynchronize(stop_final));
    float final_ms = 0;
    CHECK_CUDA(cudaEventElapsedTime(&final_ms, start_final, stop_final));

    // Device-to-Host copy
    CHECK_CUDA(cudaEventRecord(start_d2h, 0));
    CHECK_CUDA(cudaMemcpy(h_final, d_X, input_size, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaEventRecord(stop_d2h, 0));
    CHECK_CUDA(cudaEventSynchronize(stop_d2h));
    float d2h_ms = 0;
    CHECK_CUDA(cudaEventElapsedTime(&d2h_ms, start_d2h, stop_d2h));

    CHECK_CUDA(cudaEventRecord(stop_total, 0));
    CHECK_CUDA(cudaEventSynchronize(stop_total));
    float total_gpu_ms = 0;
    CHECK_CUDA(cudaEventElapsedTime(&total_gpu_ms, start_total, stop_total));

    std::cout << "CPU Total Time: " << cpu_ms << " ms\n";
    std::cout << "GPU Host-to-Device Copy Time: " << h2d_ms << " ms\n";
    std::cout << "GPU Linear Projection Time (Q, K, V): " << proj_ms << " ms\n";
    std::cout << "GPU Multi-Head Attention Time: " << attn_ms << " ms\n";
    std::cout << "GPU Final Projection Time: " << final_ms << " ms\n";
    std::cout << "GPU Device-to-Host Copy Time: " << d2h_ms << " ms\n";
    std::cout << "Total GPU Time: " << total_gpu_ms << " ms\n";

    // Cleanup host and device memory
    delete[] h_X;
    delete[] h_Wq;
    delete[] h_Wk;
    delete[] h_Wv;
    delete[] h_Wo;
    delete[] h_Q;
    delete[] h_K;
    delete[] h_V;
    delete[] h_O;
    delete[] h_final;

    cudaFree(d_X);
    cudaFree(d_Wq);
    cudaFree(d_Wk);
    cudaFree(d_Wv);
    cudaFree(d_Wo);
    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_scores);
    cudaFree(d_O);

    cudaEventDestroy(start_total);
    cudaEventDestroy(stop_total);
    cudaEventDestroy(start_h2d);
    cudaEventDestroy(stop_h2d);
    cudaEventDestroy(start_proj);
    cudaEventDestroy(stop_proj);
    cudaEventDestroy(start_attn);
    cudaEventDestroy(stop_attn);
    cudaEventDestroy(start_final);
    cudaEventDestroy(stop_final);
    cudaEventDestroy(start_d2h);
    cudaEventDestroy(stop_d2h);

    return 0;
}
