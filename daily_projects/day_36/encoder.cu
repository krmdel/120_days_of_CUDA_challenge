#include <iostream>
#include <cmath>
#include <chrono>
#include <cuda_runtime.h>

#define TILE_WIDTH 16

// CUDA error checking macro
#define CHECK_CUDA(call) {                                              \
    cudaError_t err = call;                                             \
    if(err != cudaSuccess) {                                            \
        std::cerr << "CUDA error: " << cudaGetErrorString(err)          \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl;  \
        exit(1);                                                        \
    }                                                                   \
}

// GPU Kernels

// Matrix Multiplication Kernel (for linear projections)
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

// Multi-Head Attention Kernels
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

// Softmax kernel: For each (query, head) pair, compute softmax over keys
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

// Weighted sum kernel to compute attention output
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

// New Kernels for the Full Encoder

// Element-wise addition kernel: C = A + B
__global__
void add_kernel(const float *A, const float *B, float *C, int total_elements)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < total_elements) {
        C[idx] = A[idx] + B[idx];
    }
}

// ReLU activation kernel
__global__
void relu_kernel(float *A, int total_elements)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < total_elements) {
        if(A[idx] < 0)
            A[idx] = 0;
    }
}

// Layer normalization kernel
__global__
void layer_norm_kernel(const float* input, float* output, int M, int N, float epsilon)
{
    int row = blockIdx.x; // one block per row
    if (row < M) {
        extern __shared__ float shmem[];
        int tid = threadIdx.x;
        float sum = 0.0f;
        // Compute mean
        for (int j = tid; j < N; j += blockDim.x) {
            sum += input[row * N + j];
        }
        shmem[tid] = sum;
        __syncthreads();
        for (int stride = blockDim.x/2; stride > 0; stride /= 2) {
            if(tid < stride) {
                shmem[tid] += shmem[tid + stride];
            }
            __syncthreads();
        }
        float mean = shmem[0] / N;
        __syncthreads();

        // Compute variance
        float var_sum = 0.0f;
        for (int j = tid; j < N; j += blockDim.x) {
            float diff = input[row * N + j] - mean;
            var_sum += diff * diff;
        }
        shmem[tid] = var_sum;
        __syncthreads();
        for (int stride = blockDim.x/2; stride > 0; stride /= 2) {
            if(tid < stride) {
                shmem[tid] += shmem[tid + stride];
            }
            __syncthreads();
        }
        float var = shmem[0] / N;
        float inv_std = rsqrtf(var + epsilon);
        // Normalize
        for (int j = tid; j < N; j += blockDim.x) {
            int idx = row * N + j;
            output[idx] = (input[idx] - mean) * inv_std;
        }
    }
}

// CPU Functions

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

// CPU Implementation of Multi-Head Attention
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

// CPU element-wise addition: C = A + B
void cpu_add(const float *A, const float *B, float *C, int total_elements) {
    for (int i = 0; i < total_elements; i++){
        C[i] = A[i] + B[i];
    }
}

// CPU in-place ReLU activation
void cpu_relu(float *A, int total_elements) {
    for (int i = 0; i < total_elements; i++){
        if (A[i] < 0) {
            A[i] = 0;
        }
    }
}

// CPU Layer Normalization
void cpu_layer_norm(const float* input, float* output, int M, int N, float epsilon) {
    for (int i = 0; i < M; i++){
        float sum = 0.0f;
        for (int j = 0; j < N; j++){
            sum += input[i*N + j];
        }
        float mean = sum / N;
        float var = 0.0f;
        for (int j = 0; j < N; j++){
            float diff = input[i*N + j] - mean;
            var += diff * diff;
        }
        var /= N;
        float inv_std = 1.0f / sqrt(var + epsilon);
        for (int j = 0; j < N; j++){
            output[i*N + j] = (input[i*N + j] - mean) * inv_std;
        }
    }
}

int main()
{
    // Dimensions
    const int M = 512;         // number of tokens (sequence length)
    const int d_total = 256;   // model (hidden) dimension (d_model)
    const int H = 4;           // number of attention heads
    const int d_head = d_total / H;
    const int N = M;           // number of keys/values in self-attention
    const int d_ff = 4 * d_total; // feed-forward inner dimension

    size_t input_size   = M * d_total * sizeof(float);
    size_t weight_size  = d_total * d_total * sizeof(float);
    size_t proj_size    = M * d_total * sizeof(float);
    size_t scores_size  = M * N * H * sizeof(float);
    size_t ffn1_size    = M * d_ff * sizeof(float);

    // Host allocations for input and weights
    float *h_X   = new float[M * d_total];      // Input X: [M, d_total]
    float *h_Wq  = new float[d_total * d_total];  // Weight for Q
    float *h_Wk  = new float[d_total * d_total];  // Weight for K
    float *h_Wv  = new float[d_total * d_total];  // Weight for V
    float *h_Wo  = new float[d_total * d_total];  // Weight for final projection in attention
    float *h_W1  = new float[d_total * d_ff];      // Weight for feed-forward layer 1
    float *h_W2  = new float[d_ff * d_total];      // Weight for feed-forward layer 2

    // Host allocation for final outputs
    float *h_final_gpu = new float[M * d_total];  // From GPU
    float *h_final_cpu = new float[M * d_total];  // From CPU

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
    for (int i = 0; i < d_total * d_ff; i++){
        h_W1[i] = static_cast<float>((i + 6) % 100) / 100.0f;
    }
    for (int i = 0; i < d_ff * d_total; i++){
        h_W2[i] = static_cast<float>((i + 7) % 100) / 100.0f;
    }

    // CPU Encoder Implementation

    // Allocate intermediate CPU arrays
    float *cpu_Q     = new float[M * d_total];
    float *cpu_K     = new float[M * d_total];
    float *cpu_V     = new float[M * d_total];
    float *cpu_attn  = new float[M * d_total];  // multi-head attention output (before final projection)
    float *cpu_MHA   = new float[M * d_total];  // after final projection (attention branch)
    float *cpu_add1  = new float[M * d_total];  // residual: X + MHA
    float *cpu_ln1   = new float[M * d_total];  // after first layer normalization
    float *cpu_ffn1  = new float[M * d_ff];     // feed-forward layer 1 output (before activation)
    float *cpu_ffn2  = new float[M * d_total];  // feed-forward layer 2 output
    float *cpu_add2  = new float[M * d_total];  // residual: ln1 + ffn2
    float *cpu_ln2   = new float[M * d_total];  // final CPU output

    auto cpu_start = std::chrono::high_resolution_clock::now();
    // Linear projections: Q = X * Wq, K = X * Wk, V = X * Wv
    cpu_matmul(h_X, h_Wq, cpu_Q, M, d_total, d_total);
    cpu_matmul(h_X, h_Wk, cpu_K, M, d_total, d_total);
    cpu_matmul(h_X, h_Wv, cpu_V, M, d_total, d_total);

    // Multi-Head Attention
    cpu_multi_head_attention(cpu_Q, cpu_K, cpu_V, cpu_attn, M, N, H, d_head);

    // Final projection for attention branch: MHA = attn * Wo
    cpu_matmul(cpu_attn, h_Wo, cpu_MHA, M, d_total, d_total);

    // Add & Norm after Attention: add residual X + MHA, then layer norm
    cpu_add(h_X, cpu_MHA, cpu_add1, M * d_total);
    cpu_layer_norm(cpu_add1, cpu_ln1, M, d_total, 1e-5);

    // Feed-Forward Network:

    // First linear layer: ffn1 = ln1 * W1
    cpu_matmul(cpu_ln1, h_W1, cpu_ffn1, M, d_total, d_ff);
    // ReLU activation.
    cpu_relu(cpu_ffn1, M * d_ff);
    // Second linear layer: ffn2 = ffn1 * W2
    cpu_matmul(cpu_ffn1, h_W2, cpu_ffn2, M, d_ff, d_total);

    // Add & Norm after Feed-Forward: ln1 + ffn2, then layer norm
    cpu_add(cpu_ln1, cpu_ffn2, cpu_add2, M * d_total);
    cpu_layer_norm(cpu_add2, cpu_ln2, M, d_total, 1e-5);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration_cast<std::chrono::milliseconds>(cpu_end - cpu_start).count();
    
    // GPU Encoder Implementation
    
    // Device memory allocations
    float *d_input, *d_X;
    float *d_Wq, *d_Wk, *d_Wv, *d_Wo;
    float *d_Q, *d_K, *d_V;
    float *d_scores, *d_O;
    // Buffers for encoder block
    float *d_MHA;     // multi-head attention output (after final projection)
    float *d_add1;    // residual: input + MHA
    float *d_ln1;     // after first layer normalization
    float *d_ffn1;    // feed-forward layer 1 output (before activation)
    float *d_ffn2;    // feed-forward layer 2 output
    float *d_add2;    // residual: ln1 + ffn2
    float *d_ln2;     // final output after second layer norm
    float *d_W1, *d_W2;

    CHECK_CUDA(cudaMalloc(&d_input, input_size));
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

    // Allocations for encoder intermediate results
    CHECK_CUDA(cudaMalloc(&d_MHA, proj_size));
    CHECK_CUDA(cudaMalloc(&d_add1, proj_size));
    CHECK_CUDA(cudaMalloc(&d_ln1, proj_size));
    CHECK_CUDA(cudaMalloc(&d_ffn1, ffn1_size));
    CHECK_CUDA(cudaMalloc(&d_ffn2, proj_size));
    CHECK_CUDA(cudaMalloc(&d_add2, proj_size));
    CHECK_CUDA(cudaMalloc(&d_ln2, proj_size));
    CHECK_CUDA(cudaMalloc(&d_W1, d_total * d_ff * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_W2, d_ff * d_total * sizeof(float)));

    // Create CUDA events for timing
    cudaEvent_t start_total, stop_total;
    cudaEvent_t start_h2d, stop_h2d;
    cudaEvent_t start_kernel, stop_kernel;
    cudaEvent_t start_d2h, stop_d2h;
    CHECK_CUDA(cudaEventCreate(&start_total));
    CHECK_CUDA(cudaEventCreate(&stop_total));
    CHECK_CUDA(cudaEventCreate(&start_h2d));
    CHECK_CUDA(cudaEventCreate(&stop_h2d));
    CHECK_CUDA(cudaEventCreate(&start_kernel));
    CHECK_CUDA(cudaEventCreate(&stop_kernel));
    CHECK_CUDA(cudaEventCreate(&start_d2h));
    CHECK_CUDA(cudaEventCreate(&stop_d2h));

    CHECK_CUDA(cudaEventRecord(start_total, 0));

    // Host-to-Device copy
    CHECK_CUDA(cudaEventRecord(start_h2d, 0));
    CHECK_CUDA(cudaMemcpy(d_input, h_X, input_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_X, h_X, input_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_Wq, h_Wq, weight_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_Wk, h_Wk, weight_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_Wv, h_Wv, weight_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_Wo, h_Wo, weight_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_W1, h_W1, d_total * d_ff * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_W2, h_W2, d_ff * d_total * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaEventRecord(stop_h2d, 0));
    CHECK_CUDA(cudaEventSynchronize(stop_h2d));
    float h2d_ms = 0;
    CHECK_CUDA(cudaEventElapsedTime(&h2d_ms, start_h2d, stop_h2d));

    CHECK_CUDA(cudaEventRecord(start_kernel, 0));

    // Linear Projections for Multi-Head Attention
    dim3 block(TILE_WIDTH, TILE_WIDTH);
    dim3 grid((d_total + TILE_WIDTH - 1) / TILE_WIDTH, (M + TILE_WIDTH - 1) / TILE_WIDTH);
    // Q = input * Wq
    matmul_kernel<<<grid, block>>>(d_input, d_Wq, d_Q, M, d_total, d_total);
    // K = input * Wk
    matmul_kernel<<<grid, block>>>(d_input, d_Wk, d_K, M, d_total, d_total);
    // V = input * Wv
    matmul_kernel<<<grid, block>>>(d_input, d_Wv, d_V, M, d_total, d_total);

    // Multi-Head Attention
    // Compute scaled dot-product scores
    dim3 block1(TILE_WIDTH, TILE_WIDTH);
    dim3 grid1((N + TILE_WIDTH - 1) / TILE_WIDTH,
               (M + TILE_WIDTH - 1) / TILE_WIDTH,
               H);
    compute_scores_kernel<<<grid1, block1>>>(d_Q, d_K, d_scores, M, N, H, d_head);

    // Softmax over key dimension
    int threads = 256;
    int blocks = M * H;
    size_t shared_size = threads * sizeof(float);
    softmax_kernel<<<blocks, threads, shared_size>>>(d_scores, M, N, H);

    // Weighted sum to get attention output
    dim3 block3(TILE_WIDTH, TILE_WIDTH);
    dim3 grid3((d_head + TILE_WIDTH - 1) / TILE_WIDTH,
               (M + TILE_WIDTH - 1) / TILE_WIDTH,
               H);
    weighted_sum_kernel<<<grid3, block3>>>(d_scores, d_V, d_O, M, N, H, d_head);

    // Final projection of attention output: MHA = O * Wo
    dim3 grid_final((d_total + TILE_WIDTH - 1) / TILE_WIDTH, (M + TILE_WIDTH - 1) / TILE_WIDTH);
    matmul_kernel<<<grid_final, block>>>(d_O, d_Wo, d_MHA, M, d_total, d_total);

    // Add & Norm after Multi-Head Attention
    int threads_add = 256;
    int blocks_add = (M * d_total + threads_add - 1) / threads_add;
    add_kernel<<<blocks_add, threads_add>>>(d_input, d_MHA, d_add1, M * d_total);
    int ln_threads = 256;
    layer_norm_kernel<<<M, ln_threads, ln_threads * sizeof(float)>>>(d_add1, d_ln1, M, d_total, 1e-5);

    // Feed-Forward Network

    // First linear layer: ffn1 = ln1 * W1
    dim3 grid_ffn1((d_ff + TILE_WIDTH - 1) / TILE_WIDTH, (M + TILE_WIDTH - 1) / TILE_WIDTH);
    matmul_kernel<<<grid_ffn1, block>>>(d_ln1, d_W1, d_ffn1, M, d_total, d_ff);
    // ReLU activation
    int total_ffn1 = M * d_ff;
    int blocks_relu = (total_ffn1 + threads_add - 1) / threads_add;
    relu_kernel<<<blocks_relu, threads_add>>>(d_ffn1, total_ffn1);
    // Second linear layer: ffn2 = ffn1 * W2
    dim3 grid_ffn2((d_total + TILE_WIDTH - 1) / TILE_WIDTH, (M + TILE_WIDTH - 1) / TILE_WIDTH);
    matmul_kernel<<<grid_ffn2, block>>>(d_ffn1, d_W2, d_ffn2, M, d_ff, d_total);

    // Add & Norm after Feed-Forward
    add_kernel<<<blocks_add, threads_add>>>(d_ln1, d_ffn2, d_add2, M * d_total);
    layer_norm_kernel<<<M, ln_threads, ln_threads * sizeof(float)>>>(d_add2, d_ln2, M, d_total, 1e-5);

    CHECK_CUDA(cudaEventRecord(stop_kernel, 0));
    CHECK_CUDA(cudaEventSynchronize(stop_kernel));
    float kernel_ms = 0;
    CHECK_CUDA(cudaEventElapsedTime(&kernel_ms, start_kernel, stop_kernel));

    // Device-to-Host copy
    CHECK_CUDA(cudaEventRecord(start_d2h, 0));
    CHECK_CUDA(cudaMemcpy(h_final_gpu, d_ln2, input_size, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaEventRecord(stop_d2h, 0));
    CHECK_CUDA(cudaEventSynchronize(stop_d2h));
    float d2h_ms = 0;
    CHECK_CUDA(cudaEventElapsedTime(&d2h_ms, start_d2h, stop_d2h));

    CHECK_CUDA(cudaEventRecord(stop_total, 0));
    CHECK_CUDA(cudaEventSynchronize(stop_total));
    float total_gpu_ms = 0;
    CHECK_CUDA(cudaEventElapsedTime(&total_gpu_ms, start_total, stop_total));

    // Print the timings
    std::cout << "CPU Total Time: " << cpu_ms << " ms\n";
    std::cout << "GPU Host-to-Device Copy Time: " << h2d_ms << " ms\n";
    std::cout << "GPU Kernel Execution Time: " << kernel_ms << " ms\n";
    std::cout << "GPU Device-to-Host Copy Time: " << d2h_ms << " ms\n";
    std::cout << "Total GPU Time: " << total_gpu_ms << " ms\n";

    // Cleanup host and device memory
    delete[] h_X;
    delete[] h_Wq;
    delete[] h_Wk;
    delete[] h_Wv;
    delete[] h_Wo;
    delete[] h_W1;
    delete[] h_W2;
    delete[] h_final_gpu;
    delete[] h_final_cpu;

    delete[] cpu_Q;
    delete[] cpu_K;
    delete[] cpu_V;
    delete[] cpu_attn;
    delete[] cpu_MHA;
    delete[] cpu_add1;
    delete[] cpu_ln1;
    delete[] cpu_ffn1;
    delete[] cpu_ffn2;
    delete[] cpu_add2;
    delete[] cpu_ln2;

    cudaFree(d_input);
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
    cudaFree(d_MHA);
    cudaFree(d_add1);
    cudaFree(d_ln1);
    cudaFree(d_ffn1);
    cudaFree(d_ffn2);
    cudaFree(d_add2);
    cudaFree(d_ln2);
    cudaFree(d_W1);
    cudaFree(d_W2);

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
