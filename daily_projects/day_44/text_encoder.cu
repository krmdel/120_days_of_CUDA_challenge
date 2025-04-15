#include <iostream>
#include <cmath>
#include <chrono>
#include <cuda_runtime.h>

// Parameters
const int SEQ_LEN  = 256;    // Sequence length (number of tokens)
const int d_model  = 256;    // Model (embedding) dimension
const int H        = 8;      // Number of attention heads
const int d_head   = d_model / H;   // Dimension per head (256/8 = 32)
const int d_ff     = 4 * d_model;    // Feed-forward hidden dimension (1024)

// Tiling parameter for CUDA kernels
#define TILE_WIDTH 16

// CUDA Error Checking Macro
#define CHECK_CUDA(call) {                                              \
    cudaError_t err = call;                                             \
    if(err != cudaSuccess) {                                            \
        std::cerr << "CUDA error: " << cudaGetErrorString(err)          \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl;  \
        exit(1);                                                        \
    }                                                                   \
}

// GPU Kernels

// Tiled Matrix Multiplication Kernel: C = A * B
// A: [M x K], B: [K x N], C: [M x N]
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
// Compute scaled dot-product scores for one attention head
// Q and K are stored in contiguous memory for all heads
__global__
void compute_scores_kernel(const float *Q, const float *K, float *d_scores,
                           int M, int N, int H, int d_head)
{
    int h = blockIdx.z;               // Head index
    int row = blockIdx.y * TILE_WIDTH + threadIdx.y; // Query token index
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x; // Key token index

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

// Softmax kernel: For each (query, head) pair, apply softmax over the keys
__global__
void softmax_kernel(float *d_scores, int M, int N, int H)
{
    // Each block is responsible for one (query token, head) pair
    int idx = blockIdx.x;           // idx = i * H + h, where i is query index and h is head
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
    // Reduction to find maximum
    for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
        if (t < stride)
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
    // Reduction to sum up exponentials
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

// Compute weighted sum over the values using the attention scores to form the output of one head
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

// Element-wise Addition Kernel: C = A + B
__global__
void add_kernel(const float *A, const float *B, float *C, int total_elements)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < total_elements)
        C[idx] = A[idx] + B[idx];
}

// ReLU Activation Kernel
__global__
void relu_kernel(float *A, int total_elements)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < total_elements && A[idx] < 0)
        A[idx] = 0;
}

// Layer Normalization Kernel
// Each block works on one row (token) of length N = d_model
__global__
void layer_norm_kernel(const float* input, float* output, int M, int N, float epsilon)
{
    int row = blockIdx.x;  // one block per row (token)
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
            if(tid < stride)
                shmem[tid] += shmem[tid + stride];
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
            if(tid < stride)
                shmem[tid] += shmem[tid + stride];
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

// CPU Implementation

// Matrix multiplication: C = A * B, where A: [M x K] and B: [K x N]
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

// Multi-head attention
// For each head, compute scaled dot product attention
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
                    float q_val = Q[i * (H * d_head) + h*d_head + k];
                    float k_val = K[j * (H * d_head) + h*d_head + k];
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
                    float v_val = V[j * (H * d_head) + h*d_head + k];
                    val += scores[j] * v_val;
                }
                O[i * (H * d_head) + h*d_head + k] = val;
            }
            delete[] scores;
        }
    }
}

// Element-wise addition: C = A + B
void cpu_add(const float *A, const float *B, float *C, int total_elements) {
    for (int i = 0; i < total_elements; i++){
        C[i] = A[i] + B[i];
    }
}

// ReLU activation
void cpu_relu(float *A, int total_elements) {
    for (int i = 0; i < total_elements; i++){
        if (A[i] < 0)
            A[i] = 0;
    }
}

// Layer Normalization: per row
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

    // Host Allocations
    size_t input_size    = SEQ_LEN * d_model * sizeof(float);
    size_t weight_size   = d_model * d_model * sizeof(float);    // for Wq, Wk, Wv, Wo (each)
    size_t ffn1_size     = d_model * d_ff * sizeof(float);        // for feed-forward layer 1

    // Input token embeddings and weights (initialized with sample values)
    float *h_X   = new float[SEQ_LEN * d_model];  // Input token embeddings: [SEQ_LEN, d_model]
    float *h_Wq  = new float[d_model * d_model];
    float *h_Wk  = new float[d_model * d_model];
    float *h_Wv  = new float[d_model * d_model];
    float *h_Wo  = new float[d_model * d_model];
    float *h_W1  = new float[d_model * d_ff];
    float *h_W2  = new float[d_ff * d_model];

    // Initialize input and weights with simple values for reproducibility
    for (int i = 0; i < SEQ_LEN * d_model; i++){
        h_X[i] = static_cast<float>((i + 1) % 100) / 100.0f;
    }
    for (int i = 0; i < d_model * d_model; i++){
        h_Wq[i] = static_cast<float>((i + 2) % 100) / 100.0f;
        h_Wk[i] = static_cast<float>((i + 3) % 100) / 100.0f;
        h_Wv[i] = static_cast<float>((i + 4) % 100) / 100.0f;
        h_Wo[i] = static_cast<float>((i + 5) % 100) / 100.0f;
    }
    for (int i = 0; i < d_model * d_ff; i++){
        h_W1[i] = static_cast<float>((i + 6) % 100) / 100.0f;
    }
    for (int i = 0; i < d_ff * d_model; i++){
        h_W2[i] = static_cast<float>((i + 7) % 100) / 100.0f;
    }

    // Intermediate CPU arrays for transformer encoder
    float *cpu_Q     = new float[SEQ_LEN * d_model];
    float *cpu_K     = new float[SEQ_LEN * d_model];
    float *cpu_V     = new float[SEQ_LEN * d_model];
    float *cpu_attn  = new float[SEQ_LEN * d_model];  // Output from multi-head attention
    float *cpu_MHA   = new float[SEQ_LEN * d_model];  // After final projection in attention branch
    float *cpu_add1  = new float[SEQ_LEN * d_model];  // Residual connection 1
    float *cpu_ln1   = new float[SEQ_LEN * d_model];  // After layer norm 1
    float *cpu_ffn1  = new float[SEQ_LEN * d_ff];
    float *cpu_ffn2  = new float[SEQ_LEN * d_model];
    float *cpu_add2  = new float[SEQ_LEN * d_model];
    float *cpu_ln2   = new float[SEQ_LEN * d_model];

    
    // CPU Inference Timing
    auto cpu_start = std::chrono::high_resolution_clock::now();
    
    // Linear projections for multi-head attention: Q = X*Wq, K = X*Wk, V = X*Wv
    cpu_matmul(h_X, h_Wq, cpu_Q, SEQ_LEN, d_model, d_model);
    cpu_matmul(h_X, h_Wk, cpu_K, SEQ_LEN, d_model, d_model);
    cpu_matmul(h_X, h_Wv, cpu_V, SEQ_LEN, d_model, d_model);

    // Multi-Head Attention on CPU
    cpu_multi_head_attention(cpu_Q, cpu_K, cpu_V, cpu_attn, SEQ_LEN, SEQ_LEN, H, d_head);
    // Final projection for attention branch: MHA = attn * Wo
    cpu_matmul(cpu_attn, h_Wo, cpu_MHA, SEQ_LEN, d_model, d_model);
    
    // Residual connection and first layer normalization: ln1 = LayerNorm(X + MHA)
    cpu_add(h_X, cpu_MHA, cpu_add1, SEQ_LEN * d_model);
    cpu_layer_norm(cpu_add1, cpu_ln1, SEQ_LEN, d_model, 1e-5);

    // Feed-forward network
    cpu_matmul(cpu_ln1, h_W1, cpu_ffn1, SEQ_LEN, d_model, d_ff);
    cpu_relu(cpu_ffn1, SEQ_LEN * d_ff);
    cpu_matmul(cpu_ffn1, h_W2, cpu_ffn2, SEQ_LEN, d_ff, d_model);

    // Residual connection and second layer normalization: ln2 = LayerNorm(ln1 + ffn2)
    cpu_add(cpu_ln1, cpu_ffn2, cpu_add2, SEQ_LEN * d_model);
    cpu_layer_norm(cpu_add2, cpu_ln2, SEQ_LEN, d_model, 1e-5);

    auto cpu_end = std::chrono::high_resolution_clock::now();
    double cpu_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(cpu_end - cpu_start).count();
    std::cout << "Total CPU inference time (ms): " << cpu_time_ms << " ms" << std::endl;

    // GPU Inference: Allocate Device Memory and copy Data

    float *d_X, *d_Wq, *d_Wk, *d_Wv, *d_Wo, *d_W1, *d_W2;
    float *d_Q, *d_K, *d_V, *d_scores, *d_attn, *d_MHA;
    float *d_add1, *d_ln1, *d_ffn1, *d_ffn2, *d_add2, *d_ln2;

    CHECK_CUDA(cudaMalloc(&d_X, input_size));
    CHECK_CUDA(cudaMalloc(&d_Wq, weight_size));
    CHECK_CUDA(cudaMalloc(&d_Wk, weight_size));
    CHECK_CUDA(cudaMalloc(&d_Wv, weight_size));
    CHECK_CUDA(cudaMalloc(&d_Wo, weight_size));
    CHECK_CUDA(cudaMalloc(&d_W1, d_model * d_ff * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_W2, d_ff * d_model * sizeof(float)));
    
    CHECK_CUDA(cudaMalloc(&d_Q, input_size));
    CHECK_CUDA(cudaMalloc(&d_K, input_size));
    CHECK_CUDA(cudaMalloc(&d_V, input_size));
    // Attention scores: shape [SEQ_LEN, SEQ_LEN, H]
    size_t scores_size = SEQ_LEN * SEQ_LEN * H * sizeof(float);
    CHECK_CUDA(cudaMalloc(&d_scores, scores_size));
    CHECK_CUDA(cudaMalloc(&d_attn, input_size));  // same shape as [SEQ_LEN, d_model]
    CHECK_CUDA(cudaMalloc(&d_MHA, input_size));
    CHECK_CUDA(cudaMalloc(&d_add1, input_size));
    CHECK_CUDA(cudaMalloc(&d_ln1, input_size));
    CHECK_CUDA(cudaMalloc(&d_ffn1, ffn1_size));
    CHECK_CUDA(cudaMalloc(&d_ffn2, input_size));
    CHECK_CUDA(cudaMalloc(&d_add2, input_size));
    CHECK_CUDA(cudaMalloc(&d_ln2, input_size));

    // Create CUDA events for timing GPU stages
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

    // Host-to-Device copying
    CHECK_CUDA(cudaEventRecord(start_h2d, 0));
    CHECK_CUDA(cudaMemcpy(d_X, h_X, input_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_Wq, h_Wq, weight_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_Wk, h_Wk, weight_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_Wv, h_Wv, weight_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_Wo, h_Wo, weight_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_W1, h_W1, d_model * d_ff * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_W2, h_W2, d_ff * d_model * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaEventRecord(stop_h2d, 0));
    CHECK_CUDA(cudaEventSynchronize(stop_h2d));
    float h2d_time_ms = 0;
    CHECK_CUDA(cudaEventElapsedTime(&h2d_time_ms, start_h2d, stop_h2d));

    CHECK_CUDA(cudaEventRecord(start_kernel, 0));

    // Linear Projections (Q, K, V)
    dim3 block(TILE_WIDTH, TILE_WIDTH);
    dim3 grid((d_model + TILE_WIDTH - 1) / TILE_WIDTH, (SEQ_LEN + TILE_WIDTH - 1) / TILE_WIDTH);
    matmul_kernel<<<grid, block>>>(d_X, d_Wq, d_Q, SEQ_LEN, d_model, d_model);
    matmul_kernel<<<grid, block>>>(d_X, d_Wk, d_K, SEQ_LEN, d_model, d_model);
    matmul_kernel<<<grid, block>>>(d_X, d_Wv, d_V, SEQ_LEN, d_model, d_model);

    // Multi-Head Attention
    // Compute scaled dot-product scores for each head
    // d_scores shape: [SEQ_LEN, SEQ_LEN, H]
    // Grid: one block covers TILE_WIDTH x TILE_WIDTH outputs for each head
    dim3 block_attn(TILE_WIDTH, TILE_WIDTH);
    dim3 grid_attn((SEQ_LEN + TILE_WIDTH - 1) / TILE_WIDTH,
                   (SEQ_LEN + TILE_WIDTH - 1) / TILE_WIDTH,
                   H);
    compute_scores_kernel<<<grid_attn, block_attn>>>(d_Q, d_K, d_scores, SEQ_LEN, SEQ_LEN, H, d_head);

    // Apply softmax along the keys dimension
    int threads_softmax = 256;
    int blocks_softmax = SEQ_LEN * H; // one block per (query, head) pair
    size_t shared_softmax = threads_softmax * sizeof(float);
    softmax_kernel<<<blocks_softmax, threads_softmax, shared_softmax>>>(d_scores, SEQ_LEN, SEQ_LEN, H);

    // Weighted sum: Compute attention output for each head.
    dim3 block_ws(TILE_WIDTH, TILE_WIDTH);
    dim3 grid_ws((d_head + TILE_WIDTH - 1) / TILE_WIDTH,
                 (SEQ_LEN + TILE_WIDTH - 1) / TILE_WIDTH,
                 H);
    weighted_sum_kernel<<<grid_ws, block_ws>>>(d_scores, d_V, d_attn, SEQ_LEN, SEQ_LEN, H, d_head);

    // Final projection for attention branch: d_MHA = d_attn * d_Wo
    dim3 grid_final((d_model + TILE_WIDTH - 1) / TILE_WIDTH, (SEQ_LEN + TILE_WIDTH - 1) / TILE_WIDTH);
    matmul_kernel<<<grid_final, block>>>(d_attn, d_Wo, d_MHA, SEQ_LEN, d_model, d_model);

    // GPU: Residual and First Layer Norm: d_ln1 = LayerNorm(d_X + d_MHA)
    // Element-wise addition
    int total_elements = SEQ_LEN * d_model;
    int threads_elem = 256;
    int blocks_elem = (total_elements + threads_elem - 1) / threads_elem;
    add_kernel<<<blocks_elem, threads_elem>>>(d_X, d_MHA, d_add1, total_elements);
    layer_norm_kernel<<<SEQ_LEN, 256, 256 * sizeof(float)>>>(d_add1, d_ln1, SEQ_LEN, d_model, 1e-5);

    // Feed-Forward Network (FFN)
    // d_ffn1 = d_ln1 * d_W1
    dim3 grid_ffn1((d_ff + TILE_WIDTH - 1) / TILE_WIDTH, (SEQ_LEN + TILE_WIDTH - 1) / TILE_WIDTH);
    matmul_kernel<<<grid_ffn1, block>>>(d_ln1, d_W1, d_ffn1, SEQ_LEN, d_model, d_ff);
    // ReLU Activation
    int total_ffn1 = SEQ_LEN * d_ff;
    int blocks_relu = (total_ffn1 + threads_elem - 1) / threads_elem;
    relu_kernel<<<blocks_relu, threads_elem>>>(d_ffn1, total_ffn1);
    // d_ffn2 = d_ffn1 * d_W2
    dim3 grid_ffn2((d_model + TILE_WIDTH - 1) / TILE_WIDTH, (SEQ_LEN + TILE_WIDTH - 1) / TILE_WIDTH);
    matmul_kernel<<<grid_ffn2, block>>>(d_ffn1, d_W2, d_ffn2, SEQ_LEN, d_ff, d_model);

    // Residual and second Layer Norm: d_ln2 = LayerNorm(d_ln1 + d_ffn2)
    add_kernel<<<blocks_elem, threads_elem>>>(d_ln1, d_ffn2, d_add2, total_elements);
    layer_norm_kernel<<<SEQ_LEN, 256, 256 * sizeof(float)>>>(d_add2, d_ln2, SEQ_LEN, d_model, 1e-5);

    CHECK_CUDA(cudaEventRecord(stop_kernel, 0));
    CHECK_CUDA(cudaEventSynchronize(stop_kernel));
    float kernel_time_ms = 0;
    CHECK_CUDA(cudaEventElapsedTime(&kernel_time_ms, start_kernel, stop_kernel));

    // Device-to-Host Copying: Get Final Output
    float *h_final_gpu = new float[SEQ_LEN * d_model];
    CHECK_CUDA(cudaEventRecord(start_d2h, 0));
    CHECK_CUDA(cudaMemcpy(h_final_gpu, d_ln2, input_size, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaEventRecord(stop_d2h, 0));
    CHECK_CUDA(cudaEventSynchronize(stop_d2h));
    float d2h_time_ms = 0;
    CHECK_CUDA(cudaEventElapsedTime(&d2h_time_ms, start_d2h, stop_d2h));

    CHECK_CUDA(cudaEventRecord(stop_total, 0));
    CHECK_CUDA(cudaEventSynchronize(stop_total));
    float total_gpu_time_ms = 0;
    CHECK_CUDA(cudaEventElapsedTime(&total_gpu_time_ms, start_total, stop_total));

    // Print GPU timings
    std::cout << "GPU Timings (ms):" << std::endl;
    std::cout << "  Host-to-Device copy time: " << h2d_time_ms << std::endl;
    std::cout << "  Kernel execution time:    " << kernel_time_ms << std::endl;
    std::cout << "  Device-to-Host copy time: " << d2h_time_ms << std::endl;
    std::cout << "  Total GPU inference time: " << total_gpu_time_ms << std::endl;

    // Cleanup: Free CPU and GPU Memory, Destroy CUDA Events
    delete[] h_X;
    delete[] h_Wq;
    delete[] h_Wk;
    delete[] h_Wv;
    delete[] h_Wo;
    delete[] h_W1;
    delete[] h_W2;
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
    delete[] h_final_gpu;

    cudaFree(d_X);
    cudaFree(d_Wq);
    cudaFree(d_Wk);
    cudaFree(d_Wv);
    cudaFree(d_Wo);
    cudaFree(d_W1);
    cudaFree(d_W2);
    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_scores);
    cudaFree(d_attn);
    cudaFree(d_MHA);
    cudaFree(d_add1);
    cudaFree(d_ln1);
    cudaFree(d_ffn1);
    cudaFree(d_ffn2);
    cudaFree(d_add2);
    cudaFree(d_ln2);

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