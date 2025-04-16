#include <iostream>
#include <cmath>
#include <chrono>
#include <cuda_runtime.h>

// Parameters
const int NUM_VIS    = 1024;       // number of visual tokens (e.g. image patches)
const int NUM_TEXT   = 256;        // number of text tokens from language encoder
const int d_model    = 256;        // common model dimension
const int H          = 8;          // number of attention heads
const int d_head     = (d_model / H);  // dimension per head (256/8 = 32)
const int TILE_WIDTH = 16;

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

// Tiled matrix multiplication
// Computes C = A * B for A [M x K] and B [K x N]
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
        for (int i = 0; i < TILE_WIDTH; i++)
            sum += ds_A[threadIdx.y][i] * ds_B[i][threadIdx.x];
        __syncthreads();
    }
    if (row < M && col < N)
        C[row * N + col] = sum;
}

// Scaled dot-product scores
// Q: [NUM_VIS, H*d_head], K: [NUM_TEXT, H*d_head]
// Output d_scores: [NUM_VIS, NUM_TEXT, H]
__global__
void compute_scores_kernel(const float *Q, const float *K, float *d_scores,
                           int M, int N, int H, int d_head)
{
    int h = blockIdx.z; // head index
    int row = blockIdx.y * TILE_WIDTH + threadIdx.y; // visual token (query) index
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x; // text token (key) index

    __shared__ float sQ[TILE_WIDTH][TILE_WIDTH];
    __shared__ float sK[TILE_WIDTH][TILE_WIDTH];

    float val = 0.0f;
    for (int t = 0; t < (d_head + TILE_WIDTH - 1) / TILE_WIDTH; t++) {
        int idxQ = t * TILE_WIDTH + threadIdx.x;
        int idxK = t * TILE_WIDTH + threadIdx.y;
        if (row < M && idxQ < d_head)
            sQ[threadIdx.y][threadIdx.x] = Q[row * (H * d_head) + h * d_head + idxQ];
        else
            sQ[threadIdx.y][threadIdx.x] = 0.0f;
        if (col < N && idxK < d_head)
            sK[threadIdx.y][threadIdx.x] = K[col * (H * d_head) + h * d_head + idxK];
        else
            sK[threadIdx.y][threadIdx.x] = 0.0f;
        __syncthreads();
        for (int i = 0; i < TILE_WIDTH; i++)
            val += sQ[threadIdx.y][i] * sK[i][threadIdx.x];
        __syncthreads();
    }
    val = val / sqrtf((float)d_head);
    if (row < M && col < N)
        d_scores[((row * N) + col) * H + h] = val;
}

// Softmax
// Applies softmax (row-wise) over the keys dimension for each (visual token, head) pair
__global__
void softmax_kernel(float *d_scores, int M, int N, int H)
{
    int idx = blockIdx.x;  // each block corresponds to one (i, h) pair; i.e. idx = i*H + h
    int i = idx / H;
    int h = idx % H;
    int t = threadIdx.x;
    extern __shared__ float shmem[];

    float thread_max = -1e20f;
    for (int j = t; j < N; j += blockDim.x) {
        float val = d_scores[((i * N) + j) * H + h];
        if(val > thread_max)
            thread_max = val;
    }
    shmem[t] = thread_max;
    __syncthreads();
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

// Weighted sum
// Computes for each visual token and head the weighted sum over text values
// V_text: [NUM_TEXT, H*d_head]
// Output O: [NUM_VIS, H*d_head]
__global__
void weighted_sum_kernel(const float *d_scores, const float *V_text, float *O,
                         int M, int N, int H, int d_head)
{
    int h = blockIdx.z; // head index
    int row = blockIdx.y * TILE_WIDTH + threadIdx.y; // visual token index
    int k = blockIdx.x * TILE_WIDTH + threadIdx.x;  // within-head dimension index
    if (row < M && k < d_head) {
        float sum_val = 0.0f;
        for (int j = 0; j < N; j++) {
            float score = d_scores[((row * N) + j) * H + h];
            float v_val = V_text[j * (H * d_head) + h * d_head + k];
            sum_val += score * v_val;
        }
        O[row * (H * d_head) + h * d_head + k] = sum_val;
    }
}

// Element-wise addition
__global__
void add_kernel(const float *A, const float *B, float *C, int total_elements)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < total_elements)
        C[idx] = A[idx] + B[idx];
}

// ReLU activation
__global__
void relu_kernel(float *A, int total_elements)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < total_elements && A[idx] < 0)
        A[idx] = 0;
}

// Layer normalization
// Each block processes one token (one row of length N, where N == d_model)
__global__
void layer_norm_kernel(const float* input, float* output, int M, int N, float epsilon)
{
    int row = blockIdx.x;
    if (row < M) {
        extern __shared__ float shmem[];
        int tid = threadIdx.x;
        float sum = 0.0f;
        for (int j = tid; j < N; j += blockDim.x)
            sum += input[row * N + j];
        shmem[tid] = sum;
        __syncthreads();
        for (int stride = blockDim.x/2; stride > 0; stride /= 2) {
            if(tid < stride)
                shmem[tid] += shmem[tid + stride];
            __syncthreads();
        }
        float mean = shmem[0] / N;
        __syncthreads();
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
        for (int j = tid; j < N; j += blockDim.x)
            output[row * N + j] = (input[row * N + j] - mean) * inv_std;
    }
}

// CPU Functions

// Matrix multiplication: C = A * B, where A [M x K] and B [K x N]
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

// Softmax: [rows x cols]
void cpu_softmax(float* matrix, int rows, int cols)
{
    for (int i = 0; i < rows; i++){
        float max_val = matrix[i * cols];
        for (int j = 1; j < cols; j++){
            float val = matrix[i*cols + j];
            if(val > max_val)
                max_val = val;
        }
        float sum_exp = 0.0f;
        for (int j = 0; j < cols; j++){
            matrix[i*cols + j] = exp(matrix[i*cols+j] - max_val);
            sum_exp += matrix[i*cols+j];
        }
        for (int j = 0; j < cols; j++){
            matrix[i*cols+j] /= sum_exp;
        }
    }
}

// Cross-modal attention
// Visual features (vis): [NUM_VIS, d_model] and text features (text): [NUM_TEXT, d_model]
// Weight matrices Wq, Wk, Wv, Wo are [d_model, d_model].
// The function splits Q, K, V into H heads (each with dimension d_head) and computes scaled dot-product attention per head
void cpu_cross_modal_attention(const float* vis, const float* text,
                               const float* Wq, const float* Wk, const float* Wv, const float* Wo,
                               float* output)
{
    int M = NUM_VIS;   // number of visual tokens
    int N = NUM_TEXT;  // number of text tokens

    // Allocate intermediate matrices
    float* Q = new float[M * d_model];
    float* K = new float[N * d_model];
    float* V = new float[N * d_model];
    // Linear projections
    cpu_matmul(vis, Wq, Q, M, d_model, d_model);
    cpu_matmul(text, Wk, K, N, d_model, d_model);
    cpu_matmul(text, Wv, V, N, d_model, d_model);
    
    // Multi-head attention: split Q, K, V into H heads
    float* attn_concat = new float[M * d_model]; // concatenated output
    for (int h = 0; h < H; h++){
        float* scores = new float[M * N];  // per-head scores [M, N]
        // Compute scores for head h.
        for (int i = 0; i < M; i++){
            for (int j = 0; j < N; j++){
                float dot = 0.0f;
                for (int k = 0; k < d_head; k++){
                    float q_val = Q[i * d_model + h*d_head + k];
                    float k_val = K[j * d_model + h*d_head + k];
                    dot += q_val * k_val;
                }
                dot /= sqrt((float)d_head);
                scores[i * N + j] = dot;
            }
        }
        // Apply softmax row-wise on the scores
        cpu_softmax(scores, M, N);
        // Compute weighted sum over V for head h
        for (int i = 0; i < M; i++){
            for (int k = 0; k < d_head; k++){
                float sum_val = 0.0f;
                for (int j = 0; j < N; j++){
                    float s = scores[i * N + j];
                    float v_val = V[j * d_model + h*d_head + k];
                    sum_val += s * v_val;
                }
                attn_concat[i * d_model + h*d_head + k] = sum_val;
            }
        }
        delete[] scores;
    }
    // Final projection: output = attn_concat * Wo
    cpu_matmul(attn_concat, Wo, output, M, d_model, d_model);

    delete[] Q;
    delete[] K;
    delete[] V;
    delete[] attn_concat;
}

int main()
{

    // Host allocations
    size_t vis_size  = NUM_VIS * d_model * sizeof(float);
    size_t text_size = NUM_TEXT * d_model * sizeof(float);
    size_t weight_size = d_model * d_model * sizeof(float);

    float* h_vis  = new float[NUM_VIS * d_model];   // Visual features
    float* h_text = new float[NUM_TEXT * d_model];    // Text features

    // Weight matrices for cross-modal attention
    float* h_Wq = new float[d_model * d_model];
    float* h_Wk = new float[d_model * d_model];
    float* h_Wv = new float[d_model * d_model];
    float* h_Wo = new float[d_model * d_model];

    // Initialize visual, text features and weights with random values
    for (int i = 0; i < NUM_VIS * d_model; i++){
        h_vis[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    for (int i = 0; i < NUM_TEXT * d_model; i++){
        h_text[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    for (int i = 0; i < d_model * d_model; i++){
        h_Wq[i] = static_cast<float>(rand()) / RAND_MAX;
        h_Wk[i] = static_cast<float>(rand()) / RAND_MAX;
        h_Wv[i] = static_cast<float>(rand()) / RAND_MAX;
        h_Wo[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    float* h_output_cpu = new float[NUM_VIS * d_model];
    float* h_output_gpu = new float[NUM_VIS * d_model];

    // CPU Inference Timing
    auto cpu_start = std::chrono::high_resolution_clock::now();
    cpu_cross_modal_attention(h_vis, h_text, h_Wq, h_Wk, h_Wv, h_Wo, h_output_cpu);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    double cpu_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(cpu_end - cpu_start).count();
    std::cout << "Total CPU inference time (ms): " << cpu_time_ms << std::endl;

    // GPU Inference

    // Allocate device memory and copy data
    float *d_vis, *d_text, *d_Wq, *d_Wk, *d_Wv, *d_Wo;
    float *d_Q, *d_K, *d_V, *d_scores, *d_attn, *d_output;
    CHECK_CUDA(cudaMalloc(&d_vis, vis_size));
    CHECK_CUDA(cudaMalloc(&d_text, text_size));
    CHECK_CUDA(cudaMalloc(&d_Wq, weight_size));
    CHECK_CUDA(cudaMalloc(&d_Wk, weight_size));
    CHECK_CUDA(cudaMalloc(&d_Wv, weight_size));
    CHECK_CUDA(cudaMalloc(&d_Wo, weight_size));
    // Q from visual features is [NUM_VIS, d_model]; K and V from text are [NUM_TEXT, d_model]
    CHECK_CUDA(cudaMalloc(&d_Q, vis_size));
    CHECK_CUDA(cudaMalloc(&d_K, text_size));
    CHECK_CUDA(cudaMalloc(&d_V, text_size));
    size_t scores_bytes = NUM_VIS * NUM_TEXT * H * sizeof(float);
    CHECK_CUDA(cudaMalloc(&d_scores, scores_bytes));
    CHECK_CUDA(cudaMalloc(&d_attn, vis_size));   // Attention output: [NUM_VIS, d_model]
    CHECK_CUDA(cudaMalloc(&d_output, vis_size));   // Final output

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
    // Host-to-Device copying
    CHECK_CUDA(cudaEventRecord(start_h2d, 0));
    CHECK_CUDA(cudaMemcpy(d_vis, h_vis, vis_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_text, h_text, text_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_Wq, h_Wq, weight_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_Wk, h_Wk, weight_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_Wv, h_Wv, weight_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_Wo, h_Wo, weight_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaEventRecord(stop_h2d, 0));
    CHECK_CUDA(cudaEventSynchronize(stop_h2d));
    float time_h2d = 0;
    CHECK_CUDA(cudaEventElapsedTime(&time_h2d, start_h2d, stop_h2d));

    CHECK_CUDA(cudaEventRecord(start_kernel, 0));
    
    // Linear projections
    // Q = vis * Wq
    dim3 block(TILE_WIDTH, TILE_WIDTH);
    dim3 grid_vis((d_model + TILE_WIDTH - 1) / TILE_WIDTH, (NUM_VIS + TILE_WIDTH - 1) / TILE_WIDTH);
    matmul_kernel<<<grid_vis, block>>>(d_vis, d_Wq, d_Q, NUM_VIS, d_model, d_model);
    // K = text * Wk and V = text * Wv
    dim3 grid_text((d_model + TILE_WIDTH - 1) / TILE_WIDTH, (NUM_TEXT + TILE_WIDTH - 1) / TILE_WIDTH);
    matmul_kernel<<<grid_text, block>>>(d_text, d_Wk, d_K, NUM_TEXT, d_model, d_model);
    matmul_kernel<<<grid_text, block>>>(d_text, d_Wv, d_V, NUM_TEXT, d_model, d_model);

    // Cross-modal attention
    // Compute scores using Q (visual) and K (text)
    dim3 grid_scores((NUM_TEXT + TILE_WIDTH - 1) / TILE_WIDTH,
                     (NUM_VIS + TILE_WIDTH - 1) / TILE_WIDTH,
                     H);
    compute_scores_kernel<<<grid_scores, block>>>(d_Q, d_K, d_scores, NUM_VIS, NUM_TEXT, H, d_head);
    // Softmax over keys for each (visual token, head) pair
    int softmax_threads = 256;
    int softmax_blocks = NUM_VIS * H; // one block per (visual token, head)
    size_t softmax_shared = softmax_threads * sizeof(float);
    softmax_kernel<<<softmax_blocks, softmax_threads, softmax_shared>>>(d_scores, NUM_VIS, NUM_TEXT, H);
    // Weighted sum: compute attention output
    dim3 grid_ws((d_head + TILE_WIDTH - 1) / TILE_WIDTH, (NUM_VIS + TILE_WIDTH - 1) / TILE_WIDTH, H);
    weighted_sum_kernel<<<grid_ws, block>>>(d_scores, d_V, d_attn, NUM_VIS, NUM_TEXT, H, d_head);
    // Final projection: output = attn * Wo
    dim3 grid_final((d_model + TILE_WIDTH - 1) / TILE_WIDTH, (NUM_VIS + TILE_WIDTH - 1) / TILE_WIDTH);
    matmul_kernel<<<grid_final, block>>>(d_attn, d_Wo, d_output, NUM_VIS, d_model, d_model);

    CHECK_CUDA(cudaEventRecord(stop_kernel, 0));
    CHECK_CUDA(cudaEventSynchronize(stop_kernel));
    float time_kernel = 0;
    CHECK_CUDA(cudaEventElapsedTime(&time_kernel, start_kernel, stop_kernel));

    // Device-to-Host copy
    CHECK_CUDA(cudaEventRecord(start_d2h, 0));
    CHECK_CUDA(cudaMemcpy(h_output_gpu, d_output, vis_size, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaEventRecord(stop_d2h, 0));
    CHECK_CUDA(cudaEventSynchronize(stop_d2h));
    float time_d2h = 0;
    CHECK_CUDA(cudaEventElapsedTime(&time_d2h, start_d2h, stop_d2h));

    CHECK_CUDA(cudaEventRecord(stop_total, 0));
    CHECK_CUDA(cudaEventSynchronize(stop_total));
    float total_gpu_time = 0;
    CHECK_CUDA(cudaEventElapsedTime(&total_gpu_time, start_total, stop_total));

    // Print GPU timings
    std::cout << "GPU Timings (ms):" << std::endl;
    std::cout << "  Host-to-Device copy time: " << time_h2d << std::endl;
    std::cout << "  Kernel execution time:    " << time_kernel << std::endl;
    std::cout << "  Device-to-Host copy time: " << time_d2h << std::endl;
    std::cout << "  Total GPU time:           " << total_gpu_time << std::endl;

    // Cleanup
    delete[] h_vis;
    delete[] h_text;
    delete[] h_Wq;
    delete[] h_Wk;
    delete[] h_Wv;
    delete[] h_Wo;
    delete[] h_output_cpu;
    delete[] h_output_gpu;

    cudaFree(d_vis);
    cudaFree(d_text);
    cudaFree(d_Wq);
    cudaFree(d_Wk);
    cudaFree(d_Wv);
    cudaFree(d_Wo);
    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_scores);
    cudaFree(d_attn);
    cudaFree(d_output);

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