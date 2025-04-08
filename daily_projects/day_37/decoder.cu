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

// Matrix multiplication kernel (for linear projections)
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

// Multi-head attention kernel (computes scaled dot-product scores)
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
    int idx = blockIdx.x; // one block per (query, head) pair; total blocks = M*H
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

// Weighted sum kernel: computing attention output
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

// Matrix multiplication
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

// Multi-Head Attention (masking is omitted)
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

// Normalization
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
    const int M = 512;           // number of tokens (sequence length)
    const int d_total = 256;     // model (hidden) dimension
    const int H = 4;             // number of attention heads
    const int d_head = d_total / H;
    const int d_ff = 4 * d_total; // feed-forward inner dimension

    size_t input_size   = M * d_total * sizeof(float);
    size_t weight_size  = d_total * d_total * sizeof(float);
    size_t proj_size    = M * d_total * sizeof(float);
    size_t scores_size  = M * M * H * sizeof(float);
    size_t ffn1_size    = M * d_ff * sizeof(float);

    float *h_Y = new float[M * d_total];      // Decoder input (target sequence)
    float *h_enc = new float[M * d_total];      // Simulated encoder output
    for (int i = 0; i < M * d_total; i++){
        h_Y[i]   = static_cast<float>((i + 1) % 100) / 100.0f;
        h_enc[i] = static_cast<float>((i + 2) % 100) / 100.0f;
    }


    // Allocate and initialize decoder weights
    // For self-attention
    float *h_Wq_self = new float[d_total * d_total];
    float *h_Wk_self = new float[d_total * d_total];
    float *h_Wv_self = new float[d_total * d_total];
    float *h_Wo_self = new float[d_total * d_total];
    // For encoder-decoder attention
    float *h_Wq_encdec = new float[d_total * d_total];
    float *h_Wk_encdec = new float[d_total * d_total];
    float *h_Wv_encdec = new float[d_total * d_total];
    float *h_Wo_encdec = new float[d_total * d_total];
    // For feed-forward network
    float *h_W1_dec = new float[d_total * d_ff];
    float *h_W2_dec = new float[d_ff * d_total];

    for (int i = 0; i < d_total * d_total; i++){
        h_Wq_self[i]   = static_cast<float>((i + 3) % 100) / 100.0f;
        h_Wk_self[i]   = static_cast<float>((i + 4) % 100) / 100.0f;
        h_Wv_self[i]   = static_cast<float>((i + 5) % 100) / 100.0f;
        h_Wo_self[i]   = static_cast<float>((i + 6) % 100) / 100.0f;
        h_Wq_encdec[i] = static_cast<float>((i + 7) % 100) / 100.0f;
        h_Wk_encdec[i] = static_cast<float>((i + 8) % 100) / 100.0f;
        h_Wv_encdec[i] = static_cast<float>((i + 9) % 100) / 100.0f;
        h_Wo_encdec[i] = static_cast<float>((i + 10) % 100) / 100.0f;
    }
    for (int i = 0; i < d_total * d_ff; i++){
        h_W1_dec[i] = static_cast<float>((i + 11) % 100) / 100.0f;
    }
    for (int i = 0; i < d_ff * d_total; i++){
        h_W2_dec[i] = static_cast<float>((i + 12) % 100) / 100.0f;
    }
    
    // CPU Decoder
    auto cpu_start = std::chrono::high_resolution_clock::now();

    // Self-Attention Block
    float *cpu_Q_self   = new float[M * d_total];
    float *cpu_K_self   = new float[M * d_total];
    float *cpu_V_self   = new float[M * d_total];
    float *cpu_attn_self= new float[M * d_total];  // Multi-head self-attention output (before final projection)
    float *cpu_MHA_self = new float[M * d_total];  // After final projection

    // Linear projections from decoder input (h_Y)
    cpu_matmul(h_Y, h_Wq_self, cpu_Q_self, M, d_total, d_total);
    cpu_matmul(h_Y, h_Wk_self, cpu_K_self, M, d_total, d_total);
    cpu_matmul(h_Y, h_Wv_self, cpu_V_self, M, d_total, d_total);

    // Multi-head self-attention (masking is omitted)
    cpu_multi_head_attention(cpu_Q_self, cpu_K_self, cpu_V_self, cpu_attn_self, M, M, H, d_head);

    // Final projection for self-attention block
    cpu_matmul(cpu_attn_self, h_Wo_self, cpu_MHA_self, M, d_total, d_total);

    // Residual add and layer normalization
    float *cpu_add_self = new float[M * d_total];
    float *cpu_ln_self  = new float[M * d_total];
    cpu_add(h_Y, cpu_MHA_self, cpu_add_self, M * d_total);
    cpu_layer_norm(cpu_add_self, cpu_ln_self, M, d_total, 1e-5);

    // Encoder-Decoder Attention Block
    float *cpu_Q_encdec   = new float[M * d_total];
    float *cpu_K_encdec   = new float[M * d_total];
    float *cpu_V_encdec   = new float[M * d_total];
    float *cpu_attn_encdec= new float[M * d_total];
    float *cpu_MHA_encdec = new float[M * d_total];

    // For encoder-decoder attention, query comes from the output of self-attention
    // and keys/values come from the simulated encoder output (h_enc).
    cpu_matmul(cpu_ln_self, h_Wq_encdec, cpu_Q_encdec, M, d_total, d_total);
    cpu_matmul(h_enc,     h_Wk_encdec, cpu_K_encdec, M, d_total, d_total);
    cpu_matmul(h_enc,     h_Wv_encdec, cpu_V_encdec, M, d_total, d_total);

    cpu_multi_head_attention(cpu_Q_encdec, cpu_K_encdec, cpu_V_encdec, cpu_attn_encdec, M, M, H, d_head);

    // Final projection for encoder-decoder attention
    cpu_matmul(cpu_attn_encdec, h_Wo_encdec, cpu_MHA_encdec, M, d_total, d_total);

    // Residual add and layer normalization
    float *cpu_add_encdec = new float[M * d_total];
    float *cpu_ln_encdec  = new float[M * d_total];
    cpu_add(cpu_ln_self, cpu_MHA_encdec, cpu_add_encdec, M * d_total);
    cpu_layer_norm(cpu_add_encdec, cpu_ln_encdec, M, d_total, 1e-5);

    // Feed-Forward Network Block
    float *cpu_ffn1_dec  = new float[M * d_ff];
    float *cpu_ffn2_dec  = new float[M * d_total];
    float *cpu_add_ffn_dec = new float[M * d_total];
    float *cpu_ln_dec    = new float[M * d_total];

    cpu_matmul(cpu_ln_encdec, h_W1_dec, cpu_ffn1_dec, M, d_total, d_ff);
    cpu_relu(cpu_ffn1_dec, M * d_ff);
    cpu_matmul(cpu_ffn1_dec, h_W2_dec, cpu_ffn2_dec, M, d_ff, d_total);

    cpu_add(cpu_ln_encdec, cpu_ffn2_dec, cpu_add_ffn_dec, M * d_total);
    cpu_layer_norm(cpu_add_ffn_dec, cpu_ln_dec, M, d_total, 1e-5);

    auto cpu_end = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration_cast<std::chrono::milliseconds>(cpu_end - cpu_start).count();

    // GPU Decoder

    // Device allocations
    float *d_Y, *d_enc;
    float *d_Wq_self, *d_Wk_self, *d_Wv_self, *d_Wo_self;
    float *d_Q_self, *d_K_self, *d_V_self;
    float *d_scores_self, *d_O_self;
    float *d_attn_self;
    float *d_add_self, *d_ln_self;
    // For encoder-decoder attention
    float *d_Wq_encdec, *d_Wk_encdec, *d_Wv_encdec, *d_Wo_encdec;
    float *d_Q_encdec, *d_K_encdec, *d_V_encdec;
    float *d_scores_encdec, *d_O_encdec;
    float *d_attn_encdec;
    float *d_add_encdec, *d_ln_encdec;
    // For feed-forward network
    float *d_W1_dec, *d_W2_dec;
    float *d_ffn1_dec, *d_ffn2_dec;
    float *d_add_ffn_dec, *d_ln_dec;

    CHECK_CUDA(cudaMalloc(&d_Y, input_size));
    CHECK_CUDA(cudaMalloc(&d_enc, input_size));
    // Self-attention weights
    CHECK_CUDA(cudaMalloc(&d_Wq_self, weight_size));
    CHECK_CUDA(cudaMalloc(&d_Wk_self, weight_size));
    CHECK_CUDA(cudaMalloc(&d_Wv_self, weight_size));
    CHECK_CUDA(cudaMalloc(&d_Wo_self, weight_size));
    // Self-attention linear outputs
    CHECK_CUDA(cudaMalloc(&d_Q_self, proj_size));
    CHECK_CUDA(cudaMalloc(&d_K_self, proj_size));
    CHECK_CUDA(cudaMalloc(&d_V_self, proj_size));
    // Self-attention scores and output
    CHECK_CUDA(cudaMalloc(&d_scores_self, scores_size));
    CHECK_CUDA(cudaMalloc(&d_O_self, proj_size));
    CHECK_CUDA(cudaMalloc(&d_attn_self, proj_size));
    // Self-attention residual and norm buffers
    CHECK_CUDA(cudaMalloc(&d_add_self, proj_size));
    CHECK_CUDA(cudaMalloc(&d_ln_self, proj_size));
    // Encoder-decoder weights
    CHECK_CUDA(cudaMalloc(&d_Wq_encdec, weight_size));
    CHECK_CUDA(cudaMalloc(&d_Wk_encdec, weight_size));
    CHECK_CUDA(cudaMalloc(&d_Wv_encdec, weight_size));
    CHECK_CUDA(cudaMalloc(&d_Wo_encdec, weight_size));
    // Encoder-decoder linear outputs
    CHECK_CUDA(cudaMalloc(&d_Q_encdec, proj_size));
    CHECK_CUDA(cudaMalloc(&d_K_encdec, proj_size));
    CHECK_CUDA(cudaMalloc(&d_V_encdec, proj_size));
    // Encoder-decoder scores and output
    CHECK_CUDA(cudaMalloc(&d_scores_encdec, scores_size));
    CHECK_CUDA(cudaMalloc(&d_O_encdec, proj_size));
    CHECK_CUDA(cudaMalloc(&d_attn_encdec, proj_size));
    // Encoder-decoder residual and norm buffers
    CHECK_CUDA(cudaMalloc(&d_add_encdec, proj_size));
    CHECK_CUDA(cudaMalloc(&d_ln_encdec, proj_size));
    // Feed-forward weights
    CHECK_CUDA(cudaMalloc(&d_W1_dec, d_total * d_ff * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_W2_dec, d_ff * d_total * sizeof(float)));
    // Feed-forward buffers
    CHECK_CUDA(cudaMalloc(&d_ffn1_dec, ffn1_size));
    CHECK_CUDA(cudaMalloc(&d_ffn2_dec, proj_size));
    CHECK_CUDA(cudaMalloc(&d_add_ffn_dec, proj_size));
    CHECK_CUDA(cudaMalloc(&d_ln_dec, proj_size));

    // Create CUDA events for timing on GPU
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

    // Host-to-Device copy of inputs and weights
    CHECK_CUDA(cudaEventRecord(start_h2d, 0));
    CHECK_CUDA(cudaMemcpy(d_Y, h_Y, input_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_enc, h_enc, input_size, cudaMemcpyHostToDevice));
    // Self-attention weights
    CHECK_CUDA(cudaMemcpy(d_Wq_self, h_Wq_self, weight_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_Wk_self, h_Wk_self, weight_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_Wv_self, h_Wv_self, weight_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_Wo_self, h_Wo_self, weight_size, cudaMemcpyHostToDevice));
    // Encoder-decoder weights
    CHECK_CUDA(cudaMemcpy(d_Wq_encdec, h_Wq_encdec, weight_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_Wk_encdec, h_Wk_encdec, weight_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_Wv_encdec, h_Wv_encdec, weight_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_Wo_encdec, h_Wo_encdec, weight_size, cudaMemcpyHostToDevice));
    // Feed-forward weights
    CHECK_CUDA(cudaMemcpy(d_W1_dec, h_W1_dec, d_total*d_ff*sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_W2_dec, h_W2_dec, d_ff*d_total*sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaEventRecord(stop_h2d, 0));
    CHECK_CUDA(cudaEventSynchronize(stop_h2d));
    float h2d_ms = 0;
    CHECK_CUDA(cudaEventElapsedTime(&h2d_ms, start_h2d, stop_h2d));

    CHECK_CUDA(cudaEventRecord(start_kernel, 0));

    // Self-Attention Block
    dim3 block(TILE_WIDTH, TILE_WIDTH);
    dim3 grid((d_total + TILE_WIDTH - 1) / TILE_WIDTH, (M + TILE_WIDTH - 1) / TILE_WIDTH);
    // Linear projections for self-attention
    matmul_kernel<<<grid, block>>>(d_Y, d_Wq_self, d_Q_self, M, d_total, d_total);
    matmul_kernel<<<grid, block>>>(d_Y, d_Wk_self, d_K_self, M, d_total, d_total);
    matmul_kernel<<<grid, block>>>(d_Y, d_Wv_self, d_V_self, M, d_total, d_total);

    // Multi-head self-attention: compute scaled dot-product scores, softmax, and weighted sum.
    dim3 block_att(TILE_WIDTH, TILE_WIDTH);
    dim3 grid_att((M + TILE_WIDTH - 1) / TILE_WIDTH, (M + TILE_WIDTH - 1) / TILE_WIDTH, H);
    compute_scores_kernel<<<grid_att, block_att>>>(d_Q_self, d_K_self, d_scores_self, M, M, H, d_head);
    int threads = 256;
    int blocks = M * H;
    size_t shared_size = threads * sizeof(float);
    softmax_kernel<<<blocks, threads, shared_size>>>(d_scores_self, M, M, H);
    dim3 grid_ws((d_total + TILE_WIDTH - 1) / TILE_WIDTH, (M + TILE_WIDTH - 1) / TILE_WIDTH, H);
    weighted_sum_kernel<<<grid_ws, block_att>>>(d_scores_self, d_V_self, d_O_self, M, M, H, d_head);
    // Final projection for self-attention block
    dim3 grid_final((d_total + TILE_WIDTH - 1) / TILE_WIDTH, (M + TILE_WIDTH - 1) / TILE_WIDTH);
    matmul_kernel<<<grid_final, block>>>(d_O_self, d_Wo_self, d_attn_self, M, d_total, d_total);
    // Residual connection and layer normalization
    int threads_add = 256;
    int blocks_add = (M * d_total + threads_add - 1) / threads_add;
    add_kernel<<<blocks_add, threads_add>>>(d_Y, d_attn_self, d_add_self, M * d_total);
    int ln_threads = 256;
    layer_norm_kernel<<<M, ln_threads, ln_threads * sizeof(float)>>>(d_add_self, d_ln_self, M, d_total, 1e-5);

    // Encoder-Decoder Attention Block
    // Linear projections: query from d_ln_self, key/value from encoder output d_enc
    matmul_kernel<<<grid, block>>>(d_ln_self, d_Wq_encdec, d_Q_encdec, M, d_total, d_total);
    matmul_kernel<<<grid, block>>>(d_enc, d_Wk_encdec, d_K_encdec, M, d_total, d_total);
    matmul_kernel<<<grid, block>>>(d_enc, d_Wv_encdec, d_V_encdec, M, d_total, d_total);
    dim3 grid_att2((M + TILE_WIDTH - 1) / TILE_WIDTH, (M + TILE_WIDTH - 1) / TILE_WIDTH, H);
    compute_scores_kernel<<<grid_att2, block_att>>>(d_Q_encdec, d_K_encdec, d_scores_encdec, M, M, H, d_head);
    softmax_kernel<<<blocks, threads, shared_size>>>(d_scores_encdec, M, M, H);
    dim3 grid_ws2((d_total + TILE_WIDTH - 1) / TILE_WIDTH, (M + TILE_WIDTH - 1) / TILE_WIDTH, H);
    weighted_sum_kernel<<<grid_ws2, block_att>>>(d_scores_encdec, d_V_encdec, d_O_encdec, M, M, H, d_head);
    matmul_kernel<<<grid_final, block>>>(d_O_encdec, d_Wo_encdec, d_attn_encdec, M, d_total, d_total);
    add_kernel<<<blocks_add, threads_add>>>(d_ln_self, d_attn_encdec, d_add_encdec, M * d_total);
    layer_norm_kernel<<<M, ln_threads, ln_threads * sizeof(float)>>>(d_add_encdec, d_ln_encdec, M, d_total, 1e-5);

    // Feed-Forward Network Block
    dim3 grid_ffn((d_ff + TILE_WIDTH - 1) / TILE_WIDTH, (M + TILE_WIDTH - 1) / TILE_WIDTH);
    matmul_kernel<<<grid_ffn, block>>>(d_ln_encdec, d_W1_dec, d_ffn1_dec, M, d_total, d_ff);
    int total_ffn1 = M * d_ff;
    int blocks_relu = (total_ffn1 + threads_add - 1) / threads_add;
    relu_kernel<<<blocks_relu, threads_add>>>(d_ffn1_dec, total_ffn1);
    matmul_kernel<<<grid_final, block>>>(d_ffn1_dec, d_W2_dec, d_ffn2_dec, M, d_ff, d_total);
    add_kernel<<<blocks_add, threads_add>>>(d_ln_encdec, d_ffn2_dec, d_add_ffn_dec, M * d_total);
    layer_norm_kernel<<<M, ln_threads, ln_threads * sizeof(float)>>>(d_add_ffn_dec, d_ln_dec, M, d_total, 1e-5);

    CHECK_CUDA(cudaEventRecord(stop_kernel, 0));
    CHECK_CUDA(cudaEventSynchronize(stop_kernel));
    float kernel_ms = 0;
    CHECK_CUDA(cudaEventElapsedTime(&kernel_ms, start_kernel, stop_kernel));

    // Device-to-Host copy of the final decoder output
    float *h_final_gpu_dec = new float[M * d_total];
    CHECK_CUDA(cudaEventRecord(start_d2h, 0));
    CHECK_CUDA(cudaMemcpy(h_final_gpu_dec, d_ln_dec, input_size, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaEventRecord(stop_d2h, 0));
    CHECK_CUDA(cudaEventSynchronize(stop_d2h));
    float d2h_ms = 0;
    CHECK_CUDA(cudaEventElapsedTime(&d2h_ms, start_d2h, stop_d2h));

    CHECK_CUDA(cudaEventRecord(stop_total, 0));
    CHECK_CUDA(cudaEventSynchronize(stop_total));
    float total_gpu_ms = 0;
    CHECK_CUDA(cudaEventElapsedTime(&total_gpu_ms, start_total, stop_total));

    // Print timings for the decoder
    std::cout << "CPU Decoder Total Time: " << cpu_ms << " ms\n";
    std::cout << "GPU Decoder Host-to-Device Copy Time: " << h2d_ms << " ms\n";
    std::cout << "GPU Decoder Kernel Execution Time: " << kernel_ms << " ms\n";
    std::cout << "GPU Decoder Device-to-Host Copy Time: " << d2h_ms << " ms\n";
    std::cout << "Total GPU Decoder Time: " << total_gpu_ms << " ms\n";
    
    // Cleanup CPU memory
    delete[] h_Y;
    delete[] h_enc;
    delete[] h_Wq_self; delete[] h_Wk_self; delete[] h_Wv_self; delete[] h_Wo_self;
    delete[] h_Wq_encdec; delete[] h_Wk_encdec; delete[] h_Wv_encdec; delete[] h_Wo_encdec;
    delete[] h_W1_dec; delete[] h_W2_dec;
    delete[] cpu_Q_self; delete[] cpu_K_self; delete[] cpu_V_self;
    delete[] cpu_attn_self; delete[] cpu_MHA_self;
    delete[] cpu_add_self; delete[] cpu_ln_self;
    delete[] cpu_Q_encdec; delete[] cpu_K_encdec; delete[] cpu_V_encdec;
    delete[] cpu_attn_encdec; delete[] cpu_MHA_encdec;
    delete[] cpu_add_encdec; delete[] cpu_ln_encdec;
    delete[] cpu_ffn1_dec; delete[] cpu_ffn2_dec;
    delete[] cpu_add_ffn_dec; delete[] cpu_ln_dec;
    delete[] h_final_gpu_dec;

    // Cleanup GPU memory and events
    cudaFree(d_Y); cudaFree(d_enc);
    cudaFree(d_Wq_self); cudaFree(d_Wk_self); cudaFree(d_Wv_self); cudaFree(d_Wo_self);
    cudaFree(d_Q_self); cudaFree(d_K_self); cudaFree(d_V_self);
    cudaFree(d_scores_self); cudaFree(d_O_self);
    cudaFree(d_attn_self);
    cudaFree(d_add_self); cudaFree(d_ln_self);
    cudaFree(d_Wq_encdec); cudaFree(d_Wk_encdec); cudaFree(d_Wv_encdec); cudaFree(d_Wo_encdec);
    cudaFree(d_Q_encdec); cudaFree(d_K_encdec); cudaFree(d_V_encdec);
    cudaFree(d_scores_encdec); cudaFree(d_O_encdec);
    cudaFree(d_attn_encdec);
    cudaFree(d_add_encdec); cudaFree(d_ln_encdec);
    cudaFree(d_W1_dec); cudaFree(d_W2_dec);
    cudaFree(d_ffn1_dec); cudaFree(d_ffn2_dec);
    cudaFree(d_add_ffn_dec); cudaFree(d_ln_dec);

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
