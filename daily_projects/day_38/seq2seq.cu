// seq2seq_integration.cu
// Integrated CUDA and CPU implementation for a simple encoder-decoder (seq2seq) model.
// This version uses transformer–like dimensions: 512 tokens, 256–dimensional model,
// 4 attention heads (with d_head = 256/4 = 64) and a feed-forward inner dimension of 1024.
// Both CPU and GPU inference paths are implemented and timed.

#include <iostream>
#include <sstream>
#include <vector>
#include <string>
#include <unordered_map>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cuda_runtime.h>

// Error checking macro
#define CHECK_CUDA(call) {                                 \
    cudaError_t err = call;                                \
    if(err != cudaSuccess) {                               \
        std::cerr << "CUDA error: " << cudaGetErrorString(err) \
                  << " at " << __FILE__ << ":" << __LINE__  \
                  << std::endl;                            \
        exit(1);                                           \
    }                                                      \
}

// Tokenizer and Embedding parameters
const int VOCAB_SIZE    = 4;
const int NUM_SENTENCES = 1;       // one sentence each for encoder and decoder
const int MAX_TOKENS    = 512;     // sequence length
const int EMBEDDING_DIM = 256;     // Also d_model

// Encoder/Decoder Parameters:
const int M = MAX_TOKENS;         // sequence length = 512
const int d_total = 256;          // model dimension
const int H = 4;                  // number of attention heads
const int d_head = d_total / H;   // 64
const int d_ff = 4 * d_total;     // feed-forward inner dimension = 1024

// GPU Kernels

// Embedding lookup kernel
__global__ void embedding_lookup_kernel(const int* token_ids, 
                                          const float* embedding_matrix, 
                                          float* output, 
                                          int total_tokens, 
                                          int embedding_dim) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx >= total_tokens) return;
    int token_id = token_ids[idx];
    for (int d = 0; d < embedding_dim; d++){
        if(token_id < 0 || token_id >= VOCAB_SIZE)
            output[idx * embedding_dim + d] = 0.0f;
        else
            output[idx * embedding_dim + d] = embedding_matrix[token_id * embedding_dim + d];
    }
}

// Positional encoding kernel
__global__ void positional_encoding_kernel(float* pe, int seq_len, int d_model) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int total = seq_len * d_model;
    if(idx >= total) return;
    
    int pos = idx / d_model;
    int j   = idx % d_model;
    float factor = (j % 2 == 0) ? (j / 2.0f) : ((j - 1) / 2.0f);
    float div_term = expf(-logf(10000.0f) * factor / d_model);
    float angle = pos * div_term;
    pe[idx] = (j % 2 == 0) ? sinf(angle) : cosf(angle);
}

// Tiled matrix multiplication kernel
#define TILE_WIDTH 16
__global__ void matmul_kernel(const float *A, const float *B, float *C,
                              int M_mat, int K, int N) {
    __shared__ float ds_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float ds_B[TILE_WIDTH][TILE_WIDTH];

    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;
    float sum = 0.0f;
    for (int t = 0; t < (K + TILE_WIDTH - 1) / TILE_WIDTH; t++) {
        if(row < M_mat && t * TILE_WIDTH + threadIdx.x < K)
            ds_A[threadIdx.y][threadIdx.x] = A[row * K + t * TILE_WIDTH + threadIdx.x];
        else
            ds_A[threadIdx.y][threadIdx.x] = 0.0f;
            
        if(col < N && t * TILE_WIDTH + threadIdx.y < K)
            ds_B[threadIdx.y][threadIdx.x] = B[(t * TILE_WIDTH + threadIdx.y) * N + col];
        else
            ds_B[threadIdx.y][threadIdx.x] = 0.0f;
            
        __syncthreads();
        for (int i = 0; i < TILE_WIDTH; i++){
            sum += ds_A[threadIdx.y][i] * ds_B[i][threadIdx.x];
        }
        __syncthreads();
    }
    if(row < M_mat && col < N)
        C[row * N + col] = sum;
}

// Compute attention scores kernel (multi-head attention)
__global__ void compute_scores_kernel(const float *Q, const float *K, float *d_scores,
                                        int M_val, int N_val, int H_val, int d_head) {
    int h = blockIdx.z; // head index
    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;

    __shared__ float sQ[TILE_WIDTH][TILE_WIDTH];
    __shared__ float sK[TILE_WIDTH][TILE_WIDTH];

    float val = 0.0f;
    for (int t = 0; t < (d_head + TILE_WIDTH - 1) / TILE_WIDTH; t++) {
        int indexQ = t * TILE_WIDTH + threadIdx.x;
        int indexK = t * TILE_WIDTH + threadIdx.y;
        if (row < M_val && indexQ < d_head)
            sQ[threadIdx.y][threadIdx.x] = Q[row * (H_val*d_head) + h * d_head + indexQ];
        else
            sQ[threadIdx.y][threadIdx.x] = 0.0f;
        if (col < N_val && indexK < d_head)
            sK[threadIdx.y][threadIdx.x] = K[col * (H_val*d_head) + h * d_head + indexK];
        else
            sK[threadIdx.y][threadIdx.x] = 0.0f;
        __syncthreads();
        for (int i = 0; i < TILE_WIDTH; i++){
            val += sQ[threadIdx.y][i] * sK[i][threadIdx.x];
        }
        __syncthreads();
    }
    val = val / sqrtf((float)d_head);
    if (row < M_val && col < N_val)
        d_scores[((row * N_val) + col) * H_val + h] = val;
}

// Softmax kernel (per row)
__global__ void softmax_kernel(float *d_scores, int M_val, int N_val, int H_val) {
    int idx = blockIdx.x; // one block per (i, h) pair
    int i = idx / H_val;
    int h = idx % H_val;
    int t = threadIdx.x;
    extern __shared__ float shmem[];
    float thread_max = -1e20f;
    for (int j = t; j < N_val; j += blockDim.x) {
        float s = d_scores[((i * N_val) + j) * H_val + h];
        if(s > thread_max)
            thread_max = s;
    }
    shmem[t] = thread_max;
    __syncthreads();
    for (int stride = blockDim.x/2; stride > 0; stride /= 2) {
        if(t < stride && (t + stride) < blockDim.x)
            shmem[t] = fmaxf(shmem[t], shmem[t+stride]);
        __syncthreads();
    }
    float max_val = shmem[0];
    __syncthreads();
    float sum_exp = 0.0f;
    for (int j = t; j < N_val; j += blockDim.x) {
        float exp_val = expf(d_scores[((i * N_val) + j) * H_val + h] - max_val);
        d_scores[((i * N_val) + j) * H_val + h] = exp_val;
        sum_exp += exp_val;
    }
    shmem[t] = sum_exp;
    __syncthreads();
    for (int stride = blockDim.x/2; stride > 0; stride /= 2) {
        if(t < stride)
            shmem[t] += shmem[t+stride];
        __syncthreads();
    }
    float total_exp = shmem[0];
    for (int j = t; j < N_val; j += blockDim.x) {
        d_scores[((i * N_val) + j) * H_val + h] /= total_exp;
    }
}

// Weighted sum kernel: computes attention output = scores * V
__global__ void weighted_sum_kernel(const float *d_scores, const float *V, float *O,
                                      int M_val, int N_val, int H_val, int d_head) {
    int h = blockIdx.z;
    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int k = blockIdx.x * TILE_WIDTH + threadIdx.x;
    if (row < M_val && k < d_head) {
        float sum_val = 0.0f;
        for (int j = 0; j < N_val; j++){
            float score = d_scores[((row * N_val) + j) * H_val + h];
            float v_val = V[j * (H_val * d_head) + h * d_head + k];
            sum_val += score * v_val;
        }
        O[row * (H_val * d_head) + h * d_head + k] = sum_val;
    }
}

// Element-wise addition kernel: C = A + B
__global__ void add_kernel(const float *A, const float *B, float *C, int total_elements) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx < total_elements)
        C[idx] = A[idx] + B[idx];
}

// ReLU activation kernel
__global__ void relu_kernel(float *A, int total_elements) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx < total_elements && A[idx] < 0)
        A[idx] = 0;
}

// Layer normalization kernel (per row normalization)
__global__ void layer_norm_kernel(const float* input, float* output, int M_val, int N_val, float epsilon) {
    int row = blockIdx.x; // one block per row
    if(row < M_val) {
        extern __shared__ float shmem[];
        int tid = threadIdx.x;
        float sum = 0.0f;
        for (int j = tid; j < N_val; j += blockDim.x)
            sum += input[row * N_val + j];
        shmem[tid] = sum;
        __syncthreads();
        for (int stride = blockDim.x/2; stride > 0; stride /= 2) {
            if(tid < stride)
                shmem[tid] += shmem[tid + stride];
            __syncthreads();
        }
        float mean = shmem[0] / N_val;
        __syncthreads();
        float var_sum = 0.0f;
        for (int j = tid; j < N_val; j += blockDim.x) {
            float diff = input[row * N_val + j] - mean;
            var_sum += diff * diff;
        }
        shmem[tid] = var_sum;
        __syncthreads();
        for (int stride = blockDim.x/2; stride > 0; stride /= 2) {
            if(tid < stride)
                shmem[tid] += shmem[tid + stride];
            __syncthreads();
        }
        float var = shmem[0] / N_val;
        float inv_std = rsqrtf(var + epsilon);
        for (int j = tid; j < N_val; j += blockDim.x) {
            int idx = row * N_val + j;
            output[idx] = (input[idx] - mean) * inv_std;
        }
    }
}


// CPU Implementation

// Tokenizer: splits sentences by whitespace and pads to MAX_TOKENS
void cpu_tokenizer(const std::vector<std::string>& sentences,
                   const std::unordered_map<std::string,int>& vocab,
                   std::vector<int>& token_ids) {
    int num_sentences = sentences.size();
    token_ids.assign(num_sentences * MAX_TOKENS, -1);
    for (int i = 0; i < num_sentences; i++) {
        std::istringstream iss(sentences[i]);
        std::string token;
        int token_idx = 0;
        while (iss >> token && token_idx < MAX_TOKENS) {
            if (vocab.find(token) != vocab.end())
                token_ids[i * MAX_TOKENS + token_idx] = vocab.at(token);
            else
                token_ids[i * MAX_TOKENS + token_idx] = -1;
            token_idx++;
        }
    }
}

// Embedding lookup
void cpu_embedding_lookup(const std::vector<int>& token_ids,
                          const std::vector<float>& embedding_matrix,
                          std::vector<float>& output, int total_tokens, int embedding_dim) {
    output.resize(total_tokens * embedding_dim);
    for (int i = 0; i < total_tokens; i++){
        int id = token_ids[i];
        for (int d = 0; d < embedding_dim; d++){
            if(id < 0 || id >= VOCAB_SIZE)
                output[i * embedding_dim + d] = 0.0f;
            else
                output[i * embedding_dim + d] = embedding_matrix[id * embedding_dim + d];
        }
    }
}

// Positional encoding
void cpu_positional_encoding(std::vector<float>& pe, int seq_len, int d_model) {
    pe.resize(seq_len * d_model);
    for (int pos = 0; pos < seq_len; pos++){
        for (int j = 0; j < d_model; j++){
            float factor = (j % 2 == 0) ? (j / 2.0f) : ((j - 1) / 2.0f);
            float div_term = exp(-log(10000.0f) * factor / d_model);
            float angle = pos * div_term;
            pe[pos * d_model + j] = (j % 2 == 0) ? sin(angle) : cos(angle);
        }
    }
}

// Matrix multiplication
void cpu_matmul(const std::vector<float>& A, const std::vector<float>& B,
                std::vector<float>& C, int M_mat, int K, int N) {
    C.resize(M_mat * N, 0.0f);
    for (int i = 0; i < M_mat; i++){
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
void cpu_multi_head_attention(const float *Q, const float *K, const float *V,
                              float *O, int M_val, int N_val, int H_val, int d_head) {
    for (int h = 0; h < H_val; h++){
        for (int i = 0; i < M_val; i++){
            std::vector<float> scores(N_val, 0.0f);
            float max_val = -1e20f;
            for (int j = 0; j < N_val; j++){
                float dot = 0.0f;
                for (int k = 0; k < d_head; k++){
                    float q_val = Q[i * (H_val*d_head) + h*d_head + k];
                    float k_val = K[j * (H_val*d_head) + h*d_head + k];
                    dot += q_val * k_val;
                }
                dot /= sqrt((float)d_head);
                scores[j] = dot;
                if (dot > max_val)
                    max_val = dot;
            }
            float sum_exp = 0.0f;
            for (int j = 0; j < N_val; j++){
                scores[j] = exp(scores[j] - max_val);
                sum_exp += scores[j];
            }
            for (int j = 0; j < N_val; j++){
                scores[j] /= sum_exp;
            }
            for (int k = 0; k < d_head; k++){
                float val = 0.0f;
                for (int j = 0; j < N_val; j++){
                    float v_val = V[j * (H_val*d_head) + h*d_head + k];
                    val += scores[j] * v_val;
                }
                O[i * (H_val*d_head) + h*d_head + k] = val;
            }
        }
    }
}

// Element-wise addition
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

// Layer normalization
void cpu_layer_norm(const float* input, float* output, int M_val, int N_val, float epsilon) {
    for (int i = 0; i < M_val; i++){
        float sum = 0.0f;
        for (int j = 0; j < N_val; j++){
            sum += input[i*N_val + j];
        }
        float mean = sum / N_val;
        float var = 0.0f;
        for (int j = 0; j < N_val; j++){
            float diff = input[i*N_val + j] - mean;
            var += diff * diff;
        }
        var /= N_val;
        float inv_std = 1.0f / sqrt(var + epsilon);
        for (int j = 0; j < N_val; j++){
            output[i*N_val + j] = (input[i*N_val + j] - mean) * inv_std;
        }
    }
}

// Encoder
void cpu_encoder(const std::vector<float>& X, std::vector<float>& output,
                 const std::vector<float>& Wq, const std::vector<float>& Wk,
                 const std::vector<float>& Wv, const std::vector<float>& Wo,
                 const std::vector<float>& W1, const std::vector<float>& W2) {
    std::vector<float> Q, K, V;
    cpu_matmul(X, Wq, Q, M, d_total, d_total);
    cpu_matmul(X, Wk, K, M, d_total, d_total);
    cpu_matmul(X, Wv, V, M, d_total, d_total);

    std::vector<float> attn(M * d_total, 0.0f);
    cpu_multi_head_attention(Q.data(), K.data(), V.data(), attn.data(), M, M, H, d_head);

    std::vector<float> MHA(M * d_total, 0.0f);
    cpu_matmul(attn, Wo, MHA, M, d_total, d_total);

    std::vector<float> add1(M * d_total, 0.0f);
    cpu_add(X.data(), MHA.data(), add1.data(), M * d_total);

    std::vector<float> ln1(M * d_total, 0.0f);
    cpu_layer_norm(add1.data(), ln1.data(), M, d_total, 1e-5);

    std::vector<float> ffn1(M * d_ff, 0.0f);
    cpu_matmul(ln1, W1, ffn1, M, d_total, d_ff);
    cpu_relu(ffn1.data(), M * d_ff);
    std::vector<float> ffn2(M * d_total, 0.0f);
    cpu_matmul(ffn1, W2, ffn2, M, d_ff, d_total);

    std::vector<float> add2(M * d_total, 0.0f);
    cpu_add(ln1.data(), ffn2.data(), add2.data(), M * d_total);

    std::vector<float> ln2(M * d_total, 0.0f);
    cpu_layer_norm(add2.data(), ln2.data(), M, d_total, 1e-5);

    output = ln2;
}

// Decoder
void cpu_decoder(const std::vector<float>& Y, const std::vector<float>& enc_out,
                 std::vector<float>& output,
                 const std::vector<float>& Wq_self, const std::vector<float>& Wk_self,
                 const std::vector<float>& Wv_self, const std::vector<float>& Wo_self,
                 const std::vector<float>& Wq_encdec, const std::vector<float>& Wk_encdec,
                 const std::vector<float>& Wv_encdec, const std::vector<float>& Wo_encdec,
                 const std::vector<float>& W1, const std::vector<float>& W2) {
    std::vector<float> Q_self, K_self, V_self;
    cpu_matmul(Y, Wq_self, Q_self, M, d_total, d_total);
    cpu_matmul(Y, Wk_self, K_self, M, d_total, d_total);
    cpu_matmul(Y, Wv_self, V_self, M, d_total, d_total);

    std::vector<float> attn_self(M * d_total, 0.0f);
    cpu_multi_head_attention(Q_self.data(), K_self.data(), V_self.data(), attn_self.data(), M, M, H, d_head);

    std::vector<float> MHA_self(M * d_total, 0.0f);
    cpu_matmul(attn_self, Wo_self, MHA_self, M, d_total, d_total);
    std::vector<float> add_self(M * d_total, 0.0f);
    cpu_add(Y.data(), MHA_self.data(), add_self.data(), M * d_total);
    std::vector<float> ln_self(M * d_total, 0.0f);
    cpu_layer_norm(add_self.data(), ln_self.data(), M, d_total, 1e-5);

    std::vector<float> Q_encdec, K_encdec, V_encdec;
    cpu_matmul(ln_self, Wq_encdec, Q_encdec, M, d_total, d_total);
    cpu_matmul(enc_out,    Wk_encdec, K_encdec, M, d_total, d_total);
    cpu_matmul(enc_out,    Wv_encdec, V_encdec, M, d_total, d_total);

    std::vector<float> attn_encdec(M * d_total, 0.0f);
    cpu_multi_head_attention(Q_encdec.data(), K_encdec.data(), V_encdec.data(), attn_encdec.data(), M, M, H, d_head);

    std::vector<float> MHA_encdec(M * d_total, 0.0f);
    cpu_matmul(attn_encdec, Wo_encdec, MHA_encdec, M, d_total, d_total);

    std::vector<float> add_encdec(M * d_total, 0.0f);
    cpu_add(ln_self.data(), MHA_encdec.data(), add_encdec.data(), M * d_total);

    std::vector<float> ln_encdec(M * d_total, 0.0f);
    cpu_layer_norm(add_encdec.data(), ln_encdec.data(), M, d_total, 1e-5);

    std::vector<float> ffn1(M * d_ff, 0.0f);
    cpu_matmul(ln_encdec, W1, ffn1, M, d_total, d_ff);
    cpu_relu(ffn1.data(), M * d_ff);
    std::vector<float> ffn2(M * d_total, 0.0f);
    cpu_matmul(ffn1, W2, ffn2, M, d_ff, d_total);

    std::vector<float> add_ffn(M * d_total, 0.0f);
    cpu_add(ln_encdec.data(), ffn2.data(), add_ffn.data(), M * d_total);

    std::vector<float> ln_dec(M * d_total, 0.0f);
    cpu_layer_norm(add_ffn.data(), ln_dec.data(), M, d_total, 1e-5);

    output = ln_dec;
}
int main() {
    // Tokenization on CPU
    std::vector<std::string> encoder_sentences = {"hello world"};
    std::vector<std::string> decoder_sentences = {"cuda programming"};
    std::unordered_map<std::string,int> vocab = { {"hello", 0}, {"world", 1}, {"cuda", 2}, {"programming", 3} };
    std::vector<int> encoder_token_ids, decoder_token_ids;
    cpu_tokenizer(encoder_sentences, vocab, encoder_token_ids);
    cpu_tokenizer(decoder_sentences, vocab, decoder_token_ids);
    int total_tokens_enc = NUM_SENTENCES * MAX_TOKENS;
    int total_tokens_dec = NUM_SENTENCES * MAX_TOKENS;

    // Embedding Lookup on CPU
    std::vector<float> embedding_matrix(VOCAB_SIZE * EMBEDDING_DIM);
    for (int i = 0; i < VOCAB_SIZE; i++){
        for (int j = 0; j < EMBEDDING_DIM; j++){
            embedding_matrix[i * EMBEDDING_DIM + j] = static_cast<float>((i*17 + j + 1) % 100) / 100.0f;
        }
    }
    std::vector<float> enc_embeddings, dec_embeddings;
    cpu_embedding_lookup(encoder_token_ids, embedding_matrix, enc_embeddings, total_tokens_enc, EMBEDDING_DIM);
    cpu_embedding_lookup(decoder_token_ids, embedding_matrix, dec_embeddings, total_tokens_dec, EMBEDDING_DIM);

    // Positional encoding and add to embeddings
    std::vector<float> pos_enc;
    cpu_positional_encoding(pos_enc, MAX_TOKENS, EMBEDDING_DIM);
    for (int i = 0; i < total_tokens_enc; i++){
        for (int j = 0; j < EMBEDDING_DIM; j++){
            enc_embeddings[i * EMBEDDING_DIM + j] += pos_enc[(i % MAX_TOKENS) * EMBEDDING_DIM + j];
        }
    }
    for (int i = 0; i < total_tokens_dec; i++){
        for (int j = 0; j < EMBEDDING_DIM; j++){
            dec_embeddings[i * EMBEDDING_DIM + j] += pos_enc[(i % MAX_TOKENS) * EMBEDDING_DIM + j];
        }
    }
    
    // Initialize encoder and decoder weights
    int weight_size = d_total * d_total;
    int weight_ff_size = d_total * d_ff;
    int weight_ff2_size = d_ff * d_total;
    std::vector<float> Wq_enc(weight_size), Wk_enc(weight_size), Wv_enc(weight_size), Wo_enc(weight_size);
    std::vector<float> W1_enc(weight_ff_size), W2_enc(weight_ff2_size);
    std::vector<float> Wq_dec_self(weight_size), Wk_dec_self(weight_size),
                       Wv_dec_self(weight_size), Wo_dec_self(weight_size);
    std::vector<float> Wq_dec_encdec(weight_size), Wk_dec_encdec(weight_size),
                       Wv_dec_encdec(weight_size), Wo_dec_encdec(weight_size);
    std::vector<float> W1_dec(weight_ff_size), W2_dec(weight_ff2_size);
    for (int i = 0; i < weight_size; i++){
        Wq_enc[i] = static_cast<float>((i + 2) % 100) / 100.0f;
        Wk_enc[i] = static_cast<float>((i + 3) % 100) / 100.0f;
        Wv_enc[i] = static_cast<float>((i + 4) % 100) / 100.0f;
        Wo_enc[i] = static_cast<float>((i + 5) % 100) / 100.0f;
        Wq_dec_self[i] = static_cast<float>((i + 3) % 100) / 100.0f;
        Wk_dec_self[i] = static_cast<float>((i + 4) % 100) / 100.0f;
        Wv_dec_self[i] = static_cast<float>((i + 5) % 100) / 100.0f;
        Wo_dec_self[i] = static_cast<float>((i + 6) % 100) / 100.0f;
        Wq_dec_encdec[i] = static_cast<float>((i + 7) % 100) / 100.0f;
        Wk_dec_encdec[i] = static_cast<float>((i + 8) % 100) / 100.0f;
        Wv_dec_encdec[i] = static_cast<float>((i + 9) % 100) / 100.0f;
        Wo_dec_encdec[i] = static_cast<float>((i + 10) % 100) / 100.0f;
    }
    for (int i = 0; i < weight_ff_size; i++){
        W1_enc[i] = static_cast<float>((i + 6) % 100) / 100.0f;
        W1_dec[i] = static_cast<float>((i + 11) % 100) / 100.0f;
    }
    for (int i = 0; i < weight_ff2_size; i++){
        W2_enc[i] = static_cast<float>((i + 7) % 100) / 100.0f;
        W2_dec[i] = static_cast<float>((i + 12) % 100) / 100.0f;
    }

    // 5. CPU Inference
    auto cpu_start = std::chrono::high_resolution_clock::now();
    std::vector<float> enc_out_cpu;
    cpu_encoder(enc_embeddings, enc_out_cpu, Wq_enc, Wk_enc, Wv_enc, Wo_enc, W1_enc, W2_enc);
    std::vector<float> dec_out_cpu;
    cpu_decoder(dec_embeddings, enc_out_cpu, dec_out_cpu,
                Wq_dec_self, Wk_dec_self, Wv_dec_self, Wo_dec_self,
                Wq_dec_encdec, Wk_dec_encdec, Wv_dec_encdec, Wo_dec_encdec,
                W1_dec, W2_dec);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    double cpu_time = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();
    std::cout << "CPU Total Inference Time: " << cpu_time << " ms" << std::endl;

    // 6. GPU Inference
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

    // Copy embeddings and weights to GPU
    float *d_enc_embed, *d_dec_embed;
    int embed_bytes = total_tokens_enc * EMBEDDING_DIM * sizeof(float);
    int dec_embed_bytes = total_tokens_dec * EMBEDDING_DIM * sizeof(float);
    CHECK_CUDA(cudaMalloc(&d_enc_embed, embed_bytes));
    CHECK_CUDA(cudaMalloc(&d_dec_embed, dec_embed_bytes));
    CHECK_CUDA(cudaEventRecord(start_h2d, 0));
    CHECK_CUDA(cudaMemcpy(d_enc_embed, enc_embeddings.data(), embed_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_dec_embed, dec_embeddings.data(), dec_embed_bytes, cudaMemcpyHostToDevice));
    // Allocate and copy encoder weights
    float *d_Wq_enc, *d_Wk_enc, *d_Wv_enc, *d_Wo_enc, *d_W1_enc, *d_W2_enc;
    CHECK_CUDA(cudaMalloc(&d_Wq_enc, weight_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_Wk_enc, weight_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_Wv_enc, weight_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_Wo_enc, weight_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_W1_enc, weight_ff_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_W2_enc, weight_ff2_size * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_Wq_enc, Wq_enc.data(), weight_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_Wk_enc, Wk_enc.data(), weight_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_Wv_enc, Wv_enc.data(), weight_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_Wo_enc, Wo_enc.data(), weight_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_W1_enc, W1_enc.data(), weight_ff_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_W2_enc, W2_enc.data(), weight_ff2_size * sizeof(float), cudaMemcpyHostToDevice));
    // Allocate and copy decoder weights
    float *d_Wq_dec_self, *d_Wk_dec_self, *d_Wv_dec_self, *d_Wo_dec_self;
    float *d_Wq_dec_encdec, *d_Wk_dec_encdec, *d_Wv_dec_encdec, *d_Wo_dec_encdec;
    float *d_W1_dec, *d_W2_dec;
    CHECK_CUDA(cudaMalloc(&d_Wq_dec_self, weight_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_Wk_dec_self, weight_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_Wv_dec_self, weight_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_Wo_dec_self, weight_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_Wq_dec_encdec, weight_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_Wk_dec_encdec, weight_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_Wv_dec_encdec, weight_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_Wo_dec_encdec, weight_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_W1_dec, weight_ff_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_W2_dec, weight_ff2_size * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_Wq_dec_self, Wq_dec_self.data(), weight_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_Wk_dec_self, Wk_dec_self.data(), weight_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_Wv_dec_self, Wv_dec_self.data(), weight_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_Wo_dec_self, Wo_dec_self.data(), weight_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_Wq_dec_encdec, Wq_dec_encdec.data(), weight_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_Wk_dec_encdec, Wk_dec_encdec.data(), weight_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_Wv_dec_encdec, Wv_dec_encdec.data(), weight_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_Wo_dec_encdec, Wo_dec_encdec.data(), weight_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_W1_dec, W1_dec.data(), weight_ff_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_W2_dec, W2_dec.data(), weight_ff2_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaEventRecord(stop_h2d, 0));
    CHECK_CUDA(cudaEventSynchronize(stop_h2d));
    float h2d_time;
    CHECK_CUDA(cudaEventElapsedTime(&h2d_time, start_h2d, stop_h2d));

    // GPU Encoder Processing
    size_t input_size_enc = M * d_total * sizeof(float);
    size_t proj_size_enc = M * d_total * sizeof(float);
    size_t scores_size_enc = M * M * H * sizeof(float);
    size_t ffn1_size_enc = M * d_ff * sizeof(float);
    float *d_input, *d_Wq, *d_Wk, *d_Wv, *d_Wo;
    float *d_Q, *d_K, *d_V, *d_scores, *d_O;
    float *d_MHA, *d_add1, *d_ln1, *d_ffn1, *d_ffn2, *d_add2, *d_ln2;
    float *d_W1, *d_W2;
    CHECK_CUDA(cudaMalloc(&d_input, input_size_enc));
    CHECK_CUDA(cudaMalloc(&d_Wq, weight_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_Wk, weight_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_Wv, weight_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_Wo, weight_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_Q, proj_size_enc));
    CHECK_CUDA(cudaMalloc(&d_K, proj_size_enc));
    CHECK_CUDA(cudaMalloc(&d_V, proj_size_enc));
    CHECK_CUDA(cudaMalloc(&d_scores, scores_size_enc));
    CHECK_CUDA(cudaMalloc(&d_O, proj_size_enc));
    CHECK_CUDA(cudaMalloc(&d_MHA, proj_size_enc));
    CHECK_CUDA(cudaMalloc(&d_add1, proj_size_enc));
    CHECK_CUDA(cudaMalloc(&d_ln1, proj_size_enc));
    CHECK_CUDA(cudaMalloc(&d_ffn1, ffn1_size_enc));
    CHECK_CUDA(cudaMalloc(&d_ffn2, proj_size_enc));
    CHECK_CUDA(cudaMalloc(&d_add2, proj_size_enc));
    CHECK_CUDA(cudaMalloc(&d_ln2, proj_size_enc));
    CHECK_CUDA(cudaMalloc(&d_W1, weight_ff_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_W2, weight_ff2_size * sizeof(float)));
    // Copy encoder inputs and weights from device (for weights, use device-to-device copy)
    CHECK_CUDA(cudaMemcpy(d_input, d_enc_embed, input_size_enc, cudaMemcpyDeviceToDevice));
    CHECK_CUDA(cudaMemcpy(d_Wq, d_Wq_enc, weight_size * sizeof(float), cudaMemcpyDeviceToDevice));
    CHECK_CUDA(cudaMemcpy(d_Wk, d_Wk_enc, weight_size * sizeof(float), cudaMemcpyDeviceToDevice));
    CHECK_CUDA(cudaMemcpy(d_Wv, d_Wv_enc, weight_size * sizeof(float), cudaMemcpyDeviceToDevice));
    CHECK_CUDA(cudaMemcpy(d_Wo, d_Wo_enc, weight_size * sizeof(float), cudaMemcpyDeviceToDevice));
    CHECK_CUDA(cudaMemcpy(d_W1, d_W1_enc, weight_ff_size * sizeof(float), cudaMemcpyDeviceToDevice));
    CHECK_CUDA(cudaMemcpy(d_W2, d_W2_enc, weight_ff2_size * sizeof(float), cudaMemcpyDeviceToDevice));

    cudaEventRecord(start_kernel, 0);
    dim3 block(TILE_WIDTH, TILE_WIDTH);
    dim3 grid((d_total + TILE_WIDTH - 1) / TILE_WIDTH, (M + TILE_WIDTH - 1) / TILE_WIDTH);
    // Linear projections for encoder
    matmul_kernel<<<grid, block>>>(d_input, d_Wq, d_Q, M, d_total, d_total);
    matmul_kernel<<<grid, block>>>(d_input, d_Wk, d_K, M, d_total, d_total);
    matmul_kernel<<<grid, block>>>(d_input, d_Wv, d_V, M, d_total, d_total);
    // Multi-head attention
    dim3 grid_att((M + TILE_WIDTH - 1) / TILE_WIDTH, (M + TILE_WIDTH - 1) / TILE_WIDTH, H);
    compute_scores_kernel<<<grid_att, block>>>(d_Q, d_K, d_scores, M, M, H, d_head);
    int threads_att = 256;
    int blocks_att = M * H;
    size_t shared_mem_bytes = threads_att * sizeof(float);
    softmax_kernel<<<blocks_att, threads_att, shared_mem_bytes>>>(d_scores, M, M, H);
    dim3 grid_ws((d_head + TILE_WIDTH - 1) / TILE_WIDTH, (M + TILE_WIDTH - 1) / TILE_WIDTH, H);
    weighted_sum_kernel<<<grid_ws, block>>>(d_scores, d_V, d_O, M, M, H, d_head);
    // Final projection for attention branch
    matmul_kernel<<<grid, block>>>(d_O, d_Wo, d_MHA, M, d_total, d_total);
    int total_elems_enc = M * d_total;
    int threads_add = 256;
    int blocks_add = (total_elems_enc + threads_add - 1) / threads_add;
    add_kernel<<<blocks_add, threads_add>>>(d_input, d_MHA, d_add1, total_elems_enc);
    int ln_threads = 256;
    layer_norm_kernel<<<M, ln_threads, ln_threads * sizeof(float)>>>(d_add1, d_ln1, M, d_total, 1e-5);
    // Feed forward network in encoder
    dim3 grid_ffn((d_ff + TILE_WIDTH - 1) / TILE_WIDTH, (M + TILE_WIDTH - 1) / TILE_WIDTH);
    matmul_kernel<<<grid_ffn, block>>>(d_ln1, d_W1, d_ffn1, M, d_total, d_ff);
    int total_ffn1 = M * d_ff;
    int blocks_relu = (total_ffn1 + threads_add - 1) / threads_add;
    relu_kernel<<<blocks_relu, threads_add>>>(d_ffn1, total_ffn1);
    matmul_kernel<<<grid, block>>>(d_ffn1, d_W2, d_ffn2, M, d_ff, d_total);
    add_kernel<<<blocks_add, threads_add>>>(d_ln1, d_ffn2, d_add2, total_elems_enc);
    layer_norm_kernel<<<M, ln_threads, ln_threads * sizeof(float)>>>(d_add2, d_ln2, M, d_total, 1e-5);
    cudaEventRecord(stop_kernel, 0);
    cudaEventSynchronize(stop_kernel);
    float kernel_time_enc;
    CHECK_CUDA(cudaEventElapsedTime(&kernel_time_enc, start_kernel, stop_kernel));

    // Setting encoder output pointer for decoder
    float *d_enc = d_ln2; // Final encoder output

    // GPU Decoder Processing
    float *d_Y = d_dec_embed; // decoder input embeddings

    size_t proj_size_dec = M * d_total * sizeof(float);
    size_t scores_size_dec = M * M * H * sizeof(float);
    size_t ffn1_size_dec = M * d_ff * sizeof(float);
    float *d_Q_dec, *d_K_dec, *d_V_dec;
    float *d_scores_dec, *d_O_dec;
    float *d_attn_dec;
    float *d_add_self_dec, *d_ln_self_dec;
    float *d_Q_encdec, *d_K_encdec, *d_V_encdec;
    float *d_scores_encdec, *d_O_encdec;
    float *d_attn_encdec;
    float *d_add_encdec, *d_ln_encdec;
    float *d_ffn1_dec, *d_ffn2_dec, *d_add_ffn_dec, *d_ln_dec;
    CHECK_CUDA(cudaMalloc(&d_Q_dec, proj_size_dec));
    CHECK_CUDA(cudaMalloc(&d_K_dec, proj_size_dec));
    CHECK_CUDA(cudaMalloc(&d_V_dec, proj_size_dec));
    CHECK_CUDA(cudaMalloc(&d_scores_dec, scores_size_dec));
    CHECK_CUDA(cudaMalloc(&d_O_dec, proj_size_dec));
    CHECK_CUDA(cudaMalloc(&d_attn_dec, proj_size_dec));
    CHECK_CUDA(cudaMalloc(&d_add_self_dec, proj_size_dec));
    CHECK_CUDA(cudaMalloc(&d_ln_self_dec, proj_size_dec));
    CHECK_CUDA(cudaMalloc(&d_Q_encdec, proj_size_dec));
    CHECK_CUDA(cudaMalloc(&d_K_encdec, proj_size_dec));
    CHECK_CUDA(cudaMalloc(&d_V_encdec, proj_size_dec));
    CHECK_CUDA(cudaMalloc(&d_scores_encdec, scores_size_dec));
    CHECK_CUDA(cudaMalloc(&d_O_encdec, proj_size_dec));
    CHECK_CUDA(cudaMalloc(&d_attn_encdec, proj_size_dec));
    CHECK_CUDA(cudaMalloc(&d_add_encdec, proj_size_dec));
    CHECK_CUDA(cudaMalloc(&d_ln_encdec, proj_size_dec));
    CHECK_CUDA(cudaMalloc(&d_ffn1_dec, ffn1_size_dec));
    CHECK_CUDA(cudaMalloc(&d_ffn2_dec, proj_size_dec));
    CHECK_CUDA(cudaMalloc(&d_add_ffn_dec, proj_size_dec));
    CHECK_CUDA(cudaMalloc(&d_ln_dec, proj_size_dec));

    CHECK_CUDA(cudaEventRecord(start_kernel, 0));
    dim3 grid_dec((d_total + TILE_WIDTH - 1) / TILE_WIDTH, (M + TILE_WIDTH - 1) / TILE_WIDTH);
    // Self-attention for decoder
    matmul_kernel<<<grid_dec, block>>>(d_Y, d_Wq_dec_self, d_Q_dec, M, d_total, d_total);
    matmul_kernel<<<grid_dec, block>>>(d_Y, d_Wk_dec_self, d_K_dec, M, d_total, d_total);
    matmul_kernel<<<grid_dec, block>>>(d_Y, d_Wv_dec_self, d_V_dec, M, d_total, d_total);
    dim3 grid_att_dec((M + TILE_WIDTH - 1) / TILE_WIDTH, (M + TILE_WIDTH - 1) / TILE_WIDTH, H);
    compute_scores_kernel<<<grid_att_dec, block>>>(d_Q_dec, d_K_dec, d_scores_dec, M, M, H, d_head);
    int threads_att_dec = 256;
    size_t shared_mem_dec = threads_att_dec * sizeof(float);
    softmax_kernel<<<M * H, threads_att_dec, shared_mem_dec>>>(d_scores_dec, M, M, H);
    weighted_sum_kernel<<<grid_att_dec, block>>>(d_scores_dec, d_V_dec, d_O_dec, M, M, H, d_head);
    matmul_kernel<<<grid_dec, block>>>(d_O_dec, d_Wo_dec_self, d_attn_dec, M, d_total, d_total);
    int total_elems_dec = M * d_total;
    int blocks_add_dec = (total_elems_dec + threads_add - 1) / threads_add;
    add_kernel<<<blocks_add_dec, threads_add>>>(d_Y, d_attn_dec, d_add_self_dec, total_elems_dec);
    layer_norm_kernel<<<M, ln_threads, ln_threads * sizeof(float)>>>(d_add_self_dec, d_ln_self_dec, M, d_total, 1e-5);
    
    // Encoder-decoder attention
    matmul_kernel<<<grid_dec, block>>>(d_ln_self_dec, d_Wq_dec_encdec, d_Q_encdec, M, d_total, d_total);
    // Use the encoder output (d_enc) set earlier
    matmul_kernel<<<grid_dec, block>>>(d_enc, d_Wk_dec_encdec, d_K_encdec, M, d_total, d_total);
    matmul_kernel<<<grid_dec, block>>>(d_enc, d_Wv_dec_encdec, d_V_encdec, M, d_total, d_total);
    dim3 grid_att_encdec((M + TILE_WIDTH - 1) / TILE_WIDTH, (M + TILE_WIDTH - 1) / TILE_WIDTH, H);
    compute_scores_kernel<<<grid_att_encdec, block>>>(d_Q_encdec, d_K_encdec, d_scores_encdec, M, M, H, d_head);
    softmax_kernel<<<M * H, threads_att_dec, shared_mem_dec>>>(d_scores_encdec, M, M, H);
    weighted_sum_kernel<<<grid_att_encdec, block>>>(d_scores_encdec, d_V_encdec, d_O_encdec, M, M, H, d_head);
    matmul_kernel<<<grid_dec, block>>>(d_O_encdec, d_Wo_dec_encdec, d_attn_encdec, M, d_total, d_total);
    add_kernel<<<blocks_add_dec, threads_add>>>(d_ln_self_dec, d_attn_encdec, d_add_encdec, total_elems_dec);
    layer_norm_kernel<<<M, ln_threads, ln_threads * sizeof(float)>>>(d_add_encdec, d_ln_encdec, M, d_total, 1e-5);
    
    // Feed forward network in decoder
    matmul_kernel<<<grid_ffn, block>>>(d_ln_encdec, d_W1_dec, d_ffn1_dec, M, d_total, d_ff);
    int total_ffn1_dec = M * d_ff;
    int blocks_relu_dec = (total_ffn1_dec + threads_add - 1) / threads_add;
    relu_kernel<<<blocks_relu_dec, threads_add>>>(d_ffn1_dec, total_ffn1_dec);
    matmul_kernel<<<grid_dec, block>>>(d_ffn1_dec, d_W2_dec, d_ffn2_dec, M, d_ff, d_total);
    add_kernel<<<blocks_add_dec, threads_add>>>(d_ln_encdec, d_ffn2_dec, d_add_ffn_dec, total_elems_dec);
    layer_norm_kernel<<<M, ln_threads, ln_threads * sizeof(float)>>>(d_add_ffn_dec, d_ln_dec, M, d_total, 1e-5);
    CHECK_CUDA(cudaEventRecord(stop_kernel, 0));
    CHECK_CUDA(cudaEventSynchronize(stop_kernel));
    float kernel_time_dec;
    CHECK_CUDA(cudaEventElapsedTime(&kernel_time_dec, start_kernel, stop_kernel));

    // Copy final decoder output from device to host
    std::vector<float> dec_out_gpu(M * d_total);
    CHECK_CUDA(cudaEventRecord(start_d2h, 0));
    CHECK_CUDA(cudaMemcpy(dec_out_gpu.data(), d_ln_dec, proj_size_dec, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaEventRecord(stop_d2h, 0));
    CHECK_CUDA(cudaEventSynchronize(stop_d2h));
    float d2h_time;
    CHECK_CUDA(cudaEventElapsedTime(&d2h_time, start_d2h, stop_d2h));

    CHECK_CUDA(cudaEventRecord(stop_total, 0));
    CHECK_CUDA(cudaEventSynchronize(stop_total));
    float total_gpu_time;
    CHECK_CUDA(cudaEventElapsedTime(&total_gpu_time, start_total, stop_total));

    std::cout << "GPU Inference Timings:" << std::endl;
    std::cout << "  Host-to-Device Copy Time: " << h2d_time << " ms" << std::endl;
    std::cout << "  Encoder Kernel Time: " << kernel_time_enc << " ms" << std::endl;
    std::cout << "  Decoder Kernel Time: " << kernel_time_dec << " ms" << std::endl;
    std::cout << "  Device-to-Host Copy Time: " << d2h_time << " ms" << std::endl;
    std::cout << "  Total GPU Inference Time: " << total_gpu_time << " ms" << std::endl;

    std::cout << std::endl;

    // Cleanup GPU memory and events
    cudaFree(d_enc_embed);  cudaFree(d_dec_embed);
    cudaFree(d_Wq_enc);  cudaFree(d_Wk_enc);  cudaFree(d_Wv_enc);  cudaFree(d_Wo_enc);
    cudaFree(d_W1_enc);  cudaFree(d_W2_enc);
    cudaFree(d_Wq_dec_self);  cudaFree(d_Wk_dec_self);  cudaFree(d_Wv_dec_self);  cudaFree(d_Wo_dec_self);
    cudaFree(d_Wq_dec_encdec);  cudaFree(d_Wk_dec_encdec);  cudaFree(d_Wv_dec_encdec);  cudaFree(d_Wo_dec_encdec);
    cudaFree(d_W1_dec);  cudaFree(d_W2_dec);
    cudaFree(d_input);  cudaFree(d_Wq);  cudaFree(d_Wk);  cudaFree(d_Wv);  cudaFree(d_Wo);
    cudaFree(d_Q);  cudaFree(d_K);  cudaFree(d_V);
    cudaFree(d_scores);  cudaFree(d_O);
    cudaFree(d_MHA);  cudaFree(d_add1);  cudaFree(d_ln1);
    cudaFree(d_ffn1); cudaFree(d_ffn2);
    cudaFree(d_add2); cudaFree(d_ln2);
    // Decoder GPU buffers
    cudaFree(d_Y);
    cudaFree(d_Q_dec);  cudaFree(d_K_dec);  cudaFree(d_V_dec);
    cudaFree(d_scores_dec);  cudaFree(d_O_dec);
    cudaFree(d_attn_dec);  cudaFree(d_add_self_dec);  cudaFree(d_ln_self_dec);
    cudaFree(d_Q_encdec); cudaFree(d_K_encdec); cudaFree(d_V_encdec);
    cudaFree(d_scores_encdec); cudaFree(d_O_encdec);
    cudaFree(d_attn_encdec);  cudaFree(d_add_encdec); cudaFree(d_ln_encdec);
    cudaFree(d_ffn1_dec); cudaFree(d_ffn2_dec);
    cudaFree(d_add_ffn_dec); cudaFree(d_ln_dec);
    cudaEventDestroy(start_total);  cudaEventDestroy(stop_total);
    cudaEventDestroy(start_h2d);  cudaEventDestroy(stop_h2d);
    cudaEventDestroy(start_kernel);  cudaEventDestroy(stop_kernel);
    cudaEventDestroy(start_d2h);  cudaEventDestroy(stop_d2h);

    return 0;
}
