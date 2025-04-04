// tokenizer.cu
#include <iostream>
#include <sstream>
#include <vector>
#include <string>
#include <unordered_map>
#include <cuda_runtime.h>
#include <cstdlib>

// CUDA error checking macro.
#define CHECK_CUDA(call) {                                 \
    cudaError_t err = call;                                \
    if(err != cudaSuccess) {                               \
        std::cerr << "CUDA error: " << cudaGetErrorString(err) \
                  << " at " << __FILE__ << ":" << __LINE__  \
                  << std::endl;                            \
        exit(1);                                           \
    }                                                      \
}

// Constants
const int EMBEDDING_DIM = 4;
const int VOCAB_SIZE = 4;     
const int NUM_SENTENCES = 2;
const int MAX_TOKENS = 5;     

// CUDA kernel for embedding lookup
__global__ void embedding_lookup_kernel(const int* token_ids, 
                                          const float* embedding_matrix, 
                                          float* output, 
                                          int num_sentences, 
                                          int max_tokens, 
                                          int embedding_dim) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int total_tokens = num_sentences * max_tokens;
    if (idx >= total_tokens) return;
    
    int token_id = token_ids[idx];
    for (int d = 0; d < embedding_dim; d++) {
        if (token_id < 0 || token_id >= VOCAB_SIZE) {
            output[idx * embedding_dim + d] = 0.0f;
        } else {
            output[idx * embedding_dim + d] = embedding_matrix[token_id * embedding_dim + d];
        }
    }
}

int main() {
    
    // Tokenization on CPU
    std::vector<std::string> sentences = { "hello world", "cuda programming" };

    // Predefined vocabulary mapping (word -> token id)
    std::unordered_map<std::string, int> vocab = {
        {"hello", 0},
        {"world", 1},
        {"cuda", 2},
        {"programming", 3}
    };

    // Create a flat array for token ids, padded with -1 (for missing tokens)
    std::vector<int> token_ids(NUM_SENTENCES * MAX_TOKENS, -1);
    for (int i = 0; i < NUM_SENTENCES; i++) {
        std::istringstream iss(sentences[i]);
        std::string token;
        int token_idx = 0;
        while (iss >> token && token_idx < MAX_TOKENS) {
            if (vocab.find(token) != vocab.end())
                token_ids[i * MAX_TOKENS + token_idx] = vocab[token];
            else
                token_ids[i * MAX_TOKENS + token_idx] = -1;
            token_idx++;
        }
    }

    // Define embedding matrix
    std::vector<float> h_embedding_matrix = {
        // Token 0: "hello"
        0.1f, 0.2f, 0.3f, 0.4f,
        // Token 1: "world"
        0.5f, 0.6f, 0.7f, 0.8f,
        // Token 2: "cuda"
        0.9f, 1.0f, 1.1f, 1.2f,
        // Token 3: "programming"
        1.3f, 1.4f, 1.5f, 1.6f
    };

    // Number of tokens total and size of output embeddings
    int total_tokens = NUM_SENTENCES * MAX_TOKENS;
    int output_size = total_tokens * EMBEDDING_DIM;

    // Allocate device memory
    int* d_token_ids = nullptr;
    float* d_embedding_matrix = nullptr;
    float* d_output = nullptr;
    CHECK_CUDA(cudaMalloc(&d_token_ids, total_tokens * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_embedding_matrix, VOCAB_SIZE * EMBEDDING_DIM * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_output, output_size * sizeof(float)));

    cudaEvent_t start, stop;
    cudaEvent_t copy_in_start, copy_in_stop;
    cudaEvent_t kernel_start, kernel_stop;
    cudaEvent_t copy_out_start, copy_out_stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventCreate(&copy_in_start));
    CHECK_CUDA(cudaEventCreate(&copy_in_stop));
    CHECK_CUDA(cudaEventCreate(&kernel_start));
    CHECK_CUDA(cudaEventCreate(&kernel_stop));
    CHECK_CUDA(cudaEventCreate(&copy_out_start));
    CHECK_CUDA(cudaEventCreate(&copy_out_stop));

    CHECK_CUDA(cudaEventRecord(start));

    // Copy Data from host to device
    CHECK_CUDA(cudaEventRecord(copy_in_start));
    CHECK_CUDA(cudaMemcpy(d_token_ids, token_ids.data(), total_tokens * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_embedding_matrix, h_embedding_matrix.data(), VOCAB_SIZE * EMBEDDING_DIM * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaEventRecord(copy_in_stop));
    CHECK_CUDA(cudaEventSynchronize(copy_in_stop));

    // Launch embedding kernel
    int threadsPerBlock = 256;
    int blocks = (total_tokens + threadsPerBlock - 1) / threadsPerBlock;
    CHECK_CUDA(cudaEventRecord(kernel_start));
    embedding_lookup_kernel<<<blocks, threadsPerBlock>>>(d_token_ids, d_embedding_matrix, d_output, NUM_SENTENCES, MAX_TOKENS, EMBEDDING_DIM);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaEventRecord(kernel_stop));
    CHECK_CUDA(cudaEventSynchronize(kernel_stop));

    // Copy back from device to host
    std::vector<float> h_output(output_size);
    CHECK_CUDA(cudaEventRecord(copy_out_start));
    CHECK_CUDA(cudaMemcpy(h_output.data(), d_output, output_size * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaEventRecord(copy_out_stop));
    CHECK_CUDA(cudaEventSynchronize(copy_out_stop));

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float total_time, copy_in_time, kernel_time, copy_out_time;
    CHECK_CUDA(cudaEventElapsedTime(&total_time, start, stop));
    CHECK_CUDA(cudaEventElapsedTime(&copy_in_time, copy_in_start, copy_in_stop));
    CHECK_CUDA(cudaEventElapsedTime(&kernel_time, kernel_start, kernel_stop));
    CHECK_CUDA(cudaEventElapsedTime(&copy_out_time, copy_out_start, copy_out_stop));

    std::cout << "CPU -> GPU copy time: " << copy_in_time << " ms" << std::endl;
    std::cout << "Kernel execution time: " << kernel_time << " ms" << std::endl;
    std::cout << "GPU -> CPU copy time: " << copy_out_time << " ms" << std::endl;
    std::cout << "Total elapsed time: " << total_time << " ms" << std::endl;

    // Print embeddings
    std::cout << "\nOutput Embeddings:" << std::endl;
    for (int i = 0; i < NUM_SENTENCES; i++) {
        std::cout << "Sentence " << i << ":" << std::endl;
        for (int t = 0; t < MAX_TOKENS; t++) {
            std::cout << "  Token " << t << ": ";
            for (int d = 0; d < EMBEDDING_DIM; d++) {
                std::cout << h_output[(i * MAX_TOKENS + t) * EMBEDDING_DIM + d] << " ";
            }
            std::cout << std::endl;
        }
    }

    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaEventDestroy(copy_in_start);
    cudaEventDestroy(copy_in_stop);
    cudaEventDestroy(kernel_start);
    cudaEventDestroy(kernel_stop);
    cudaEventDestroy(copy_out_start);
    cudaEventDestroy(copy_out_stop);
    cudaFree(d_token_ids);
    cudaFree(d_embedding_matrix);
    cudaFree(d_output);

    return 0;
}
