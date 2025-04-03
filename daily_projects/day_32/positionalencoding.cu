// positional_encoding.cu
#include <iostream>
#include <cuda_runtime.h>
#include <cmath>

// CUDA error checking macro.
#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if(err != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(err) \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(1); \
    } \
}

// Kernel to compute positional encodings.
// The output tensor 'pe' is assumed to have shape (seq_len * d_model)
__global__ void positional_encoding_kernel(float* pe, int seq_len, int d_model) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int total_elements = seq_len * d_model;
    if (idx >= total_elements) return;

    int pos = idx / d_model;      // Position index [0, seq_len)
    int j   = idx % d_model;        // Dimension index [0, d_model)

    // Compute the factor index for even/odd splitting
    // For even j: factor = j/2; for odd j: factor = (j-1)/2
    float factor = (j % 2 == 0) ? (j / 2.0f) : ((j - 1) / 2.0f);
    
    // Compute the denominator term: exp(-log(10000) * factor / d_model)
    float div_term = expf(-logf(10000.0f) * factor / d_model);
    
    // Multiply position by div_term to get the angle
    float angle = pos * div_term;

    // Use sin for even indices and cos for odd indices
    pe[idx] = (j % 2 == 0) ? sinf(angle) : cosf(angle);
}

void printMatrix(const float* data, int seq_len, int d_model) {
    for (int pos = 0; pos < seq_len; pos++) {
        for (int j = 0; j < d_model; j++) {
            std::cout << data[pos * d_model + j] << "\t";
        }
        std::cout << "\n";
    }
}

int main() {
    const int seq_len = 5;  // sequence length
    const int d_model = 5;  // model dimension
    const int total_elements = seq_len * d_model;

    // Allocate host memory for positional encoding
    float* h_pe = new float[total_elements];

    // Allocate device memory
    float* d_pe = nullptr;
    CHECK_CUDA(cudaMalloc(&d_pe, total_elements * sizeof(float)));

    // Launch the positional encoding kernel
    int threadsPerBlock = 256;
    int blocks = (total_elements + threadsPerBlock - 1) / threadsPerBlock;
    positional_encoding_kernel<<<blocks, threadsPerBlock>>>(d_pe, seq_len, d_model);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Copy the result back to host
    CHECK_CUDA(cudaMemcpy(h_pe, d_pe, total_elements * sizeof(float), cudaMemcpyDeviceToHost));

    // Print the resulting positional encoding matrix
    std::cout << "Positional Encoding (shape: 1 x " << seq_len << " x " << d_model << "):\n";
    printMatrix(h_pe, seq_len, d_model);

    // Cleanup
    delete[] h_pe;
    cudaFree(d_pe);

    return 0;
}
