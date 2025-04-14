#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include <cstdlib>
#include <cmath>

// Error Checking Macro for CUDA calls
#define cudaCheck(error)                                                   \
    if (error != cudaSuccess) {                                            \
        std::cerr << "CUDA Error: " << cudaGetErrorString(error)           \
                  << " in file " << __FILE__ << " at line " << __LINE__      \
                  << std::endl;                                            \
        exit(EXIT_FAILURE);                                                \
    }

// Parameters
#define IMAGE_HEIGHT     1024
#define IMAGE_WIDTH      1024
#define PATCH_SIZE       32
#define NUM_PATCHES_H    (IMAGE_HEIGHT / PATCH_SIZE)  // 32
#define NUM_PATCHES_W    (IMAGE_WIDTH  / PATCH_SIZE)   // 32
#define NUM_PATCHES      (NUM_PATCHES_H * NUM_PATCHES_W) // 1024
#define MODEL_DIM        128      // Patch embedding and transformer model dimension
#define PATCH_VEC_SIZE   (PATCH_SIZE * PATCH_SIZE) // 32x32 = 1024

// Encoder parameters
#define SEQ_LEN          NUM_PATCHES   // 1024 tokens
#define TILE_WIDTH       16

// CPU Functions

// Patch Embedding: Divide the input image into patches and compute a linear projection
// Each patch (flattened to a vector of size PATCH_VEC_SIZE) is multiplied by a weight matrix
// of size [PATCH_VEC_SIZE x MODEL_DIM].
void cpu_patch_embedding(const float* image, const float* weight, float* output)
{
    for (int patchIdx = 0; patchIdx < NUM_PATCHES; ++patchIdx) {
        int patch_row = patchIdx / NUM_PATCHES_W;
        int patch_col = patchIdx % NUM_PATCHES_W;
        int start_row = patch_row * PATCH_SIZE;
        int start_col = patch_col * PATCH_SIZE;
        for (int d = 0; d < MODEL_DIM; ++d) {
            float sum = 0.0f;
            for (int i = 0; i < PATCH_SIZE; ++i) {
                for (int j = 0; j < PATCH_SIZE; ++j) {
                    int imgRow = start_row + i;
                    int imgCol = start_col + j;
                    float pix = image[imgRow * IMAGE_WIDTH + imgCol];
                    int weightIdx = ((i * PATCH_SIZE) + j) * MODEL_DIM + d;
                    sum += pix * weight[weightIdx];
                }
            }
            output[patchIdx * MODEL_DIM + d] = sum;
        }
    }
}

// Positional Embeddings: Add sinusoidal positional encodings to patch embeddings
void add_positional_embeddings_cpu(float* embeddings, int num_patches, int embedding_dim)
{
    for (int p = 0; p < num_patches; p++) {
        for (int d = 0; d < embedding_dim; d++) {
            float exponent = (2 * (d / 2)) / (float)embedding_dim;
            float angle = p / pow(10000.0f, exponent);
            float pos_val = ((d % 2) == 0) ? sin(angle) : cos(angle);
            embeddings[p * embedding_dim + d] += pos_val;
        }
    }
}

// Matrix Multiplication: C = A * B; A is [M x K], B is [K x N]
void cpuMatMul(const float* A, const float* B, float* C, int M, int K, int N)
{
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// Transpose a matrix: out = transpose(in) where in is [rows x cols]
void cpuTranspose(const float* in, float* out, int rows, int cols)
{
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            out[c * rows + r] = in[r * cols + c];
        }
    }
}

// Row-wise softmax for a matrix [rows x cols]
void cpuSoftmax(float* matrix, int rows, int cols)
{
    for (int r = 0; r < rows; r++) {
        float maxVal = matrix[r * cols];
        for (int j = 1; j < cols; j++) {
            float val = matrix[r * cols + j];
            if (val > maxVal)
                maxVal = val;
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

// Transformer encoder layer (with attention)
void cpu_encoder(const float* h_X, const float* h_Wq, const float* h_Wk,
                 const float* h_Wv, const float* h_Wo, float* h_out)
{
    float* Q  = new float[SEQ_LEN * MODEL_DIM];
    float* K  = new float[SEQ_LEN * MODEL_DIM];
    float* V  = new float[SEQ_LEN * MODEL_DIM];
    float* Kt = new float[SEQ_LEN * MODEL_DIM];
    float* scores    = new float[SEQ_LEN * SEQ_LEN];
    float* attention = new float[SEQ_LEN * MODEL_DIM];

    // Compute Q, K, V = X * W (linear projections)
    cpuMatMul(h_X, h_Wq, Q, SEQ_LEN, MODEL_DIM, MODEL_DIM);
    cpuMatMul(h_X, h_Wk, K, SEQ_LEN, MODEL_DIM, MODEL_DIM);
    cpuMatMul(h_X, h_Wv, V, SEQ_LEN, MODEL_DIM, MODEL_DIM);

    // Transpose K to get Kᵀ
    cpuTranspose(K, Kt, SEQ_LEN, MODEL_DIM);

    // Compute scores = Q * Kᵀ
    cpuMatMul(Q, Kt, scores, SEQ_LEN, MODEL_DIM, SEQ_LEN);

    // Scale scores by 1/sqrt(MODEL_DIM)
    float scale_factor = 1.0f / sqrtf((float)MODEL_DIM);
    for (int i = 0; i < SEQ_LEN * SEQ_LEN; i++)
        scores[i] *= scale_factor;

    // Apply softmax
    cpuSoftmax(scores, SEQ_LEN, SEQ_LEN);

    // Compute attention = scores * V
    cpuMatMul(scores, V, attention, SEQ_LEN, SEQ_LEN, MODEL_DIM);

    // Final output = attention * Wo
    cpuMatMul(attention, h_Wo, h_out, SEQ_LEN, MODEL_DIM, MODEL_DIM);

    // Clean up
    delete[] Q;
    delete[] K;
    delete[] V;
    delete[] Kt;
    delete[] scores;
    delete[] attention;
}

// GPU Kernels

// Patch Embedding Kernel
__global__ void patch_embedding_kernel_tiled(const float* image, const float* weight, float* output,
                                               int imageWidth, int patchSize, int embeddingDim)
{
    int patchIdx = blockIdx.x;
    if (patchIdx >= NUM_PATCHES) return;
    
    int numPatches_w = imageWidth / patchSize;
    int patch_row = patchIdx / numPatches_w;
    int patch_col = patchIdx % numPatches_w;
    int start_row = patch_row * patchSize;
    int start_col = patch_col * patchSize;
    
    // Shared memory for one flattened patch
    __shared__ float s_patch[PATCH_VEC_SIZE];
    
    int tid = threadIdx.x;
    // Load patch pixels into shared memory
    if (tid < PATCH_VEC_SIZE) {
         int i = tid / patchSize;
         int j = tid % patchSize;
         int imgRow = start_row + i;
         int imgCol = start_col + j;
         s_patch[tid] = image[imgRow * imageWidth + imgCol];
    }
    __syncthreads();
    
    // The first 'embeddingDim' threads compute one output element per patch
    if (tid < embeddingDim) {
         float sum = 0.0f;
         #pragma unroll
         for (int k = 0; k < PATCH_VEC_SIZE; ++k) {
              int weightIdx = k * embeddingDim + tid;
              sum += s_patch[k] * weight[weightIdx];
         }
         output[patchIdx * embeddingDim + tid] = sum;
    }
}

// Sinusoidal positional embedding kernel
__global__ void add_positional_embeddings_tiled(float* embeddings, int num_patches, int embedding_dim)
{
    int patch = blockIdx.x;
    if (patch >= num_patches) return;

    extern __shared__ float s_tile[];

    // Process the embedding vector in tiles
    for (int tile_start = 0; tile_start < embedding_dim; tile_start += blockDim.x) {
        int idx = tile_start + threadIdx.x;
        if (idx < embedding_dim) {
            int global_idx = patch * embedding_dim + idx;
            s_tile[threadIdx.x] = embeddings[global_idx];

            float exponent = (2 * (idx / 2)) / (float)embedding_dim;
            float angle = patch / powf(10000.0f, exponent);
            float pos_val = ((idx % 2) == 0) ? sinf(angle) : cosf(angle);
            s_tile[threadIdx.x] += pos_val;
        }
        __syncthreads();

        if (idx < embedding_dim) {
            int global_idx = patch * embedding_dim + idx;
            embeddings[global_idx] = s_tile[threadIdx.x];
        }
        __syncthreads();
    }
}

// Tiled matrix multiplication kernel: C = A * B
// A: [M x K], B: [K x N], C: [M x N]
__global__ void tiledMatMulKernel(const float* A, const float* B, float* C,
                                  int M, int K, int N)
{
    __shared__ float tileA[TILE_WIDTH][TILE_WIDTH];
    __shared__ float tileB[TILE_WIDTH][TILE_WIDTH];

    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;
    float value = 0.0f;

    // Loop through tiles to compute the output value
    for (int t = 0; t < (K + TILE_WIDTH - 1) / TILE_WIDTH; t++) {
        if (row < M && (t * TILE_WIDTH + threadIdx.x) < K)
            tileA[threadIdx.y][threadIdx.x] = A[row * K + t * TILE_WIDTH + threadIdx.x];
        else
            tileA[threadIdx.y][threadIdx.x] = 0.0f;
        if (col < N && (t * TILE_WIDTH + threadIdx.y) < K)
            tileB[threadIdx.y][threadIdx.x] = B[(t * TILE_WIDTH + threadIdx.y) * N + col];
        else
            tileB[threadIdx.y][threadIdx.x] = 0.0f;
        __syncthreads();

        for (int i = 0; i < TILE_WIDTH; i++) {
            value += tileA[threadIdx.y][i] * tileB[i][threadIdx.x];
        }
        __syncthreads();
    }
    if(row < M && col < N)
        C[row * N + col] = value;
}

// Kernel to transpose the matrix: out = transpose(in)
__global__ void transposeKernel(const float* in, float* out, int rows, int cols)
{
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (r < rows && c < cols)
        out[c * rows + r] = in[r * cols + c];
}

// Kernel to scale the matrix by a constant factor
__global__ void scaleKernel(float* matrix, int size, float factor)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < size)
        matrix[idx] *= factor;
}

// Row-wise softmax kernel: one block per row
__global__ void softmaxKernel(float* matrix, int rows, int cols)
{
    int row = blockIdx.x;
    if (row < rows) {
        float maxVal = matrix[row * cols];
        for (int j = 1; j < cols; j++) {
            float val = matrix[row * cols + j];
            if (val > maxVal)
                maxVal = val;
        }
        float sum = 0.0f;
        for (int j = 0; j < cols; j++) {
            float exp_val = expf(matrix[row * cols + j] - maxVal);
            matrix[row * cols + j] = exp_val;
            sum += exp_val;
        }
        for (int j = 0; j < cols; j++) {
            matrix[row * cols + j] /= sum;
        }
    }
}

// Encoder
// Q = X*Wq, K = X*Wk, V = X*Wv, compute scores = Q * Kᵀ, scale, apply softmax,
// compute attention = scores * V, and finally output = attention * Wo.
void gpu_encoder(const float* d_X, const float* d_Wq, const float* d_Wk,
                 const float* d_Wv, const float* d_Wo, float* d_out)
{
    float *d_Q, *d_K, *d_V, *d_Kt, *d_scores, *d_attention;
    size_t size_X_bytes = SEQ_LEN * MODEL_DIM * sizeof(float);
    size_t size_scores_bytes = SEQ_LEN * SEQ_LEN * sizeof(float);
    
    cudaCheck(cudaMalloc(&d_Q, size_X_bytes));
    cudaCheck(cudaMalloc(&d_K, size_X_bytes));
    cudaCheck(cudaMalloc(&d_V, size_X_bytes));
    cudaCheck(cudaMalloc(&d_Kt, size_X_bytes));
    cudaCheck(cudaMalloc(&d_scores, size_scores_bytes));
    cudaCheck(cudaMalloc(&d_attention, size_X_bytes));
    
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid_Q((MODEL_DIM + TILE_WIDTH - 1) / TILE_WIDTH, (SEQ_LEN + TILE_WIDTH - 1) / TILE_WIDTH);
    
    // Compute Q, K, V = X * W
    tiledMatMulKernel<<<dimGrid_Q, dimBlock>>>(d_X, d_Wq, d_Q, SEQ_LEN, MODEL_DIM, MODEL_DIM);
    tiledMatMulKernel<<<dimGrid_Q, dimBlock>>>(d_X, d_Wk, d_K, SEQ_LEN, MODEL_DIM, MODEL_DIM);
    tiledMatMulKernel<<<dimGrid_Q, dimBlock>>>(d_X, d_Wv, d_V, SEQ_LEN, MODEL_DIM, MODEL_DIM);
    
    // Transpose K into Kᵀ
    dim3 dimBlockT(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGridT((MODEL_DIM + TILE_WIDTH - 1) / TILE_WIDTH, (SEQ_LEN + TILE_WIDTH - 1) / TILE_WIDTH);
    transposeKernel<<<dimGridT, dimBlockT>>>(d_K, d_Kt, SEQ_LEN, MODEL_DIM);
    
    // Compute scores = Q * Kᵀ; resulting in a [SEQ_LEN x SEQ_LEN] matrix.
    dim3 dimGrid_scores((SEQ_LEN + TILE_WIDTH - 1) / TILE_WIDTH, (SEQ_LEN + TILE_WIDTH - 1) / TILE_WIDTH);
    tiledMatMulKernel<<<dimGrid_scores, dimBlock>>>(d_Q, d_Kt, d_scores, SEQ_LEN, MODEL_DIM, SEQ_LEN);
    
    // Scale scores by 1/sqrt(MODEL_DIM)
    float scale_factor = 1.0f / sqrtf((float)MODEL_DIM);
    int total_scores = SEQ_LEN * SEQ_LEN;
    int threadsPerBlock = 256;
    int blocks = (total_scores + threadsPerBlock - 1) / threadsPerBlock;
    scaleKernel<<<blocks, threadsPerBlock>>>(d_scores, total_scores, scale_factor);
    
    // Apply row-wise softmax on scores (launching one block per row)
    softmaxKernel<<<SEQ_LEN, 1>>>(d_scores, SEQ_LEN, SEQ_LEN);
    
    // Compute attention = scores * V
    tiledMatMulKernel<<<dimGrid_Q, dimBlock>>>(d_scores, d_V, d_attention, SEQ_LEN, SEQ_LEN, MODEL_DIM);
    
    // Final output = attention * Wo
    tiledMatMulKernel<<<dimGrid_Q, dimBlock>>>(d_attention, d_Wo, d_out, SEQ_LEN, MODEL_DIM, MODEL_DIM);
    
    // Free intermediate buffers
    cudaCheck(cudaFree(d_Q));
    cudaCheck(cudaFree(d_K));
    cudaCheck(cudaFree(d_V));
    cudaCheck(cudaFree(d_Kt));
    cudaCheck(cudaFree(d_scores));
    cudaCheck(cudaFree(d_attention));
}

int main()
{
    // Host memory sizes
    int imageSize = IMAGE_HEIGHT * IMAGE_WIDTH;
    int patchWeightSize = PATCH_VEC_SIZE * MODEL_DIM; // Weight for patch embedding [1024 x 128]
    int patchEmbSize = NUM_PATCHES * MODEL_DIM; // Output of patch embedding [1024 x 128]

    // Allocate and initialize host data
    float* h_image = new float[imageSize];
    float* h_patch_weight = new float[patchWeightSize];
    float* h_patch_emb_cpu = new float[patchEmbSize];  // CPU result after patch embedding + positional encoding

    // Initialize image and patch embedding weight with random values
    for (int i = 0; i < imageSize; i++)
        h_image[i] = static_cast<float>(rand()) / RAND_MAX;
    for (int i = 0; i < patchWeightSize; i++)
        h_patch_weight[i] = static_cast<float>(rand()) / RAND_MAX;

    // Encoder weights for the transformer encoder layer
    int enc_weight_size = MODEL_DIM * MODEL_DIM;
    float* h_Wq = new float[enc_weight_size];
    float* h_Wk = new float[enc_weight_size];
    float* h_Wv = new float[enc_weight_size];
    float* h_Wo = new float[enc_weight_size];
    for (int i = 0; i < enc_weight_size; i++) {
        h_Wq[i] = static_cast<float>(rand()) / RAND_MAX;
        h_Wk[i] = static_cast<float>(rand()) / RAND_MAX;
        h_Wv[i] = static_cast<float>(rand()) / RAND_MAX;
        h_Wo[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // CPU Inference and Timing
    auto cpu_start = std::chrono::high_resolution_clock::now();

    // Patch embedding on CPU
    cpu_patch_embedding(h_image, h_patch_weight, h_patch_emb_cpu);
    // Add positional embeddings
    add_positional_embeddings_cpu(h_patch_emb_cpu, NUM_PATCHES, MODEL_DIM);
    // Run transformer encoder layer
    float* h_out_cpu = new float[patchEmbSize];
    cpu_encoder(h_patch_emb_cpu, h_Wq, h_Wk, h_Wv, h_Wo, h_out_cpu);

    auto cpu_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_time = cpu_end - cpu_start;
    std::cout << "Total CPU inference time (ms): " << cpu_time.count() << std::endl;

    // GPU Inference and Timing Setup
    float *d_image, *d_patch_weight, *d_patch_emb, *d_out;
    float *d_Wq, *d_Wk, *d_Wv, *d_Wo;
    cudaCheck(cudaMalloc(&d_image, imageSize * sizeof(float)));
    cudaCheck(cudaMalloc(&d_patch_weight, patchWeightSize * sizeof(float)));
    cudaCheck(cudaMalloc(&d_patch_emb, patchEmbSize * sizeof(float)));
    cudaCheck(cudaMalloc(&d_out, patchEmbSize * sizeof(float)));
    cudaCheck(cudaMalloc(&d_Wq, enc_weight_size * sizeof(float)));
    cudaCheck(cudaMalloc(&d_Wk, enc_weight_size * sizeof(float)));
    cudaCheck(cudaMalloc(&d_Wv, enc_weight_size * sizeof(float)));
    cudaCheck(cudaMalloc(&d_Wo, enc_weight_size * sizeof(float)));

    // Create CUDA events to measure individual stages
    cudaEvent_t start_total, stop_total;
    cudaEvent_t start_H2D, stop_H2D;
    cudaEvent_t start_kernel, stop_kernel;
    cudaEvent_t start_D2H, stop_D2H;
    cudaCheck(cudaEventCreate(&start_total));
    cudaCheck(cudaEventCreate(&stop_total));
    cudaCheck(cudaEventCreate(&start_H2D));
    cudaCheck(cudaEventCreate(&stop_H2D));
    cudaCheck(cudaEventCreate(&start_kernel));
    cudaCheck(cudaEventCreate(&stop_kernel));
    cudaCheck(cudaEventCreate(&start_D2H));
    cudaCheck(cudaEventCreate(&stop_D2H));

    // Record overall GPU time start
    cudaCheck(cudaEventRecord(start_total, 0));

    // Host-to-Device (H2D) transfers
    cudaCheck(cudaEventRecord(start_H2D, 0));
    cudaCheck(cudaMemcpy(d_image, h_image, imageSize * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_patch_weight, h_patch_weight, patchWeightSize * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_Wq, h_Wq, enc_weight_size * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_Wk, h_Wk, enc_weight_size * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_Wv, h_Wv, enc_weight_size * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_Wo, h_Wo, enc_weight_size * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaEventRecord(stop_H2D, 0));
    cudaCheck(cudaEventSynchronize(stop_H2D));
    float time_H2D;
    cudaCheck(cudaEventElapsedTime(&time_H2D, start_H2D, stop_H2D));

    // Kernel Execution
    cudaCheck(cudaEventRecord(start_kernel, 0));

    // Launch Patch Embedding kernel: one block per patch and PATCH_VEC_SIZE threads per block
    dim3 blockDim_patch(PATCH_VEC_SIZE);
    dim3 gridDim_patch(NUM_PATCHES);
    patch_embedding_kernel_tiled<<<gridDim_patch, blockDim_patch>>>(d_image, d_patch_weight, d_patch_emb,
                                                                      IMAGE_WIDTH, PATCH_SIZE, MODEL_DIM);
    cudaCheck(cudaDeviceSynchronize());

    // Launch Positional Embedding Addition kernel
    int tile_size = 32;
    dim3 blockDim_pos(tile_size);
    dim3 gridDim_pos(NUM_PATCHES);
    size_t sharedMemSize = tile_size * sizeof(float);
    add_positional_embeddings_tiled<<<gridDim_pos, blockDim_pos, sharedMemSize>>>(d_patch_emb, NUM_PATCHES, MODEL_DIM);
    cudaCheck(cudaDeviceSynchronize());

    // Run the transformer encoder kernel using the patched embeddings as input
    gpu_encoder(d_patch_emb, d_Wq, d_Wk, d_Wv, d_Wo, d_out);
    cudaCheck(cudaEventRecord(stop_kernel, 0));
    cudaCheck(cudaEventSynchronize(stop_kernel));
    float time_kernel;
    cudaCheck(cudaEventElapsedTime(&time_kernel, start_kernel, stop_kernel));

    // Device-to-Host (D2H) copy for the final output
    cudaCheck(cudaEventRecord(start_D2H, 0));
    float* h_out_gpu = new float[patchEmbSize];
    cudaCheck(cudaMemcpy(h_out_gpu, d_out, patchEmbSize * sizeof(float), cudaMemcpyDeviceToHost));
    cudaCheck(cudaEventRecord(stop_D2H, 0));
    cudaCheck(cudaEventSynchronize(stop_D2H));
    float time_D2H;
    cudaCheck(cudaEventElapsedTime(&time_D2H, start_D2H, stop_D2H));

    // Overall GPU timer
    cudaCheck(cudaEventRecord(stop_total, 0));
    cudaCheck(cudaEventSynchronize(stop_total));
    float time_total;
    cudaCheck(cudaEventElapsedTime(&time_total, start_total, stop_total));

    // Output GPU timing details
    std::cout << "GPU Timings (ms):" << std::endl;
    std::cout << "  Host-to-Device copy time: " << time_H2D << std::endl;
    std::cout << "  Kernel execution time:    " << time_kernel << std::endl;
    std::cout << "  Device-to-Host copy time: " << time_D2H << std::endl;
    std::cout << "  Total GPU inference time: " << time_total << std::endl;

    // Cleanup Host and Device memory and CUDA events
    delete[] h_image;
    delete[] h_patch_weight;
    delete[] h_patch_emb_cpu;
    delete[] h_out_cpu;
    delete[] h_out_gpu;
    delete[] h_Wq;
    delete[] h_Wk;
    delete[] h_Wv;
    delete[] h_Wo;
    cudaCheck(cudaFree(d_image));
    cudaCheck(cudaFree(d_patch_weight));
    cudaCheck(cudaFree(d_patch_emb));
    cudaCheck(cudaFree(d_out));
    cudaCheck(cudaFree(d_Wq));
    cudaCheck(cudaFree(d_Wk));
    cudaCheck(cudaFree(d_Wv));
    cudaCheck(cudaFree(d_Wo));
    cudaCheck(cudaEventDestroy(start_total));
    cudaCheck(cudaEventDestroy(stop_total));
    cudaCheck(cudaEventDestroy(start_H2D));
    cudaCheck(cudaEventDestroy(stop_H2D));
    cudaCheck(cudaEventDestroy(start_kernel));
    cudaCheck(cudaEventDestroy(stop_kernel));
    cudaCheck(cudaEventDestroy(start_D2H));
    cudaCheck(cudaEventDestroy(stop_D2H));

    return 0;
}