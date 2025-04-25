#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <random>
#include <cassert>
#include <cstdlib>

#define CHECK_CUDA(call)                                                   \
    do {                                                                  \
        cudaError_t err = (call);                                         \
        if (err != cudaSuccess) {                                         \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err)        \
                      << " (" << __FILE__ << ":" << __LINE__ << ")\n";     \
            exit(EXIT_FAILURE);                                           \
        }                                                                 \
    } while (0)

// Baseline kernel – one thread per (row,col)
__global__ void matmul_naive(const float* __restrict__ Q,
                             const float* __restrict__ K,
                             float* __restrict__ S,
                             int L, int d) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= L || col >= L) return;
    float acc = 0.f;
    for (int k = 0; k < d; ++k) acc += Q[row*d+k] * K[col*d+k];
    S[row*L+col] = acc;
}

// Tiled I/O-aware kernel parameters
constexpr int BLOCK_M = 32;
constexpr int BLOCK_N = 32;
constexpr int BLOCK_K = 64;

__global__ void matmul_tiled(const float* __restrict__ Q,
                             const float* __restrict__ K,
                             float* __restrict__ S,
                             int L, int d) {
    int block_row = blockIdx.y;
    int block_col = blockIdx.x;
    int tx = threadIdx.x;                  // 0..BLOCK_M-1 → row of Q handled by this thread

    extern __shared__ float shQ[];        // BLOCK_M × d

    int row = block_row*BLOCK_M + tx;
    if (row >= L) return;

    float* sh_row = shQ + tx*d;
    // Stage query row once
    for (int k_base = 0; k_base < d; k_base += BLOCK_K) {
        int k = k_base + threadIdx.y;      // y-threads load
        if (threadIdx.y < BLOCK_K && k < d)
            sh_row[k] = Q[row*d + k];
    }
    __syncthreads();

    float acc = 0.f;
    int col = block_col*BLOCK_N + tx;
    if (col >= L) col = -1;

    for (int k_base = 0; k_base < d; k_base += BLOCK_K) {
        float k_reg[BLOCK_K];
        if (col >= 0) {
#pragma unroll
            for (int kk = 0; kk < BLOCK_K; ++kk)
                k_reg[kk] = K[col*d + (k_base + kk)];
        } else {
#pragma unroll
            for (int kk = 0; kk < BLOCK_K; ++kk) k_reg[kk] = 0.f;
        }
#pragma unroll
        for (int kk = 0; kk < BLOCK_K; ++kk)
            acc += sh_row[k_base + kk] * k_reg[kk];
    }
    if (col >= 0) S[row*L + col] = acc;
}

void fill_random(std::vector<float>& v) {
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    for (auto& x : v) x = dist(gen);
}

int main(int argc, char** argv) {
    int L = (argc > 1) ? std::atoi(argv[1]) : 512;
    int d = (argc > 2) ? std::atoi(argv[2]) : 64;
    assert(d % BLOCK_K == 0 && "head_dim must be multiple of BLOCK_K");

    size_t vec_bytes   = static_cast<size_t>(L)*d*sizeof(float);
    size_t score_bytes = static_cast<size_t>(L)*L*sizeof(float);

    std::vector<float> h_Q(L*d), h_K(L*d), h_S(L*L);
    fill_random(h_Q);
    fill_random(h_K);

    float *d_Q, *d_K, *d_S;
    CHECK_CUDA(cudaMalloc(&d_Q, vec_bytes));
    CHECK_CUDA(cudaMalloc(&d_K, vec_bytes));
    CHECK_CUDA(cudaMalloc(&d_S, score_bytes));

    // Create events
    cudaEvent_t h2d1s,h2d1e,k1s,k1e,d2h1s,d2h1e,h2d2s,h2d2e,k2s,k2e,d2h2s,d2h2e;
    for(auto& ev : {&h2d1s,&h2d1e,&k1s,&k1e,&d2h1s,&d2h1e,&h2d2s,&h2d2e,&k2s,&k2e,&d2h2s,&d2h2e}) CHECK_CUDA(cudaEventCreate(ev));

    // Naive kernel
    dim3 blkN(16,16);
    dim3 grdN((L+blkN.x-1)/blkN.x, (L+blkN.y-1)/blkN.y);

    CHECK_CUDA(cudaEventRecord(h2d1s));
    CHECK_CUDA(cudaMemcpy(d_Q, h_Q.data(), vec_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_K, h_K.data(), vec_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaEventRecord(h2d1e));

    CHECK_CUDA(cudaEventRecord(k1s));
    matmul_naive<<<grdN, blkN>>>(d_Q, d_K, d_S, L, d);
    CHECK_CUDA(cudaEventRecord(k1e));

    CHECK_CUDA(cudaEventRecord(d2h1s));
    CHECK_CUDA(cudaMemcpy(h_S.data(), d_S, score_bytes, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaEventRecord(d2h1e));

    CHECK_CUDA(cudaDeviceSynchronize());

    float tH2D1, tKer1, tD2H1;
    CHECK_CUDA(cudaEventElapsedTime(&tH2D1, h2d1s, h2d1e));
    CHECK_CUDA(cudaEventElapsedTime(&tKer1,  k1s, k1e));
    CHECK_CUDA(cudaEventElapsedTime(&tD2H1, d2h1s, d2h1e));

    std::cout << "Naive kernel (L="<<L<<", d="<<d<<")\n";
    std::cout << "  Host-to-Device copy time:      : "<<tH2D1<<" ms\n";
    std::cout << "  Kernel execution time:         : "<<tKer1<<" ms\n";
    std::cout << "  Device-to-Host copy time:      : "<<tD2H1<<" ms\n";
    std::cout << "  Total GPU time:                : "<<(tH2D1+tKer1+tD2H1)<<" ms\n\n";

    // Tiled kernel
    dim3 blkT(BLOCK_M, BLOCK_K>32?1:BLOCK_K);
    dim3 grdT((L+BLOCK_N-1)/BLOCK_N, (L+BLOCK_M-1)/BLOCK_M);
    size_t shmem = BLOCK_M*d*sizeof(float);

    CHECK_CUDA(cudaEventRecord(h2d2s));
    CHECK_CUDA(cudaMemcpy(d_Q, h_Q.data(), vec_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_K, h_K.data(), vec_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaEventRecord(h2d2e));

    CHECK_CUDA(cudaEventRecord(k2s));
    matmul_tiled<<<grdT, blkT, shmem>>>(d_Q, d_K, d_S, L, d);
    CHECK_CUDA(cudaEventRecord(k2e));

    CHECK_CUDA(cudaEventRecord(d2h2s));
    CHECK_CUDA(cudaMemcpy(h_S.data(), d_S, score_bytes, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaEventRecord(d2h2e));

    CHECK_CUDA(cudaDeviceSynchronize());

    float tH2D2, tKer2, tD2H2;
    CHECK_CUDA(cudaEventElapsedTime(&tH2D2, h2d2s, h2d2e));
    CHECK_CUDA(cudaEventElapsedTime(&tKer2,  k2s, k2e));
    CHECK_CUDA(cudaEventElapsedTime(&tD2H2, d2h2s, d2h2e));

    std::cout << "Tiled kernel  (BLOCK_M=N="<<BLOCK_M<<", BLOCK_K="<<BLOCK_K<<")\n";
    std::cout << "  Host-to-Device copy time:      : "<<tH2D2<<" ms\n";
    std::cout << "  Kernel execution time:         : "<<tKer2<<" ms\n";
    std::cout << "  Device-to-Host copy time:      : "<<tD2H2<<" ms\n";
    std::cout << "  Total GPU time:                : "<<(tH2D2+tKer2+tD2H2)<<" ms\n";

    CHECK_CUDA(cudaFree(d_Q));
    CHECK_CUDA(cudaFree(d_K));
    CHECK_CUDA(cudaFree(d_S));
    return 0;
}