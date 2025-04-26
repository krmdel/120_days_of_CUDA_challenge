#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <random>
#include <cassert>
#include <cmath>
#include <cstdlib>

#define CHECK_CUDA(call)                                                  \
    do {                                                                  \
        cudaError_t err = (call);                                         \
        if (err != cudaSuccess) {                                         \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err)        \
                      << " (" << __FILE__ << ":" << __LINE__ << ")\n";     \
            exit(EXIT_FAILURE);                                           \
        }                                                                 \
    } while (0)

// Vanilla softmax

__global__ void softmax_naive(const float* __restrict__ S,
                              float* __restrict__ P,
                              int L)
{
    extern __shared__ float sh[];   // size = L

    int row = blockIdx.x;           // one block per row
    int tid = threadIdx.x;

    // 1st pass: find max
    float v = (tid < L) ? S[row * L + tid] : -FLT_MAX;
    // parallel reduction to row_max
    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        float other = __shfl_down_sync(0xffffffff, v, offset);
        v = fmaxf(v, other);
    }
    float row_max = v;
    row_max = __shfl_sync(0xffffffff, row_max, 0); // broadcast leader’s value

    // 2nd pass: compute exp(x - max) and row sum
    float ex = (tid < L) ? __expf(S[row * L + tid] - row_max) : 0.f;
    sh[tid] = ex;   // store for later normalization
    __syncthreads();

    // reduction to sum
    float s = ex;
    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        s += __shfl_down_sync(0xffffffff, s, offset);
    }
    float row_sum = s;
    row_sum = __shfl_sync(0xffffffff, row_sum, 0);

    // write normalized probabilities
    if (tid < L)
        P[row * L + tid] = sh[tid] / row_sum;
}


// Online streaming softmax

// Pass 1: compute row max
// Pass 2: compute row sum of exp(x - max)
// Pass 3: write probabilities  exp(x - max)/sum
// Only a few scalars per row kept in registers; no shared row buffer

__global__ void softmax_online(const float* __restrict__ S,
                               float* __restrict__ P,
                               int L)
{
    int row = blockIdx.x;
    int tid = threadIdx.x;

    // Pass 1: max
    float local_max = -FLT_MAX;
    for (int i = tid; i < L; i += blockDim.x) {
        local_max = fmaxf(local_max, S[row * L + i]);
    }
    // reduce within block
    for (int offset = warpSize >> 1; offset > 0; offset >>= 1)
        local_max = fmaxf(local_max, __shfl_down_sync(0xffffffff, local_max, offset));
    __shared__ float row_max;
    if ((threadIdx.x & (warpSize - 1)) == 0) {
        atomicMax((int*)&row_max, __float_as_int(local_max)); // store as bits
    }
    __syncthreads();
    float m = __int_as_float(*((int*)&row_max));

    // Pass 2: sum exp
    float local_sum = 0.f;
    for (int i = tid; i < L; i += blockDim.x)
        local_sum += __expf(S[row * L + i] - m);

    // reduce sum
    for (int offset = warpSize >> 1; offset > 0; offset >>= 1)
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
    __shared__ float row_sum;
    if ((threadIdx.x & (warpSize - 1)) == 0) atomicAdd(&row_sum, local_sum);
    __syncthreads();
    float s = row_sum;

    // Pass 3: write probs
    for (int i = tid; i < L; i += blockDim.x) {
        float p = __expf(S[row * L + i] - m) / s;
        P[row * L + i] = p;
    }
}

void fill_random(std::vector<float>& v) {
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> d(-1.f, 1.f);
    for (auto& x : v) x = d(gen);
}

int main(int argc, char** argv) {
    int L = (argc > 1) ? std::atoi(argv[1]) : 2048;      // row length
    int rows = L;                                        // square matrix L×L

    size_t bytes = static_cast<size_t>(rows) * L * sizeof(float);

    std::vector<float> h_S(bytes / sizeof(float)), h_P(bytes / sizeof(float));
    fill_random(h_S);

    float *d_S, *d_P;
    CHECK_CUDA(cudaMalloc(&d_S, bytes));
    CHECK_CUDA(cudaMalloc(&d_P, bytes));

    // events
    cudaEvent_t H2D1s,H2D1e,K1s,K1e,D2H1s,D2H1e,H2D2s,H2D2e,K2s,K2e,D2H2s,D2H2e;
    for(auto* e : {&H2D1s,&H2D1e,&K1s,&K1e,&D2H1s,&D2H1e,&H2D2s,&H2D2e,&K2s,&K2e,&D2H2s,&D2H2e}) CHECK_CUDA(cudaEventCreate(e));


    // Naive softmax
    dim3 blk1(256);
    dim3 grd1(rows);
    size_t shmem_naive = L * sizeof(float);

    CHECK_CUDA(cudaEventRecord(H2D1s));
    CHECK_CUDA(cudaMemcpy(d_S, h_S.data(), bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaEventRecord(H2D1e));

    CHECK_CUDA(cudaEventRecord(K1s));
    softmax_naive<<<grd1, blk1, shmem_naive>>>(d_S, d_P, L);
    CHECK_CUDA(cudaEventRecord(K1e));

    CHECK_CUDA(cudaEventRecord(D2H1s));
    CHECK_CUDA(cudaMemcpy(h_P.data(), d_P, bytes, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaEventRecord(D2H1e));

    CHECK_CUDA(cudaDeviceSynchronize());

    float tH2D1,tKer1,tD2H1;
    CHECK_CUDA(cudaEventElapsedTime(&tH2D1,H2D1s,H2D1e));
    CHECK_CUDA(cudaEventElapsedTime(&tKer1,K1s,K1e));
    CHECK_CUDA(cudaEventElapsedTime(&tD2H1,D2H1s,D2H1e));

    std::cout << "Naive softmax (L="<<L<<")\n";
    std::cout << "  Host-to-Device copy time:      : "<<tH2D1<<" ms\n";
    std::cout << "  Kernel execution time:         : "<<tKer1<<" ms\n";
    std::cout << "  Device-to-Host copy time:      : "<<tD2H1<<" ms\n";
    std::cout << "  Total GPU time:                : "<<(tH2D1+tKer1+tD2H1)<<" ms\n\n";

    // Online softmax
    dim3 blk2(256);
    dim3 grd2(rows);

    CHECK_CUDA(cudaEventRecord(H2D2s));
    CHECK_CUDA(cudaMemcpy(d_S, h_S.data(), bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaEventRecord(H2D2e));

    CHECK_CUDA(cudaEventRecord(K2s));
    softmax_online<<<grd2, blk2>>>(d_S, d_P, L);
    CHECK_CUDA(cudaEventRecord(K2e));

    CHECK_CUDA(cudaEventRecord(D2H2s));
    CHECK_CUDA(cudaMemcpy(h_P.data(), d_P, bytes, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaEventRecord(D2H2e));

    CHECK_CUDA(cudaDeviceSynchronize());

    float tH2D2,tKer2,tD2H2;
    CHECK_CUDA(cudaEventElapsedTime(&tH2D2,H2D2s,H2D2e));
    CHECK_CUDA(cudaEventElapsedTime(&tKer2,K2s,K2e));
    CHECK_CUDA(cudaEventElapsedTime(&tD2H2,D2H2s,D2H2e));

    std::cout << "Online softmax (L="<<L<<")\n";
    std::cout << "  Host-to-Device copy time:      : "<<tH2D2<<" ms\n";
    std::cout << "  Kernel execution time:         : "<<tKer2<<" ms\n";
    std::cout << "  Device-to-Host copy time:      : "<<tD2H2<<" ms\n";
    std::cout << "  Total GPU time:                : "<<(tH2D2+tKer2+tD2H2)<<" ms\n";

    CHECK_CUDA(cudaFree(d_S));
    CHECK_CUDA(cudaFree(d_P));
    return 0;
}