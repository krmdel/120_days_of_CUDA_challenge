#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <random>
#include <cassert>
#include <cfloat>
#include <cstdlib>

#define CHECK_CUDA(x)                                                        \
    do {                                                                     \
        cudaError_t e = (x);                                                 \
        if (e != cudaSuccess) {                                              \
            printf("CUDA %s:%d: %s\n", __FILE__, __LINE__,                   \
                   cudaGetErrorString(e));                                   \
            exit(1);                                                         \
        }                                                                    \
    } while (0)

// Baseline kernels
__global__ void matmul_qk(const float* __restrict__ Q,
                          const float* __restrict__ K,
                          float*       __restrict__ S,
                          int L, int d)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= L || col >= L) return;

    float acc = 0.f;
    for (int k = 0; k < d; ++k) acc += Q[row*d + k] * K[col*d + k];
    S[row*L + col] = acc;
}

__global__ void softmax_inplace(float* __restrict__ S, int L)
{
    extern __shared__ float sh[];
    int row = blockIdx.x, tid = threadIdx.x;

    float v = (tid < L) ? S[row*L + tid] : -FLT_MAX;
    for (int off = warpSize>>1; off; off >>= 1)
        v = fmaxf(v, __shfl_down_sync(0xffffffff, v, off));
    __shared__ float m; if (tid == 0) m = v;  __syncthreads();

    float ex = (tid < L) ? __expf(S[row*L + tid] - m) : 0.f;
    sh[tid] = ex;  __syncthreads();

    float s = ex;
    for (int off = warpSize>>1; off; off >>= 1)
        s += __shfl_down_sync(0xffffffff, s, off);
    __shared__ float l; if (tid == 0) l = s;  __syncthreads();

    if (tid < L) S[row*L + tid] = sh[tid] / l;
}

__global__ void matmul_pv(const float* __restrict__ P,
                          const float* __restrict__ V,
                          float*       __restrict__ O,
                          int L, int d)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= L || col >= d) return;

    float acc = 0.f;
    for (int k = 0; k < L; ++k) acc += P[row*L + k] * V[k*d + col];
    O[row*d + col] = acc;
}

// Cooperative Thread Array (CTA)-tiled FlashAttention
constexpr int MAX_D  = 128;   // register buffer per thread
constexpr int BLOCK_M = 32;   // query rows per CTA
constexpr int BLOCK_K = 64;   // K/V vectors streamed per iteration

__global__ void flash_attention_tiled(const float* __restrict__ Q,
                                      const float* __restrict__ K,
                                      const float* __restrict__ V,
                                      float*       __restrict__ O,
                                      int L, int d)
{
    int q0 = blockIdx.x * BLOCK_M;                 // first query row this CTA owns
    if (q0 >= L) return;

    extern __shared__ float sm[];
    float* shQ = sm;                               // BLOCK_M × MAX_D
    float* shK = shQ + BLOCK_M * MAX_D;            // BLOCK_K × MAX_D
    float* shV = shK + BLOCK_K * MAX_D;            // BLOCK_K × MAX_D

    // Load Q slice once
    for (int idx = threadIdx.x; idx < BLOCK_M*d; idx += blockDim.x) {
        int r = idx / d;
        int c = idx % d;
        if (q0 + r < L)
            shQ[r*MAX_D + c] = Q[(q0 + r)*d + c];
    }
    __syncthreads();

    // Per-thread accumulators
    int q_idx = threadIdx.x;                       // one thread per query row
    if (q_idx >= BLOCK_M) return;

    float o[MAX_D] = {0.f};
    float m = -FLT_MAX, l = 0.f;

    // Stream over K/V tiles
    for (int k0 = 0; k0 < L; k0 += BLOCK_K) {
        int tile = min(BLOCK_K, L - k0);

        // Load a BLOCK_K tile of K & V into shared memory
        for (int idx = threadIdx.x; idx < tile*d; idx += blockDim.x) {
            int r = idx / d; int c = idx % d;
            shK[r*MAX_D + c] = K[(k0 + r)*d + c];
            shV[r*MAX_D + c] = V[(k0 + r)*d + c];
        }
        __syncthreads();

        // Each thread processes its own query row against the tile
        if (q0 + q_idx < L) {
            float* qv = shQ + q_idx*MAX_D;

            for (int j = 0; j < tile; ++j) {
                /* dot(q, k) */
                float s = 0.f;
                #pragma unroll 8
                for (int kk = 0; kk < MAX_D; ++kk)
                    if (kk < d) s += qv[kk] * shK[j*MAX_D + kk];

                /* online softmax update */
                float m_new   = fmaxf(m, s);
                float exp_s   = __expf(s - m_new);
                float l_scale = __expf(m - m_new);
                float l_new   = l * l_scale + exp_s;
                float c1 = exp_s / l_new;          // weight for current V
                float c2 = (l * l_scale) / l_new;  // scale existing output

                #pragma unroll 8
                for (int kk = 0; kk < MAX_D; ++kk)
                    if (kk < d) {
                        float v = shV[j*MAX_D + kk];
                        o[kk] = o[kk] * c2 + c1 * v;
                    }

                m = m_new; l = l_new;
            }
        }
        __syncthreads();
    }

    // Write result
    if (q0 + q_idx < L)
        for (int kk = 0; kk < d; ++kk) O[(q0 + q_idx)*d + kk] = o[kk];
}


// Utility helpers
void fill_random(std::vector<float>& v)
{
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    for (auto& x : v) x = dist(gen);
}

int main(int argc, char** argv)
{
    int L = (argc > 1) ? std::atoi(argv[1]) : 512;   // sequence length
    int d = (argc > 2) ? std::atoi(argv[2]) : 64;    // head dimension
    assert(d <= MAX_D);

    size_t vec_bytes   = static_cast<size_t>(L) * d * sizeof(float);
    size_t score_bytes = static_cast<size_t>(L) * L * sizeof(float);

    std::vector<float> hQ(L*d), hK(L*d), hV(L*d), hO(L*d);
    fill_random(hQ); fill_random(hK); fill_random(hV);

    float *dQ, *dK, *dV, *dS, *dO;
    CHECK_CUDA(cudaMalloc(&dQ, vec_bytes));
    CHECK_CUDA(cudaMalloc(&dK, vec_bytes));
    CHECK_CUDA(cudaMalloc(&dV, vec_bytes));
    CHECK_CUDA(cudaMalloc(&dO, vec_bytes));
    CHECK_CUDA(cudaMalloc(&dS, score_bytes));

    /* Create events */
    enum {H2D1s,H2D1e,K1s,K1e,D2H1s,D2H1e,H2D2s,H2D2e,K2s,K2e,D2H2s,D2H2e,EV_COUNT};
    cudaEvent_t ev[EV_COUNT];
    for (int i = 0; i < EV_COUNT; ++i) CHECK_CUDA(cudaEventCreate(&ev[i]));

    // Baseline
    CHECK_CUDA(cudaEventRecord(ev[H2D1s]));
    CHECK_CUDA(cudaMemcpy(dQ, hQ.data(), vec_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dK, hK.data(), vec_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dV, hV.data(), vec_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaEventRecord(ev[H2D1e]));

    dim3 blkQK(16,16), grdQK((L+15)/16, (L+15)/16);
    dim3 blkSoft(256),  grdSoft(L);
    dim3 blkPV(16,16),  grdPV((d+15)/16, (L+15)/16);

    CHECK_CUDA(cudaEventRecord(ev[K1s]));
    matmul_qk<<<grdQK, blkQK>>>(dQ, dK, dS, L, d);
    softmax_inplace<<<grdSoft, blkSoft, L*sizeof(float)>>>(dS, L);
    matmul_pv<<<grdPV, blkPV>>>(dS, dV, dO, L, d);
    CHECK_CUDA(cudaEventRecord(ev[K1e]));

    CHECK_CUDA(cudaEventRecord(ev[D2H1s]));
    CHECK_CUDA(cudaMemcpy(hO.data(), dO, vec_bytes, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaEventRecord(ev[D2H1e]));
    CHECK_CUDA(cudaDeviceSynchronize());

    float tH2D1, tKer1, tD2H1;
    CHECK_CUDA(cudaEventElapsedTime(&tH2D1, ev[H2D1s], ev[H2D1e]));
    CHECK_CUDA(cudaEventElapsedTime(&tKer1, ev[K1s],  ev[K1e]));
    CHECK_CUDA(cudaEventElapsedTime(&tD2H1, ev[D2H1s], ev[D2H1e]));

    std::cout << "Baseline (L=" << L << ", d=" << d << ")\n"
              << "  H2D   : " << tH2D1 << " ms\n"
              << "  Kernels: " << tKer1 << " ms\n"
              << "  D2H   : " << tD2H1 << " ms\n"
              << "  Total : " << (tH2D1 + tKer1 + tD2H1) << " ms\n\n";

    // FlashAttention
    CHECK_CUDA(cudaEventRecord(ev[H2D2s]));
    CHECK_CUDA(cudaMemcpy(dQ, hQ.data(), vec_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dK, hK.data(), vec_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dV, hV.data(), vec_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaEventRecord(ev[H2D2e]));

    dim3 blkFA(BLOCK_M);
    dim3 grdFA((L + BLOCK_M - 1) / BLOCK_M);
    size_t smem = (BLOCK_M + BLOCK_K) * MAX_D * 2 * sizeof(float);

    CHECK_CUDA(cudaEventRecord(ev[K2s]));
    flash_attention_tiled<<<grdFA, blkFA, smem>>>(dQ, dK, dV, dO, L, d);
    CHECK_CUDA(cudaEventRecord(ev[K2e]));

    CHECK_CUDA(cudaEventRecord(ev[D2H2s]));
    CHECK_CUDA(cudaMemcpy(hO.data(), dO, vec_bytes, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaEventRecord(ev[D2H2e]));
    CHECK_CUDA(cudaDeviceSynchronize());

    float tH2D2, tKer2, tD2H2;
    CHECK_CUDA(cudaEventElapsedTime(&tH2D2, ev[H2D2s], ev[H2D2e]));
    CHECK_CUDA(cudaEventElapsedTime(&tKer2, ev[K2s],  ev[K2e]));
    CHECK_CUDA(cudaEventElapsedTime(&tD2H2, ev[D2H2s], ev[D2H2e]));

    std::cout << "FlashAttention-CTA (L=" << L << ", d=" << d << ")\n"
              << "  H2D   : " << tH2D2 << " ms\n"
              << "  Kernel: " << tKer2 << " ms\n"
              << "  D2H   : " << tD2H2 << " ms\n"
              << "  Total : " << (tH2D2 + tKer2 + tD2H2) << " ms\n";

    // Cleanup 
    cudaFree(dQ); cudaFree(dK); cudaFree(dV);
    cudaFree(dS); cudaFree(dO);
    return 0;
}
