#include <cuda_runtime.h>

#include <vector>
#include <random>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <cstdlib>
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#define MAX_TAPS 1024        // constant-memory upper bound

// Error helper
#define CUDA_CHECK(x)  do{ cudaError_t rc=(x); if(rc!=cudaSuccess){            \
    std::cerr<<"CUDA "<<cudaGetErrorString(rc)<<" @"<<__FILE__<<":"<<__LINE__;\
    std::exit(EXIT_FAILURE);} }while(0)

// Taps in constant memory
__constant__ float d_taps_const[MAX_TAPS];

// Kernel: one thread → one output, global memory only
__global__ void fir_naive_kernel(const float* __restrict__ x,
                                 float*       __restrict__ y,
                                 int N, int T)
{
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if(n >= N) return;

    float acc = 0.f;
    #pragma unroll 4
    for(int k = 0; k < T; ++k){
        int idx = n - k;
        if(idx >= 0) acc += d_taps_const[k] * x[idx];
    }
    y[n] = acc;
}

// Shared-memory tiled kernel
template<int TILE> __global__
void fir_tiled_kernel(const float* __restrict__ x,
                      float*       __restrict__ y,
                      int N, int T)
{
    extern __shared__ float s_data[];              // size = TILE + T − 1
    const int gid  = blockIdx.x * TILE + threadIdx.x;   // output index
    const int halo = T - 1;
    const int base = blockIdx.x * TILE - halo;          // first element needed

    // Cooperative load: every thread loads ≥1 input samples
    for(int offset = threadIdx.x; offset < TILE + halo; offset += TILE){
        int gidx = base + offset;
        s_data[offset] = (gidx >= 0 && gidx < N) ? x[gidx] : 0.f;
    }
    __syncthreads();

    if(gid >= N) return;

    float acc = 0.f;
    #pragma unroll 8
    for(int k = 0; k < MAX_TAPS; ++k){          // unrolled to 8-way
        if(k >= T) break;
        acc += d_taps_const[k] * s_data[threadIdx.x + halo - k];
    }
    y[gid] = acc;
}

struct Times { float h2d, ker, d2h; float total()const{return h2d+ker+d2h;} };

Times run_kernel(const float* d_sig, float* d_out, int N, int T,
                 void(*kptr)(const float*,float*,int,int),
                 size_t shMem = 0)
{
    // Events
    cudaEvent_t t0,t1,t2,t3;
    for(auto& e : {&t0,&t1,&t2,&t3}) cudaEventCreate(e);

    // H2D
    cudaEventRecord(t0);
    const int THREADS = 256;
    const int BLOCKS  = (N + THREADS - 1) / THREADS;
    (*kptr)<<<BLOCKS, THREADS, shMem>>>(d_sig, d_out, N, T);
    cudaEventRecord(t1);

    std::vector<float> h_tmp(N);
    CUDA_CHECK(cudaMemcpyAsync(h_tmp.data(), d_out,
                               sizeof(float)*N, cudaMemcpyDeviceToHost));
    cudaEventRecord(t2);
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEventRecord(t3);

    float ker_ms, d2h_ms;
    cudaEventElapsedTime(&ker_ms, t0, t1);
    cudaEventElapsedTime(&d2h_ms, t1, t2);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    cudaEventDestroy(t2); cudaEventDestroy(t3);
    return {0.0f, ker_ms, d2h_ms};
}

int main(int argc, char** argv)
{
    const int N = (argc > 1) ? std::atoi(argv[1]) : 1'048'576;   // 2^20
    const int T = (argc > 2) ? std::atoi(argv[2]) : 255;
    if(T > MAX_TAPS){ std::cerr<<"T > MAX_TAPS ("<<MAX_TAPS<<")\n"; return 1; }

    std::cout << "Signal length N = " << N << "   taps = " << T << "\n\n";

    // Generate host data
    std::vector<float> h_sig(N), h_taps(T);
    std::mt19937 rng(0);
    std::uniform_real_distribution<float> dist(-1.f,1.f);
    for(float& v : h_sig ) v = dist(rng);
    for(float& v : h_taps) v = dist(rng);

    // Device buffers
    float *d_sig, *d_out_naive, *d_out_tile;
    CUDA_CHECK(cudaMalloc(&d_sig       , sizeof(float)*N));
    CUDA_CHECK(cudaMalloc(&d_out_naive , sizeof(float)*N));
    CUDA_CHECK(cudaMalloc(&d_out_tile  , sizeof(float)*N));

    // Copy taps to constant memory once
    CUDA_CHECK(cudaMemcpyToSymbol(d_taps_const, h_taps.data(), sizeof(float)*T));

    // Measure Host → Device copy once (signal only)
    cudaEvent_t eH0,eH1;
    cudaEventCreate(&eH0); cudaEventCreate(&eH1);
    cudaEventRecord(eH0);
    CUDA_CHECK(cudaMemcpy(d_sig, h_sig.data(), sizeof(float)*N,
                          cudaMemcpyHostToDevice));
    cudaEventRecord(eH1);  CUDA_CHECK(cudaDeviceSynchronize());
    float h2d_ms; cudaEventElapsedTime(&h2d_ms, eH0, eH1);
    cudaEventDestroy(eH0); cudaEventDestroy(eH1);

    // Run naïve
    Times t_naive = run_kernel(d_sig, d_out_naive, N, T,
                               [](const float* s, float* o,int n,int t){
                                   fir_naive_kernel<<<(n+255)/256,256>>>(s,o,n,t);
                               });

    // Run tiled (shared)
    constexpr int TILE = 256;
    size_t shBytes = (TILE + MAX_TAPS - 1) * sizeof(float);
    Times t_tile = run_kernel(d_sig, d_out_tile, N, T,
                              [](const float* s,float*o,int n,int t){
                                  fir_tiled_kernel<TILE><<<(n+TILE-1)/TILE,
                                                         TILE,
                                                         (TILE+MAX_TAPS-1)*sizeof(float)>>>
                                                         (s,o,n,t);
                              },
                              shBytes);

    // Add common H2D time into totals
    t_naive.h2d = t_tile.h2d = h2d_ms;

    // Print timings
    auto prn=[&](const char* name,const Times& tm){
        std::cout<<std::setw(20)<<std::left<<name
                 <<" H2D "<<std::setw(7)<<tm.h2d
                 <<" Kern "<<std::setw(9)<<tm.ker
                 <<" D2H "<<std::setw(7)<<tm.d2h
                 <<" Total "<<tm.total()<<" ms\n";
    };
    std::cout << std::fixed << std::setprecision(3);
    prn("Naïve (global)", t_naive);
    prn("Shared-mem tile", t_tile);

    // Cleanup
    cudaFree(d_sig); cudaFree(d_out_naive); cudaFree(d_out_tile);
    return 0;
}