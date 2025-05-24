/****************************************************************************************
 *  fir_naive.cu  – Day-82: FIR filtering (one thread per output sample)
 *
 *  Build :  nvcc -O3 -std=c++17 fir_naive.cu -o fir
 *  Run   :  ./fir  [N] [T]
 *           N … signal length     (default 1 048 576)
 *           T … number of taps    (default 63)
 *****************************************************************************************/
#include <cuda_runtime.h>

#include <vector>
#include <random>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <cstdlib>
#include <algorithm>        // std::max

// Error helpers
#define CUDA_CHECK(x)  do{ cudaError_t rc=(x); if(rc!=cudaSuccess){            \
    std::cerr<<"CUDA "<<cudaGetErrorString(rc)<<" @"<<__FILE__<<":"<<__LINE__;\
    std::exit(EXIT_FAILURE);} }while(0)

// FIR kernel
__global__ void fir_naive_kernel(const float* __restrict__ x,
                                 const float* __restrict__ h,
                                 float*       __restrict__ y,
                                 int N, int T)
{
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if(n >= N) return;

    float acc = 0.f;
    #pragma unroll 4
    for(int k = 0; k < T; ++k){
        int idx = n - k;
        if(idx >= 0) acc += h[k] * x[idx];          // zero-padding for idx<0
    }
    y[n] = acc;
}

int main(int argc, char** argv)
{
    const int N = (argc > 1) ? std::atoi(argv[1]) : 1'048'576;   // 2^20
    const int T = (argc > 2) ? std::atoi(argv[2]) : 63;
    std::cout << "Naïve FIR  —  N = " << N << ",  taps = " << T << "\n\n";

    // Host data
    std::vector<float> h_sig(N), h_taps(T);
    std::mt19937 rng(0);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    for(float& v : h_sig ) v = dist(rng);
    for(float& v : h_taps) v = dist(rng);

    // Device buffers
    float *d_sig, *d_taps, *d_out;
    CUDA_CHECK(cudaMalloc(&d_sig , sizeof(float)*N));
    CUDA_CHECK(cudaMalloc(&d_taps, sizeof(float)*T));
    CUDA_CHECK(cudaMalloc(&d_out , sizeof(float)*N));

    // Timing events
    cudaEvent_t eH2D0, eH2D1, eK0, eK1, eD2H0, eD2H1;
    for(auto& ev : {&eH2D0,&eH2D1,&eK0,&eK1,&eD2H0,&eD2H1})
        cudaEventCreate(ev);

    // Host → Device
    cudaEventRecord(eH2D0);
    CUDA_CHECK(cudaMemcpy(d_sig , h_sig .data(), sizeof(float)*N, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_taps, h_taps.data(), sizeof(float)*T, cudaMemcpyHostToDevice));
    cudaEventRecord(eH2D1);

    // Kernel launch
    const int THREADS = 256;
    const int BLOCKS  = (N + THREADS - 1) / THREADS;
    cudaEventRecord(eK0);
    fir_naive_kernel<<<BLOCKS, THREADS>>>(d_sig, d_taps, d_out, N, T);
    cudaEventRecord(eK1);

    // Device → Host
    std::vector<float> h_out(N);
    cudaEventRecord(eD2H0);
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, sizeof(float)*N, cudaMemcpyDeviceToHost));
    cudaEventRecord(eD2H1);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Timings
    float h2d_ms, ker_ms, d2h_ms;
    cudaEventElapsedTime(&h2d_ms, eH2D0, eH2D1);
    cudaEventElapsedTime(&ker_ms, eK0 , eK1 );
    cudaEventElapsedTime(&d2h_ms, eD2H0, eD2H1);

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "GPU H2D copy : " << h2d_ms << " ms\n";
    std::cout << "GPU Kernel   : " << ker_ms << " ms\n";
    std::cout << "GPU D2H copy : " << d2h_ms << " ms\n";
    std::cout << "GPU Total    : " << (h2d_ms + ker_ms + d2h_ms) << " ms\n";

    // Cleanup
    cudaFree(d_sig); cudaFree(d_taps); cudaFree(d_out);
    for(auto ev : {eH2D0,eH2D1,eK0,eK1,eD2H0,eD2H1}) cudaEventDestroy(ev);
    return 0;
}