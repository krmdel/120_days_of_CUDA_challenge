#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <iostream>

#ifndef SIG_LEN
#define SIG_LEN 16384        // N  – matches earlier demos
#endif
#ifndef NUM_BINS
#define NUM_BINS 64          // K  – choose ≪ SIG_LEN
#endif
static_assert(SIG_LEN > 0 && NUM_BINS > 0, "SIG_LEN and NUM_BINS must be >0");

#define CUDA_CHECK(x) \
    do { cudaError_t rc=(x); if(rc!=cudaSuccess){                               \
        std::fprintf(stderr,"CUDA %s @ %s:%d\n", cudaGetErrorString(rc),        \
                     __FILE__,__LINE__); std::exit(EXIT_FAILURE);} } while(0)

struct Cpx { float re, im; };

// Complex helpers (device)
__device__ __forceinline__ Cpx c_add(Cpx a, Cpx b){ return {a.re+b.re, a.im+b.im}; }
__device__ __forceinline__ Cpx c_mul(Cpx a, Cpx b){
    return {a.re*b.re - a.im*b.im, a.re*b.im + a.im*b.re};
}
__device__ __forceinline__ Cpx c_exp(float th){   // e^{j th}
    float s, c; sincosf(th, &s, &c); return {c, s};
}

// Goertzel kernel
// One block = one requested bin k (from kList)
// Each thread accumulates a slice of the N-sample recursion.
 
__global__ void goertzel_kernel(const float* __restrict__ x,   // real signal
                                const int*   __restrict__ kList,
                                Cpx*         __restrict__ Xout,
                                int          N)
{
    extern __shared__ float sh[];           // two floats per block: s0, s1
    float &s_prev = sh[0];
    float &s_prev2= sh[1];

    int k = kList[blockIdx.x];              // DFT index for this block
    float omega   = 2.f * M_PI * k / N;
    float coef    = 2.f * cosf(omega);
    float sine    = sinf(omega);            // needed for imag part later

    // Local accumulators per thread
    float sp = 0.f, sp2 = 0.f;

    // Sride over the signal
    for(int n=threadIdx.x; n<N; n+=blockDim.x){
        float s  = x[n] + coef*sp - sp2;
        sp2 = sp;
        sp  = s;
    }

    // Warp reduction of sp and sp2
    for(int off=16; off; off>>=1){
        sp  += __shfl_down_sync(0xffffffff, sp , off);
        sp2 += __shfl_down_sync(0xffffffff, sp2, off);
    }

    // Thread 0 writes block sums to shared mem
    if((threadIdx.x & 31) == 0){
        atomicAdd(&s_prev , sp );
        atomicAdd(&s_prev2, sp2);
    }
    __syncthreads();

    // Thread 0 finalises this bin
    if(threadIdx.x == 0){
        // = s_prev - e^{−jω} s_prev2 ,    e^{−jω}=cos−jsin
        float real = s_prev -  cosf(omega)*s_prev2;
        float imag =       -(-sine)       *s_prev2;   // = −(−sin)*s_prev2
        Xout[blockIdx.x] = {real, imag};

        // Reset shared for next launch
        s_prev = s_prev2 = 0.f;
    }
}


int main()
{
    constexpr int N = SIG_LEN;
    constexpr int K = NUM_BINS;

    std::cout<<"Sparse FFT (Goertzel bank)  N="<<N<<"  K="<<K<<"\n";

    // Host signal
    std::vector<float> h_x(N);
    std::mt19937 rng(1); std::uniform_real_distribution<float> dist(-1,1);
    for(float& v : h_x) v = dist(rng);

    // Requested bin list
    std::vector<int> h_ks(K);
    for(int i=0;i<K;++i) h_ks[i] = (i+1)*5 % N;   // arbitrary unique bins

    // Device buffers
    float *d_x;  int *d_k;  Cpx *d_X;
    CUDA_CHECK(cudaMalloc(&d_x, sizeof(float)*N));
    CUDA_CHECK(cudaMalloc(&d_k, sizeof(int)*K));
    CUDA_CHECK(cudaMalloc(&d_X, sizeof(Cpx)*K));

    // Events for timing
    cudaEvent_t t0,t1,t2,t3;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventCreate(&t2); cudaEventCreate(&t3);

    // Host → Device
    cudaEventRecord(t0);
    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), sizeof(float)*N, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_k, h_ks.data(), sizeof(int)*K , cudaMemcpyHostToDevice));
    cudaEventRecord(t1);

    // Kernel
    int threads = 256;
    size_t shMem = 2*sizeof(float);          // s_prev & s_prev2
    goertzel_kernel<<<K, threads, shMem>>>(d_x, d_k, d_X, N);
    cudaEventRecord(t2);

    // Device → Host
    std::vector<Cpx> h_X(K);
    CUDA_CHECK(cudaMemcpy(h_X.data(), d_X, sizeof(Cpx)*K, cudaMemcpyDeviceToHost));
    cudaEventRecord(t3); cudaEventSynchronize(t3);

    // Timings
    float tH2D, tKer, tD2H;
    cudaEventElapsedTime(&tH2D, t0, t1);
    cudaEventElapsedTime(&tKer, t1, t2);
    cudaEventElapsedTime(&tD2H, t2, t3);

    std::cout<<"Host → Device copy : "<<tH2D<<" ms\n";
    std::cout<<"Goertzel kernel    : "<<tKer <<" ms\n";
    std::cout<<"Device → Host copy : "<<tD2H<<" ms\n";
    std::cout<<"Total GPU time     : "<<tH2D+tKer+tD2H<<" ms\n";

    cudaFree(d_x); cudaFree(d_k); cudaFree(d_X);
    return 0;
}