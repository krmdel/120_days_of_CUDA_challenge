#include <cuda_runtime.h>
#include <cufft.h>
#include <vector>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <iostream>

#ifndef BLU_LEN
#define BLU_LEN 16384
#endif
static_assert(BLU_LEN > 2, "length must be >2");

// CUDA helpers
#define CUDA_CHECK(x) \
    do { cudaError_t rc=(x); if(rc!=cudaSuccess){                             \
        fprintf(stderr,"CUDA %s @ %s:%d\n", cudaGetErrorString(rc),           \
                __FILE__,__LINE__); std::exit(EXIT_FAILURE);} } while(0)
#define CUFFT_CHECK(x) \
    do { cufftResult rc=(x); if(rc!=CUFFT_SUCCESS){                           \
        fprintf(stderr,"cuFFT error %d @ %s:%d\n",rc,__FILE__,__LINE__);      \
        std::exit(EXIT_FAILURE);} } while(0)

__device__ __forceinline__ cufftComplex cmul(cufftComplex a, cufftComplex b){
    cufftComplex r; r.x=a.x*b.x - a.y*b.y; r.y=a.x*b.y + a.y*b.x; return r;
}
__device__ __forceinline__ cufftComplex cexpf_wrap(float theta){
    cufftComplex r; sincosf(theta,&r.y,&r.x); return r;       // cos+jsin
}

// Kernel 1 – build chirp-weighted a[n] and b[n] (0≤n<N)
__global__ void build_chirp(const cufftComplex* x,
                            cufftComplex*       A,
                            cufftComplex*       B,
                            int                 Nlen,
                            float               pi_over_N)
{
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if(n >= Nlen) return;
    float ang   =  pi_over_N * n * n;         //  π n² / N
    cufftComplex w_n   = cexpf_wrap( ang);
    cufftComplex w_n_m = cexpf_wrap(-ang);

    A[n] = cmul(x[n], w_n);                   // a[n] = x[n]·w^{ n² }
    B[n] = w_n_m;                             // b[n] =          w^{−n²}
}

//  Kernel 2 – zero-pad B to length M and copy mirrored tail
__global__ void pad_B(cufftComplex* B, int Nlen, int Mlen){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i >= Mlen) return;

    if(i >= Nlen && i < Mlen - Nlen + 1) {        // middle region → 0
        B[i].x = B[i].y = 0.0f;
    }
    // Mirror the first (N-1) non-zero taps into the tail */
    if(i > 0 && i < Nlen){
        B[Mlen - i] = B[i];
    }
}

// Kernel 3 – element-wise complex product
__global__ void pointwise_mul(const cufftComplex* A,
                              const cufftComplex* B,
                              cufftComplex*       C,
                              int                 Mlen)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i < Mlen) C[i] = cmul(A[i], B[i]);
}

// Kernel 4 – final chirp & IFFT scaling
__global__ void final_chirp(const cufftComplex* y,
                            cufftComplex*       X,
                            int                 Nlen,
                            int                 Mlen,
                            float               pi_over_N)
{
    int k = blockIdx.x*blockDim.x + threadIdx.x;
    if(k >= Nlen) return;
    float ang = -pi_over_N * k * k;           // −π k² / N
    cufftComplex w_k = cexpf_wrap(ang);
    cufftComplex val = y[k];
    val.x /= Mlen;  val.y /= Mlen;            // normalise IFFT
    X[k]   = cmul(val, w_k);                  // X[k] = w^{−k²}·y[k]/M
}

// Host utility – next power-of-two
int next_pow2(int n){ int p=1; while(p<n) p<<=1; return p; }

int main()
{
    const int N = BLU_LEN;
    const int M = next_pow2(2*N - 1);         // convolution length
    const float pi_over_N = M_PI / N;

    std::cout << "Bluestein length N : " << N << "\n"
              << "Convolution M     : " << M << "\n\n";

    // Generate random input
    std::vector<cufftComplex> h_x(N);
    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    for(auto& v : h_x){ v.x = dist(rng); v.y = dist(rng); }

    // Device buffers
    cufftComplex *d_x, *d_A, *d_B, *d_C, *d_X;
    CUDA_CHECK(cudaMalloc(&d_x, sizeof(cufftComplex)*N));
    CUDA_CHECK(cudaMalloc(&d_A, sizeof(cufftComplex)*M));
    CUDA_CHECK(cudaMalloc(&d_B, sizeof(cufftComplex)*M));
    CUDA_CHECK(cudaMalloc(&d_C, sizeof(cufftComplex)*M));
    CUDA_CHECK(cudaMalloc(&d_X, sizeof(cufftComplex)*N));

    // Events for timing
    cudaEvent_t e0,e1,e2,e3,e4,e5;
    cudaEventCreate(&e0); cudaEventCreate(&e1); cudaEventCreate(&e2);
    cudaEventCreate(&e3); cudaEventCreate(&e4); cudaEventCreate(&e5);

    // H2D copy
    cudaEventRecord(e0);
    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(),
                          sizeof(cufftComplex)*N, cudaMemcpyHostToDevice));
    cudaEventRecord(e1);

    // Chirp build & padding
    int threads=256;
    int blocksN=(N+threads-1)/threads;
    int blocksM=(M+threads-1)/threads;

    build_chirp<<<blocksN,threads>>>(d_x, d_A, d_B, N, pi_over_N);
    pad_B     <<<blocksM,threads>>>(d_B, N, M);
    cudaEventRecord(e2);

    // Convolution via cuFFT
    cufftHandle plan; CUFFT_CHECK(cufftPlan1d(&plan, M, CUFFT_C2C, 1));

    CUFFT_CHECK(cufftExecC2C(plan, d_A, d_A, CUFFT_FORWARD));
    CUFFT_CHECK(cufftExecC2C(plan, d_B, d_B, CUFFT_FORWARD));
    pointwise_mul<<<blocksM,threads>>>(d_A, d_B, d_C, M);
    CUFFT_CHECK(cufftExecC2C(plan, d_C, d_C, CUFFT_INVERSE));

    // Final chirp & scale
    final_chirp<<<blocksN,threads>>>(d_C, d_X, N, M, pi_over_N);
    cudaEventRecord(e3);

    // D2H copy
    std::vector<cufftComplex> h_X(N);
    CUDA_CHECK(cudaMemcpy(h_X.data(), d_X,
                          sizeof(cufftComplex)*N, cudaMemcpyDeviceToHost));
    cudaEventRecord(e4); cudaEventSynchronize(e4);

    // Timings
    float tH2D, tKernel, tD2H;
    cudaEventElapsedTime(&tH2D, e0, e1);
    cudaEventElapsedTime(&tKernel, e1, e3);
    cudaEventElapsedTime(&tD2H, e3, e4);

    std::cout << "Host → Device copy      : " << tH2D    << " ms\n";
    std::cout << "GPU kernels + FFTs      : " << tKernel << " ms\n";
    std::cout << "Device → Host copy      : " << tD2H    << " ms\n";
    std::cout << "Total GPU time          : " << tH2D + tKernel + tD2H
              << " ms\n";

    // Cleanup
    cufftDestroy(plan);
    cudaFree(d_x); cudaFree(d_A); cudaFree(d_B); cudaFree(d_C); cudaFree(d_X);
    return 0;
}