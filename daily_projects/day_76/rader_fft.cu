#include <cuda_runtime.h>
#include <cufft.h>
#include <vector>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <iostream>

#ifndef N
#define N 257                    // default prime length (must be prime)
#endif
static_assert(N > 2,  "N must be > 2");
static_assert(N % 2 != 0, "N must be odd (prime)");

// CUDA helpers
#define CUDA_CHECK(x) \
    do { cudaError_t rc=(x); if(rc!=cudaSuccess){                              \
        std::fprintf(stderr,"CUDA %s @ %s:%d\n",cudaGetErrorString(rc),        \
                     __FILE__,__LINE__); std::exit(EXIT_FAILURE);} } while(0)
#define CUFFT_CHECK(x) \
    do { cufftResult rc=(x); if(rc!=CUFFT_SUCCESS){                            \
        std::fprintf(stderr,"cuFFT error %d @ %s:%d\n",rc,__FILE__,__LINE__);  \
        std::exit(EXIT_FAILURE);} } while(0)

__device__ __forceinline__ float2 operator+(float2 a,float2 b){
    return make_float2(a.x+b.x , a.y+b.y);
}
__device__ __forceinline__ float2 operator*(float2 a,float2 b){
    return make_float2(a.x*b.x - a.y*b.y , a.x*b.y + a.y*b.x);
}

// Pointwise multiply kernel
__global__ void mul_complex(const cufftComplex* A,
                            const cufftComplex* B,
                            cufftComplex*       C,
                            int                 n)
{
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n){
        cufftComplex a=A[i], b=B[i];
        C[i].x = a.x*b.x - a.y*b.y;
        C[i].y = a.x*b.y + a.y*b.x;
    }
}

// Primitive root (host)
/*  Finds the smallest generator of (Z_N)*.  N is prime and small
( < 10^6 here ), so simple trial works.                          */
int primitive_root(int p)
{
    std::vector<int> fact;
    int phi=p-1, n=phi;
    for(int i=2; i*i<=n; ++i)
        if(n%i==0){
            fact.push_back(i);
            while(n%i==0) n/=i;
        }
    if(n>1) fact.push_back(n);
    for(int g=2; g<p; ++g){
        bool ok=true;
        for(int f:fact)
            if(std::pow(g,phi/f) - std::floor(std::pow(g,phi/f)+0.5) == 0){ // integer pow
                int t=1;
                for(int i=0;i<phi/f;++i) t=(t*g)%p;
                if(t==1){ ok=false; break;}
            }
        if(ok) return g;
    }
    return -1;
}

// Host utility: next pow-2
int next_pow2(int n){ int p=1; while(p<n) p<<=1; return p; }

int main()
{
    constexpr int PRIME = N;
    const int M  = PRIME - 1;                 // convolution length
    const int L  = next_pow2(M);              // cuFFT size
    const float TWO_PI = 2.0f * M_PI;

    std::cout<<"Prime length N  : "<<PRIME<<"\n";
    std::cout<<"Convolution len : "<<M<<"\n";
    std::cout<<"FFT size (pow2) : "<<L<<"\n\n";

    // Generate random complex input x[n]
    std::vector<float2> h_x(PRIME);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.f,1.f);
    for(auto& v: h_x){ v.x=dist(rng); v.y=dist(rng); }

    // Step 1 : choose primitive root g
    int g = 3;                       // 3 is a primitive root for many primes incl. 257
#if N != 257
    g = primitive_root(PRIME);
#endif
    if(g<0){ std::cerr<<"Cannot find primitive root\n"; return 1; }

    // Step 2 : build sequences A (re-ordered data) and B (twiddles)
    std::vector<cufftComplex> A(L), B(L);
    int k=1;
    for(int m=0;m<M;++m){
        A[m].x = h_x[k].x;  A[m].y = h_x[k].y;          // a_m = x[g^m]
        float ang = -TWO_PI * k / PRIME;
        B[m].x = cosf(ang);  B[m].y = sinf(ang);        // b_m = e^{-j2π k/N}
        k = (k * g) % PRIME;
    }
    for(int i=M;i<L;++i){ A[i].x=A[i].y=0; B[i].x=B[i].y=0; }

    // GPU buffers
    cufftComplex *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A,sizeof(cufftComplex)*L));
    CUDA_CHECK(cudaMalloc(&d_B,sizeof(cufftComplex)*L));
    CUDA_CHECK(cudaMalloc(&d_C,sizeof(cufftComplex)*L));

    // Events for timing
    cudaEvent_t e0,e1,e2,e3,e4; cudaEventCreate(&e0);cudaEventCreate(&e1);
    cudaEventCreate(&e2);cudaEventCreate(&e3);cudaEventCreate(&e4);

    cudaEventRecord(e0);
    CUDA_CHECK(cudaMemcpy(d_A,A.data(),sizeof(cufftComplex)*L,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B,B.data(),sizeof(cufftComplex)*L,cudaMemcpyHostToDevice));
    cudaEventRecord(e1);

    // cuFFT plans
    cufftHandle planF, planI;
    CUFFT_CHECK(cufftPlan1d(&planF,L,CUFFT_C2C,1));
    CUFFT_CHECK(cufftPlan1d(&planI,L,CUFFT_C2C,1));

    // Forward FFT
    CUFFT_CHECK(cufftExecC2C(planF,d_A,d_A,CUFFT_FORWARD));
    CUFFT_CHECK(cufftExecC2C(planF,d_B,d_B,CUFFT_FORWARD));

    // Pointwise multiply
    int threads=256, blocks=(L+threads-1)/threads;
    mul_complex<<<blocks,threads>>>(d_A,d_B,d_C,L);

    // Inverse FFT
    CUFFT_CHECK(cufftExecC2C(planI,d_C,d_C,CUFFT_INVERSE));
    cudaEventRecord(e2);

    // Copy convolution result back
    std::vector<cufftComplex> h_C(L);
    CUDA_CHECK(cudaMemcpy(h_C.data(),d_C,sizeof(cufftComplex)*L,cudaMemcpyDeviceToHost));
    cudaEventRecord(e3); cudaEventSynchronize(e3);

    CUDA_CHECK(cudaFree(d_A)); CUDA_CHECK(cudaFree(d_B)); CUDA_CHECK(cudaFree(d_C));
    cufftDestroy(planF); cufftDestroy(planI);

    // Build final DFT result X[k]
    std::vector<float2> X(PRIME);
    float2 X0{0,0};
    for(auto &v: h_x){ X0.x+=v.x; X0.y+=v.y; }
    X[0]=X0;

    // Normalise IFFT output (cuFFT does not scale)
    for(int i=0;i<M;++i){ h_C[i].x/=L; h_C[i].y/=L; }

    int idx=1;
    for(int m=0;m<M;++m){
        X[idx].x = X0.x + h_C[m].x;
        X[idx].y = X0.y + h_C[m].y;
        idx = (idx * g) % PRIME;
    }

    cudaEventRecord(e4); cudaEventSynchronize(e4);

    // Timings
    float tH2D,tFFT,tMul,tD2H;
    cudaEventElapsedTime(&tH2D,e0,e1);
    cudaEventElapsedTime(&tFFT,e1,e2);              // includes point-mult
    cudaEventElapsedTime(&tMul,e2,e3);              // IFFT+copy
    cudaEventElapsedTime(&tD2H,e3,e4);

    std::cout<<"H2D copy        : "<<tH2D<<" ms\n";
    std::cout<<"FFT+mul+IFFT    : "<<tFFT<<" ms\n";
    std::cout<<"D2H copy+build  : "<<tMul+tD2H<<" ms\n";
    std::cout<<"Total GPU time  : "<<tH2D+tFFT+tMul+tD2H<<" ms\n";

    return 0;
}