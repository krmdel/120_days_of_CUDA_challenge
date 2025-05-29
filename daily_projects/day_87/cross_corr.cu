#include <cuda_runtime.h>
#include <cufft.h>

#include <vector>
#include <random>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <cstdlib>
#include <algorithm>          // std::max_element

// Helpers
#define THREADS 256
#define CUDA_CHECK(x)  do{ cudaError_t rc=(x); if(rc!=cudaSuccess){            \
    std::cerr<<"CUDA "<<cudaGetErrorString(rc)<<" @"<<__FILE__<<":"<<__LINE__;\
    std::exit(EXIT_FAILURE);} }while(0)
#define CUFFT_CHECK(x) do{ cufftResult rc=(x); if(rc!=CUFFT_SUCCESS){          \
    std::cerr<<"cuFFT "<<rc<<" @"<<__FILE__<<":"<<__LINE__<<"\n";              \
    std::exit(EXIT_FAILURE);} }while(0)

// Naïve O(N²) kernel (one thread per lag)
__global__ void xcorr_naive(const float* __restrict__ x,
                            const float* __restrict__ y,
                            float*       __restrict__ c,
                            int N)
{
    int M   = 2*N - 1;
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx >= M) return;

    int lag = idx - (N - 1);          // negative ⇒ y leads, positive ⇒ y lags
    float acc = 0.f;
    for(int n=0; n<N; ++n){
        int j = n + lag;
        if(j >= 0 && j < N) acc += x[n] * y[j];
    }
    c[idx] = acc;
}

// Complex multiply: X · conj(Y)
__global__ void cmul_conj(const cufftComplex* __restrict__ A,
                          const cufftComplex* __restrict__ B,
                          cufftComplex*       __restrict__ C,
                          int M)
{
    int k = blockIdx.x*blockDim.x + threadIdx.x;
    if(k < M){
        cufftComplex a = A[k], b = B[k];
        C[k].x = a.x*b.x + a.y*b.y;           // real( a * conj(b) )
        C[k].y = a.y*b.x - a.x*b.y;           // imag( … )
    }
}

// Copy first M samples of IFFT result & scale
__global__ void scaleCopyKernel(float* dst,
                                const cufftComplex* src,
                                float s, int M)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i < M) dst[i] = src[i].x * s;          // take real part
}

// Utility
static int next_pow2(int v){ int p=1; while(p<v) p<<=1; return p; }
struct Times{ float h2d, ker, d2h; float total()const{return h2d+ker+d2h;} };

int main(int argc,char**argv)
{
    const int N = (argc>1)? std::atoi(argv[1]) : 262'144;   // default 2^18
    const int M = 2*N - 1;                                  // corr length
    const int Mp2 = next_pow2(M);                           // FFT size (power-2)

    std::cout<<"Signal length N           : "<<N<<"\n"
             <<"Correlation sequence size : "<<M<<"\n"
             <<"FFT size (next pow-2)      : "<<Mp2<<"\n\n";

    // Host input signals (white noise)
    std::vector<float> h_x(N), h_y(N);
    std::mt19937 rng(42); std::uniform_real_distribution<float> dist(-1.f,1.f);
    for(float& v:h_x) v=dist(rng);
    for(float& v:h_y) v=dist(rng);

    // Device buffers
    float *d_x, *d_y, *d_corr_time, *d_corr_fft;
    CUDA_CHECK(cudaMalloc(&d_x,           sizeof(float)*N));
    CUDA_CHECK(cudaMalloc(&d_y,           sizeof(float)*N));
    CUDA_CHECK(cudaMalloc(&d_corr_time,   sizeof(float)*M));
    CUDA_CHECK(cudaMalloc(&d_corr_fft ,   sizeof(float)*M));

    // Common H2D copy
    cudaEvent_t eH0,eH1; cudaEventCreate(&eH0); cudaEventCreate(&eH1);
    cudaEventRecord(eH0);
    CUDA_CHECK(cudaMemcpy(d_x,h_x.data(),sizeof(float)*N,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y,h_y.data(),sizeof(float)*N,cudaMemcpyHostToDevice));
    cudaEventRecord(eH1); CUDA_CHECK(cudaDeviceSynchronize());
    float h2d_ms; cudaEventElapsedTime(&h2d_ms,eH0,eH1);
    cudaEventDestroy(eH0); cudaEventDestroy(eH1);

    // 1) Time-domain O(N²)
    Times tTime;
    {
        cudaEvent_t k0,k1,c0,c1; cudaEventCreate(&k0); cudaEventCreate(&k1);
        cudaEventCreate(&c0); cudaEventCreate(&c1);

        int blocks = (M + THREADS - 1)/THREADS;
        cudaEventRecord(k0);
        xcorr_naive<<<blocks,THREADS>>>(d_x,d_y,d_corr_time,N);
        cudaEventRecord(k1);

        std::vector<float> h_corr_time(M);
        cudaEventRecord(c0);
        CUDA_CHECK(cudaMemcpy(h_corr_time.data(),d_corr_time,
                              sizeof(float)*M,cudaMemcpyDeviceToHost));
        cudaEventRecord(c1); CUDA_CHECK(cudaDeviceSynchronize());

        cudaEventElapsedTime(&tTime.ker ,k0,k1);
        cudaEventElapsedTime(&tTime.d2h,c0,c1);
        tTime.h2d = h2d_ms;
        
        cudaEventDestroy(k0); cudaEventDestroy(k1);
        cudaEventDestroy(c0); cudaEventDestroy(c1);
    }

    // 2) FFT  O(N log N)
    Times tFFT;
    {
        // Complex scratch arrays (zero-padded)
        cufftComplex *d_X,*d_Y,*d_Z;
        CUDA_CHECK(cudaMalloc(&d_X,sizeof(cufftComplex)*Mp2));
        CUDA_CHECK(cudaMalloc(&d_Y,sizeof(cufftComplex)*Mp2));
        CUDA_CHECK(cudaMalloc(&d_Z,sizeof(cufftComplex)*Mp2));
        CUDA_CHECK(cudaMemset(d_X,0,sizeof(cufftComplex)*Mp2));
        CUDA_CHECK(cudaMemset(d_Y,0,sizeof(cufftComplex)*Mp2));

        // Copy real -> complex (imag=0)
        CUDA_CHECK(cudaMemcpy2D(d_X,sizeof(cufftComplex),
                                d_x,sizeof(float),
                                sizeof(float),N,
                                cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy2D(d_Y,sizeof(cufftComplex),
                                d_y,sizeof(float),
                                sizeof(float),N,
                                cudaMemcpyDeviceToDevice));

        cufftHandle plan; CUFFT_CHECK(cufftPlan1d(&plan,Mp2,CUFFT_C2C,1));

        cudaEvent_t k0,k1,c0,c1; cudaEventCreate(&k0); cudaEventCreate(&k1);
        cudaEventCreate(&c0); cudaEventCreate(&c1);

        cudaEventRecord(k0);
        // Forward FFTs
        CUFFT_CHECK(cufftExecC2C(plan,d_X,d_X,CUFFT_FORWARD));
        CUFFT_CHECK(cufftExecC2C(plan,d_Y,d_Y,CUFFT_FORWARD));
        // Multiply X · conj(Y)
        int blocksMul = (Mp2 + THREADS - 1)/THREADS;
        cmul_conj<<<blocksMul,THREADS>>>(d_X,d_Y,d_Z,Mp2);
        // Inverse FFT
        CUFFT_CHECK(cufftExecC2C(plan,d_Z,d_Z,CUFFT_INVERSE));

        // Scale & copy first M real samples
        float scale = 1.0f / Mp2;
        int blocksSC = (M + THREADS - 1)/THREADS;
        scaleCopyKernel<<<blocksSC,THREADS>>>(d_corr_fft,d_Z,scale,M);
        cudaEventRecord(k1);

        // Copy back
        std::vector<float> h_corr_fft(M);
        cudaEventRecord(c0);
        CUDA_CHECK(cudaMemcpy(h_corr_fft.data(),d_corr_fft,
                              sizeof(float)*M,cudaMemcpyDeviceToHost));
        cudaEventRecord(c1); CUDA_CHECK(cudaDeviceSynchronize());

        cudaEventElapsedTime(&tFFT.ker ,k0,k1);
        cudaEventElapsedTime(&tFFT.d2h,c0,c1);
        tFFT.h2d = h2d_ms;

        cudaEventDestroy(k0); cudaEventDestroy(k1);
        cudaEventDestroy(c0); cudaEventDestroy(c1);
        cufftDestroy(plan);
        cudaFree(d_X); cudaFree(d_Y); cudaFree(d_Z);
    }

    // Timings
    auto line=[&](const char* n,const Times& t){
        std::cout<<std::setw(13)<<std::left<<n
                 <<" H2D "<<std::setw(7)<<t.h2d
                 <<" Kern "<<std::setw(9)<<t.ker
                 <<" D2H "<<std::setw(7)<<t.d2h
                 <<" Total "<<t.total()<<" ms\n";
    };
    std::cout<<std::fixed<<std::setprecision(3);
    std::cout<<"\n               H2D      Kern        D2H     Total\n";
    line("Time-domain", tTime);
    line("FFT        ", tFFT);

    // Cleanup
    cudaFree(d_x); cudaFree(d_y);
    cudaFree(d_corr_time); cudaFree(d_corr_fft);
    return 0;
}