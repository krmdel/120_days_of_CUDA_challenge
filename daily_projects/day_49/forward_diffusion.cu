#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <iostream>
#include <chrono>
#include <random>
#include <vector>
#include <cmath>

#define cudaCheck(e)  do{ cudaError_t _e=(e);                     \
    if(_e!=cudaSuccess){                                          \
        std::cerr<<"CUDA "<<cudaGetErrorString(_e)                \
                 <<" @ "<<__FILE__<<":"<<__LINE__<<"\n";          \
        std::exit(EXIT_FAILURE);} }while(0)

// Parameters
constexpr int H = 1024, W = 1024, C = 3;
constexpr int PIXELS   = H*W*C;
constexpr int T_STEPS  = 1000;
constexpr int T_SAMPLE = 250;

// β‑schedule  → cumulative ᾱ
std::vector<float> make_alpha_cum()
{
    std::vector<float> a(T_STEPS);
    float beta0 = 1e-4f, beta1 = 2e-2f, cum = 1.f;
    for(int t=0;t<T_STEPS;++t){
        float beta = beta0 + (beta1-beta0)*t/(T_STEPS-1);
        cum *= (1.f - beta);
        a[t] = cum;
    }
    return a;
}

// CPU functions
void fd_cpu(const float* x0,const float* eps,float* out,
            float sa,float soa)
{
    for(int i=0;i<PIXELS;++i)
        out[i] = sa*x0[i] + soa*eps[i];
}

// GPU kernels

// Generating random noise and apply forward diffusion
template<int TPB>
__global__ void fd_gpu_rand(const float* __restrict__ x0,
                            float* __restrict__ noise,
                            float* __restrict__ out,
                            float sa, float soa,
                            unsigned long long seed)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx >= PIXELS) return;

    curandStatePhilox4_32_10_t state;
    curand_init(seed, idx, 0, &state);          // one RNG state per element

    float eps = curand_normal(&state);          // N(0,1)
    noise[idx] = eps;                           // keep for host‑side check
    out  [idx] = sa * x0[idx] + soa * eps;
}

int main()
{
    const size_t BYTES = PIXELS * sizeof(float);

    float *h_x0, *h_out_cpu, *h_eps, *h_out_gpu;
    cudaCheck(cudaHostAlloc(&h_x0,      BYTES, cudaHostAllocDefault));
    cudaCheck(cudaHostAlloc(&h_out_cpu, BYTES, cudaHostAllocDefault));
    cudaCheck(cudaHostAlloc(&h_eps,     BYTES, cudaHostAllocDefault));
    cudaCheck(cudaHostAlloc(&h_out_gpu, BYTES, cudaHostAllocDefault));

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> u(-1.f,1.f);
    for(int i=0;i<PIXELS;++i) h_x0[i]=u(gen);

    // diffusion constants
    auto ac  = make_alpha_cum();
    float sa   = std::sqrt(ac[T_SAMPLE]);
    float soa  = std::sqrt(1.f - ac[T_SAMPLE]);

    // Memory allocation on device
    float *d_x0, *d_out, *d_eps;
    cudaCheck(cudaMalloc(&d_x0,  BYTES));
    cudaCheck(cudaMalloc(&d_out, BYTES));
    cudaCheck(cudaMalloc(&d_eps, BYTES));     // to retrieve ε for validation

    // CUDA streams and events for timing
    cudaStream_t s_copy, s_comp;
    cudaCheck(cudaStreamCreate(&s_copy));
    cudaCheck(cudaStreamCreate(&s_comp));

    cudaEvent_t ev_start, ev_h2d, ev_kernel, ev_before_d2h, ev_d2h, ev_end;
    cudaCheck(cudaEventCreate(&ev_start));    cudaCheck(cudaEventCreate(&ev_h2d));
    cudaCheck(cudaEventCreate(&ev_kernel));   cudaCheck(cudaEventCreate(&ev_before_d2h));
    cudaCheck(cudaEventCreate(&ev_d2h));      cudaCheck(cudaEventCreate(&ev_end));

    // CPU baseline
    auto cpu_t0 = std::chrono::high_resolution_clock::now();
    auto cpu_t1 = std::chrono::high_resolution_clock::now();
    double cpu_ms_placeholder =
        std::chrono::duration<double,std::milli>(cpu_t1-cpu_t0).count();
    
    cudaEventRecord(ev_start);

    // Host to device copy (x0)
    cudaMemcpyAsync(d_x0, h_x0, BYTES, cudaMemcpyHostToDevice, s_copy);
    cudaEventRecord(ev_h2d, s_copy);

    // Kernel launch
    constexpr int TPB = 256;
    int blocks = (PIXELS + TPB - 1) / TPB;
    unsigned long long seed = 123ULL;
    fd_gpu_rand<TPB><<<blocks,TPB,0,s_comp>>>(d_x0,d_eps,d_out,sa,soa,seed);
    cudaCheck(cudaGetLastError());
    cudaEventRecord(ev_kernel, s_comp);

    // Device to host copy (eps, out)
    cudaEventRecord(ev_before_d2h, s_copy);
    cudaMemcpyAsync(h_eps,     d_eps, BYTES, cudaMemcpyDeviceToHost, s_copy);
    cudaMemcpyAsync(h_out_gpu, d_out, BYTES, cudaMemcpyDeviceToHost, s_copy);
    cudaEventRecord(ev_d2h, s_copy);

    cudaEventRecord(ev_end);
    cudaCheck(cudaEventSynchronize(ev_end));

    // CPU computation (for validation)
    auto cpu2_t0 = std::chrono::high_resolution_clock::now();
    fd_cpu(h_x0, h_eps, h_out_cpu, sa, soa);
    auto cpu2_t1 = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double,std::milli>(cpu2_t1-cpu2_t0).count();

    // Timing results
    float ms_h2d, ms_kernel, ms_d2h, ms_total;
    cudaEventElapsedTime(&ms_h2d,   ev_start, ev_h2d);
    cudaEventElapsedTime(&ms_kernel,ev_h2d,   ev_kernel);   // overlap view
    cudaEventElapsedTime(&ms_d2h,  ev_before_d2h, ev_d2h);
    cudaEventElapsedTime(&ms_total, ev_start, ev_end);

    // Display results
    std::cout<<"CPU inference time (ms) : "<<cpu_ms <<" ms\n";
    std::cout<<"GPU timings (ms):\n"
    <<"  Host-to-Device copy time:      : "<<ms_h2d<<"\n"
    <<"  Kernel execution time:         : "<<ms_kernel<<"\n"
    <<"  Device-to-Host copy time:      : "<<ms_d2h<<"\n"
    <<"  Total GPU time:                : "<<ms_total<<"\n";
   
    // Cleanup
    cudaFree(d_x0); cudaFree(d_out); cudaFree(d_eps);
    cudaFreeHost(h_x0); cudaFreeHost(h_out_cpu);
    cudaFreeHost(h_eps); cudaFreeHost(h_out_gpu);
    cudaStreamDestroy(s_copy); cudaStreamDestroy(s_comp);
    cudaEventDestroy(ev_start); cudaEventDestroy(ev_h2d); cudaEventDestroy(ev_kernel);
    cudaEventDestroy(ev_before_d2h); cudaEventDestroy(ev_d2h); cudaEventDestroy(ev_end);
}
