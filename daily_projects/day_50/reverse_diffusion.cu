#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <iostream>
#include <chrono>
#include <vector>
#include <random>
#include <cmath>

#define cudaCheck(e) \
  do{ cudaError_t _e=(e); if(_e!=cudaSuccess){                      \
        std::cerr<<"CUDA "<<cudaGetErrorString(_e)                  \
                 <<" @ "<<__FILE__<<":"<<__LINE__<<"\n"; std::exit(EXIT_FAILURE);} }while(0)

// Parameters
constexpr int  H = 1024, W = 1024, C = 3;
constexpr int  PIXELS   = H*W*C;
constexpr int  T_STEPS  = 1000;
constexpr int  T_SAMPLE = 250;        // which timestep we reverse
constexpr float PRED_NOISE_STD = 0.1f;// σ_pred in ε̂ = ε + σ_pred·ζ

// β‑schedule helpers
struct Sched {
    std::vector<float> beta, alpha, alpha_cum;
    float beta_t, alpha_t, alpha_cum_t, alpha_cum_prev,
          sqrt_alpha_cum_t, sqrt_one_minus_acum_t,
          inv_sqrt_alpha_t, coeff_eps, sigma_t;
};
Sched make_schedule() {
    Sched s;
    s.beta.resize(T_STEPS);
    s.alpha.resize(T_STEPS);
    s.alpha_cum.resize(T_STEPS);

    float beta0=1e-4f, beta1=2e-2f, cum=1.f;
    for(int t=0;t<T_STEPS;++t){
        float beta = beta0 + (beta1-beta0)*t/(T_STEPS-1);
        s.beta[t]=beta;
        s.alpha[t]=1.f-beta;
        cum*=s.alpha[t];
        s.alpha_cum[t]=cum;
    }
    int t=T_SAMPLE;
    s.beta_t  = s.beta[t];
    s.alpha_t = s.alpha[t];
    s.alpha_cum_t   = s.alpha_cum[t];
    s.alpha_cum_prev= (t==0? 1.f : s.alpha_cum[t-1]);

    s.sqrt_alpha_cum_t        = std::sqrt(s.alpha_cum_t);
    s.sqrt_one_minus_acum_t   = std::sqrt(1.f - s.alpha_cum_t);
    s.inv_sqrt_alpha_t        = 1.f / std::sqrt(s.alpha_t);
    s.coeff_eps               = s.beta_t / s.sqrt_one_minus_acum_t;

    // β‑tilde   (variance for reverse noise)
    float beta_tilde = (1.f - s.alpha_cum_prev)/(1.f - s.alpha_cum_t) * s.beta_t;
    s.sigma_t = std::sqrt(beta_tilde);
    return s;
}

// CPU function

// Reverse diffusion step
void reverse_cpu(const float* x0,const float* xt,
                 float* eps_pred,float* x_prev,float* loss,
                 const Sched& sc)
{
    std::mt19937 rng(123); std::normal_distribution<float> g(0.f,1.f);
    for(int i=0;i<PIXELS;++i){
        float eps = (xt[i]-sc.sqrt_alpha_cum_t*x0[i]) / sc.sqrt_one_minus_acum_t;
        float z   = g(rng);                   // ζ
        eps_pred[i] = eps + PRED_NOISE_STD*z; // ε̂
        float z2  = g(rng);                   // ζ′
        x_prev[i] = sc.inv_sqrt_alpha_t * (xt[i] - sc.coeff_eps*eps_pred[i])
                    + sc.sigma_t * z2;
        loss[i]   = (eps_pred[i]-eps)*(eps_pred[i]-eps);
    }
}

// GPU kernel

// Reverse diffusion step
template<int TPB>
__global__ void reverse_gpu(const float* __restrict__ x0,
                            const float* __restrict__ xt,
                            float* __restrict__ eps_pred,
                            float* __restrict__ x_prev,
                            float* __restrict__ loss,
                            float sqrt_alpha_cum_t,
                            float sqrt_one_minus_acum_t,
                            float inv_sqrt_alpha_t,
                            float coeff_eps,
                            float sigma_t,
                            unsigned long long seed)
{
    int i = blockIdx.x*TPB + threadIdx.x;
    if(i>=PIXELS) return;

    // RNG state – use Philox on element index
    curandStatePhilox4_32_10_t st;
    curand_init(seed, i, 0, &st);
    float z  = curand_normal(&st);
    float z2 = curand_normal(&st);

    float eps = (xt[i] - sqrt_alpha_cum_t * x0[i]) / sqrt_one_minus_acum_t;
    float eps_hat = eps + PRED_NOISE_STD * z;

    eps_pred[i] = eps_hat;
    x_prev [i]  = inv_sqrt_alpha_t * (xt[i] - coeff_eps * eps_hat)
                  + sigma_t * z2;
    float diff  = eps_hat - eps;
    loss   [i]  = diff*diff;
}

int main()
{
    const size_t BYTES = PIXELS*sizeof(float);

    // Host buffers
    float *h_x0,*h_xt,*h_eps_pred,*h_xprev,*h_loss;
    cudaCheck(cudaHostAlloc(&h_x0,       BYTES, cudaHostAllocDefault));
    cudaCheck(cudaHostAlloc(&h_xt,       BYTES, cudaHostAllocDefault));
    cudaCheck(cudaHostAlloc(&h_eps_pred, BYTES, cudaHostAllocDefault));
    cudaCheck(cudaHostAlloc(&h_xprev,    BYTES, cudaHostAllocDefault));
    cudaCheck(cudaHostAlloc(&h_loss,     BYTES, cudaHostAllocDefault));

    // Make synthetic x₀, ε, x_t
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> uni(-1.f,1.f);
    std::normal_distribution<float> gauss(0.f,1.f);
    for(int i=0;i<PIXELS;++i) h_x0[i]=uni(gen);

    Sched sc = make_schedule();

    for(int i=0;i<PIXELS;++i){
        float eps = gauss(gen);
        h_xt[i] = sc.sqrt_alpha_cum_t * h_x0[i] +
                  sc.sqrt_one_minus_acum_t * eps;
    }

    // Device buffers
    float *d_x0,*d_xt,*d_eps_pred,*d_xprev,*d_loss;
    cudaCheck(cudaMalloc(&d_x0, BYTES));  cudaCheck(cudaMalloc(&d_xt, BYTES));
    cudaCheck(cudaMalloc(&d_eps_pred, BYTES));
    cudaCheck(cudaMalloc(&d_xprev,    BYTES));
    cudaCheck(cudaMalloc(&d_loss,     BYTES));

    // Streams & events
    cudaStream_t s_cpy,s_cmp; cudaCheck(cudaStreamCreate(&s_cpy));
    cudaCheck(cudaStreamCreate(&s_cmp));

    cudaEvent_t ev0,ev_h2d,ev_k,ev_bef_d2h,ev_d2h,ev_end;
    cudaCheck(cudaEventCreate(&ev0));     cudaCheck(cudaEventCreate(&ev_h2d));
    cudaCheck(cudaEventCreate(&ev_k));    cudaCheck(cudaEventCreate(&ev_bef_d2h));
    cudaCheck(cudaEventCreate(&ev_d2h));  cudaCheck(cudaEventCreate(&ev_end));

    // Copy to device and run kernel
    cudaEventRecord(ev0);
    cudaMemcpyAsync(d_x0,h_x0,BYTES,cudaMemcpyHostToDevice,s_cpy);
    cudaMemcpyAsync(d_xt,h_xt,BYTES,cudaMemcpyHostToDevice,s_cpy);
    cudaEventRecord(ev_h2d,s_cpy);

    constexpr int TPB=256;
    int blocks=(PIXELS+TPB-1)/TPB;
    unsigned long long seed=1234ULL;
    reverse_gpu<TPB><<<blocks,TPB,0,s_cmp>>>(
        d_x0,d_xt,d_eps_pred,d_xprev,d_loss,
        sc.sqrt_alpha_cum_t, sc.sqrt_one_minus_acum_t,
        sc.inv_sqrt_alpha_t, sc.coeff_eps, sc.sigma_t, seed);
    cudaCheck(cudaGetLastError());
    cudaEventRecord(ev_k,s_cmp);

    cudaEventRecord(ev_bef_d2h,s_cpy);
    cudaMemcpyAsync(h_eps_pred,d_eps_pred,BYTES,cudaMemcpyDeviceToHost,s_cpy);
    cudaMemcpyAsync(h_xprev,   d_xprev,   BYTES,cudaMemcpyDeviceToHost,s_cpy);
    cudaMemcpyAsync(h_loss,    d_loss,    BYTES,cudaMemcpyDeviceToHost,s_cpy);
    cudaEventRecord(ev_d2h,s_cpy);

    cudaEventRecord(ev_end);
    cudaCheck(cudaEventSynchronize(ev_end));

    // CPU baseline
    auto c0=std::chrono::high_resolution_clock::now();
    reverse_cpu(h_x0,h_xt,h_eps_pred/*reuse buffer for identical ε̂*/,
                h_xprev/*reuse*/,h_loss/*reuse*/,sc);
    auto c1=std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double,std::milli>(c1-c0).count();

    // Timings
    float ms_h2d,ms_k,ms_d2h,ms_tot;
    cudaEventElapsedTime(&ms_h2d,ev0,       ev_h2d);
    cudaEventElapsedTime(&ms_k,  ev_h2d,    ev_k);
    cudaEventElapsedTime(&ms_d2h,ev_bef_d2h,ev_d2h);
    cudaEventElapsedTime(&ms_tot,ev0,       ev_end);

    std::cout<<"CPU inference time (ms) : "<<cpu_ms <<" ms\n";
    std::cout<<"GPU timings (ms):\n"
    <<"  Host-to-Device copy time:      : "<<ms_h2d<<"\n"
    <<"  Kernel execution time:         : "<<ms_k<<"\n"
    <<"  Device-to-Host copy time:      : "<<ms_d2h<<"\n"
    <<"  Total GPU time:                : "<<ms_tot<<"\n";

    // Cleanup
    cudaFree(d_x0); cudaFree(d_xt); cudaFree(d_eps_pred);
    cudaFree(d_xprev); cudaFree(d_loss);
    cudaFreeHost(h_x0); cudaFreeHost(h_xt); cudaFreeHost(h_eps_pred);
    cudaFreeHost(h_xprev); cudaFreeHost(h_loss);
    cudaStreamDestroy(s_cpy); cudaStreamDestroy(s_cmp);
    cudaEventDestroy(ev0); cudaEventDestroy(ev_h2d); cudaEventDestroy(ev_k);
    cudaEventDestroy(ev_bef_d2h); cudaEventDestroy(ev_d2h); cudaEventDestroy(ev_end);
}
