#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cmath>

#define cudaCheck(e)  do{ cudaError_t _e=(e); if(_e!=cudaSuccess){          \
     std::cerr<<"CUDA "<<cudaGetErrorString(_e)<<" @ "<<__FILE__<<":"<<__LINE__<<"\n"; \
     std::exit(EXIT_FAILURE);} }while(0)

// Parameters
constexpr int H=64, W=64;
constexpr int CIN=4, CM=8, COUT=4;
constexpr int K=3;
constexpr int PIX_IN =H*W*CIN;
constexpr int PIX_MID=H*W*CM;
constexpr int PIX_OUT=H*W*COUT;
constexpr int TILE=16;

// Diffusion hyper parameters
constexpr int T_FULL = 1000;                 // reference chain length
constexpr int T_FAST = 50;                   // optimised schedule length

// Helper: linear β schedule
std::vector<float> make_beta(int T){
    std::vector<float> b(T);
    float b0=1e-4f,b1=2e-2f;
    for(int t=0;t<T;++t) b[t]=b0+(b1-b0)*t/(T-1);
    return b;
}
struct Schedule{
    std::vector<int>    idx;       // timesteps (descending)
    std::vector<float>  alpha,alpha_cum,beta;
    std::vector<float>  coeff,sigma,inv_sqrt_alpha;
};
Schedule build_schedule(int T,const std::vector<int>& steps){
    auto beta = make_beta(T);
    std::vector<float> alpha(T),alpha_cum(T);
    float cum=1.f;
    for(int t=0;t<T;++t){ alpha[t]=1.f-beta[t]; cum*=alpha[t]; alpha_cum[t]=cum; }

    Schedule S; S.idx=steps; S.beta.resize(T); S.alpha=alpha; S.alpha_cum=alpha_cum;
    S.beta=beta;
    int n=steps.size();
    S.coeff.resize(n); S.sigma.resize(n); S.inv_sqrt_alpha.resize(n);
    for(int k=0;k<n;++k){
        int t=steps[k];
        float beta_t=beta[t];
        float alpha_t=alpha[t];
        float ac_t=alpha_cum[t];
        float ac_prev = (t==0?1.f:alpha_cum[t-1]);
        float sqrt_one_minus = std::sqrt(1.f-ac_t);
        S.coeff[k] = beta_t / sqrt_one_minus;
        S.inv_sqrt_alpha[k] = 1.f/std::sqrt(alpha_t);
        float beta_tilde = (1.f-ac_prev)/(1.f-ac_t)*beta_t;
        S.sigma[k] = std::sqrt(beta_tilde);
    }
    return S;
}

// Square‑spaced indices ↓ (T‑1 ... 0)
std::vector<int> sqrt_schedule(int T,int N){
    std::vector<int> s(N);
    for(int i=0;i<N;++i){
        float f = static_cast<float>(i)/ (N-1);
        int t = static_cast<int>(std::round((1.f-f*f)*(T-1)));
        s[i]=t;
    }
    return s;
}

// Tiny U‑Net kernels
__host__ __device__ inline int off(int c,int y,int x,int C){
    return (c*H + y)*W + x;
}

template<int IC,int OC>
__device__ inline float Wget(const float* w,int oc,int ic,int ky,int kx){
    return w[(((oc*IC+ic)*K)+ky)*K+kx]; }

template<int IC,int OC>
__global__ void conv3(const float* __restrict__ in, const float* __restrict__ w,
                      float* __restrict__ out){
    __shared__ float tile[TILE+2][TILE+2][IC];
    int oc=blockIdx.z, by=blockIdx.y*TILE, bx=blockIdx.x*TILE;
    int ty=threadIdx.y, tx=threadIdx.x;
    int iy=by+ty-1, ix=bx+tx-1;
    for(int ic=0;ic<IC;++ic){
        float v=0.f; if(iy>=0&&iy<H&&ix>=0&&ix<W) v=in[off(ic,iy,ix,IC)];
        tile[ty][tx][ic]=v;
    } __syncthreads();
    if(ty<TILE&&tx<TILE){
        int y=by+ty,x=bx+tx; if(y>=H||x>=W) return;
        float s=0.f;
        #pragma unroll
        for(int ic=0;ic<IC;++ic)
          #pragma unroll
          for(int ky=0;ky<3;++ky)
            #pragma unroll
            for(int kx=0;kx<3;++kx)
                s+=tile[ty+ky][tx+kx][ic]*Wget<IC,OC>(w,oc,ic,ky,kx);
        out[off(oc,y,x,OC)]=s;
    }
}
__global__ void relu(float* t,int N){ int i=blockIdx.x*blockDim.x+threadIdx.x;
                                      if(i<N) t[i]=fmaxf(t[i],0.f);}
__global__ void add_res(const float* x,float* y,int N){int i=blockIdx.x*blockDim.x+threadIdx.x;
                                                       if(i<N) y[i]+=x[i];}

// CPU Denoiser forward
void conv3_cpu(const float* in,const float* w,float* out,int IC,int OC){
    for(int oc=0;oc<OC;++oc)
      for(int y=0;y<H;++y)
        for(int x=0;x<W;++x){
            float s=0.f;
            for(int ic=0;ic<IC;++ic)
              for(int ky=-1;ky<=1;++ky)
                for(int kx=-1;kx<=1;++kx){
                    int yy=y+ky,xx=x+kx;
                    if(yy<0||yy>=H||xx<0||xx>=W) continue;
                    s+=in[off(ic,yy,xx,IC)]*
                       w[(((oc*IC+ic)*K)+(ky+1))*K+(kx+1)];
                }
            out[off(oc,y,x,OC)]=s;
        }
}
void denoiser_cpu(const float* w1,const float* w2,
                  const float* x,float* tmp,float* y){
    conv3_cpu(x,w1,tmp,CIN,CM);
    for(int i=0;i<PIX_MID;++i) tmp[i]=fmaxf(tmp[i],0.f);
    conv3_cpu(tmp,w2,y,CM,COUT);
    for(int i=0;i<PIX_OUT;++i) y[i]+=x[i];           // residual
}

// GPU kernel

// RNG & reverse step
__global__ void rng_init(curandStatePhilox4_32_10_t* st,int n,int seed){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) curand_init(seed,i,0,&st[i]);}
__global__ void reverse_step(float* xt,const float* eps_pred,
                             curandStatePhilox4_32_10_t* st,int n,
                             float inv_sqrt_alpha,float coeff,float sigma){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n){
        float z=curand_normal(&st[i]);
        xt[i]=inv_sqrt_alpha*(xt[i]-coeff*eps_pred[i])+sigma*z;
    }
}

int main(){
    // Host allocations
    std::mt19937 rng(42); std::normal_distribution<float> wd(0.f,0.05f);
    size_t W1=CM*CIN*K*K, W2=COUT*CM*K*K;
    std::vector<float> h_w1(W1),h_w2(W2);
    for(auto& w:h_w1) w=wd(rng); for(auto& w:h_w2) w=wd(rng);

    // Build schedules
    auto fast_idx = sqrt_schedule(T_FULL,T_FAST);     // √‑spaced 50‑step
    Schedule S    = build_schedule(T_FULL,fast_idx);

    // Host latent & buffers
    std::normal_distribution<float> g(0.f,1.f);
    std::vector<float> h_latent(PIX_IN); for(float&v:h_latent) v=g(rng);
    std::vector<float> h_mid(PIX_MID),h_eps(PIX_OUT);

    // Device allocations
    float *d_xt,*d_mid,*d_eps,*d_w1,*d_w2;
    cudaMalloc(&d_xt,PIX_IN*sizeof(float));
    cudaMalloc(&d_mid,PIX_MID*sizeof(float));
    cudaMalloc(&d_eps,PIX_OUT*sizeof(float));
    cudaMalloc(&d_w1,W1*sizeof(float));
    cudaMalloc(&d_w2,W2*sizeof(float));
    cudaMemcpy(d_w1,h_w1.data(),W1*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_w2,h_w2.data(),W2*sizeof(float),cudaMemcpyHostToDevice);

    // RNG states
    curandStatePhilox4_32_10_t* d_states;
    cudaMalloc(&d_states,PIX_IN*sizeof(curandStatePhilox4_32_10_t));
    rng_init<<<(PIX_IN+255)/256,256>>>(d_states,PIX_IN,1234);

    // CUDA stream & events
    cudaStream_t s_cpy,s_cmp; cudaStreamCreate(&s_cpy); cudaStreamCreate(&s_cmp);
    cudaEvent_t ev0,ev_h2d,ev_k,ev_d2h,ev_end;
    cudaEventCreate(&ev0); cudaEventCreate(&ev_h2d);
    cudaEventCreate(&ev_k); cudaEventCreate(&ev_d2h); cudaEventCreate(&ev_end);

    // GPU sampling (50‑step fast schedule)
    cudaEventRecord(ev0);
    cudaMemcpyAsync(d_xt,h_latent.data(),PIX_IN*sizeof(float),cudaMemcpyHostToDevice,s_cpy);
    cudaEventRecord(ev_h2d,s_cpy);

    dim3 g1((W+TILE-1)/TILE,(H+TILE-1)/TILE,CM);
    dim3 g2((W+TILE-1)/TILE,(H+TILE-1)/TILE,COUT);
    dim3 blk(TILE+2,TILE+2);

    for(int k=0;k<T_FAST;++k){
        // eps_pred = U‑Net(x_t)
        conv3<CIN,CM><<<g1,blk,0,s_cmp>>>(d_xt,d_w1,d_mid);
        relu<<<(PIX_MID+255)/256,256,0,s_cmp>>>(d_mid,PIX_MID);
        conv3<CM,COUT><<<g2,blk,0,s_cmp>>>(d_mid,d_w2,d_eps);
        add_res<<<(PIX_OUT+255)/256,256,0,s_cmp>>>(d_xt,d_eps,PIX_OUT);

        reverse_step<<<(PIX_IN+255)/256,256,0,s_cmp>>>
            (d_xt,d_eps,d_states,PIX_IN,
             S.inv_sqrt_alpha[k],S.coeff[k],S.sigma[k]);
    }
    cudaEventRecord(ev_k,s_cmp);

    cudaMemcpyAsync(h_latent.data(),d_xt,PIX_IN*sizeof(float),cudaMemcpyDeviceToHost,s_cpy);
    cudaEventRecord(ev_d2h,s_cpy);
    cudaEventRecord(ev_end); cudaEventSynchronize(ev_end);

    float ms_h2d,ms_k,ms_d2h,ms_tot;
    cudaEventElapsedTime(&ms_h2d,ev0,ev_h2d);
    cudaEventElapsedTime(&ms_k,ev_h2d,ev_k);
    cudaEventElapsedTime(&ms_d2h,ev_k,ev_d2h);
    cudaEventElapsedTime(&ms_tot,ev0,ev_end);

    // CPU sampling (50‑step fast schedule)
    std::vector<float> xcpu(PIX_IN); for(float&v:xcpu) v=g(rng);
    auto cpu0=std::chrono::high_resolution_clock::now();
    for(int k=0;k<T_FAST;++k){
        // eps_pred
        denoiser_cpu(h_w1.data(),h_w2.data(),xcpu.data(),h_mid.data(),h_eps.data());
        // step params
        float inv_a = S.inv_sqrt_alpha[k], coeff=S.coeff[k], sig=S.sigma[k];
        for(int i=0;i<PIX_IN;++i){
            float z=g(rng);
            xcpu[i]=inv_a*(xcpu[i]-coeff*h_eps[i]) + sig*z;
        }
    }
    auto cpu1=std::chrono::high_resolution_clock::now();
    double cpu_ms=std::chrono::duration<double,std::milli>(cpu1-cpu0).count();

    // Timings
    std::cout<<"CPU inference time           : "<<cpu_ms <<" ms\n";
    std::cout<<"GPU timings (ms):\n"
    <<"  Host-to-Device copy time:      : "<<ms_h2d<<"\n"
    <<"  Kernel execution time:         : "<<ms_k<<"\n"
    <<"  Device-to-Host copy time:      : "<<ms_d2h<<"\n"
    <<"  Total GPU time:                : "<<ms_tot<<"\n";


    // Cleanup
    cudaFree(d_xt); cudaFree(d_mid); cudaFree(d_eps);
    cudaFree(d_w1); cudaFree(d_w2);  cudaFree(d_states);
    cudaEventDestroy(ev0); cudaEventDestroy(ev_h2d); cudaEventDestroy(ev_k);
    cudaEventDestroy(ev_d2h); cudaEventDestroy(ev_end);
    cudaStreamDestroy(s_cpy); cudaStreamDestroy(s_cmp);
}
