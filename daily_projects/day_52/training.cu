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
constexpr int CIN=4, CINTM=8, COUT=4;
constexpr int K=3;
constexpr int PIX_IN =H*W*CIN;
constexpr int PIX_MID=H*W*CINTM;
constexpr int PIX_OUT=H*W*COUT;
constexpr int TILE=16;
constexpr int BATCH=8;
constexpr float LR=1e-2f;

// Schedule (linear β)
constexpr int T_STEPS=1000;
std::vector<float> make_alpha_cum(){
    std::vector<float> ac(T_STEPS); float b0=1e-4f,b1=2e-2f,c=1.f;
    for(int t=0;t<T_STEPS;++t){ float b=b0+(b1-b0)*t/(T_STEPS-1); c*=1.f-b; ac[t]=c;}
    return ac;
}
struct Sched{
    std::vector<float> ac,beta;
    Sched(){ ac=make_alpha_cum(); beta.resize(T_STEPS);
             for(int t=0;t<T_STEPS;++t) beta[t]=1.f-(t?ac[t]/ac[t-1]:ac[t]);}
} sched;

__host__ __device__ inline int off(int c,int y,int x,int C){return (c*H+y)*W+x;}

// CUDA kernels
template<int IC,int OC>
__device__ inline float get_w(const float* w,int oc,int ic,int ky,int kx){
    return w[(((oc*IC+ic)*K)+ky)*K+kx];
}
// Forward conv
template<int IC,int OC>
__global__ void conv3_fwd(const float* __restrict__ in,
                          const float* __restrict__ w,
                          float*       __restrict__ out){
    __shared__ float tile[TILE+2][TILE+2][IC];
    int oc=blockIdx.z, by=blockIdx.y*TILE, bx=blockIdx.x*TILE;
    int ty=threadIdx.y, tx=threadIdx.x;
    int iy=by+ty-1, ix=bx+tx-1;
    for(int ic=0;ic<IC;++ic){
        float v=0.f;
        if(iy>=0&&iy<H&&ix>=0&&ix<W) v=in[off(ic,iy,ix,IC)];
        tile[ty][tx][ic]=v;
    }
    __syncthreads();
    if(ty<TILE&&tx<TILE){
        int y=by+ty,x=bx+tx; if(y>=H||x>=W) return;
        float s=0.f;
        #pragma unroll
        for(int ic=0;ic<IC;++ic)
         #pragma unroll
         for(int ky=0;ky<3;++ky)
          #pragma unroll
          for(int kx=0;kx<3;++kx)
              s+=tile[ty+ky][tx+kx][ic]*get_w<IC,OC>(w,oc,ic,ky,kx);
        out[off(oc,y,x,OC)]=s;
    }
}
__global__ void relu_fwd(float* t,int N){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<N)t[i]=fmaxf(t[i],0.f);}
__global__ void relu_bwd(const float* in,const float* g_out,float* g_in,int N){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<N) g_in[i]=(in[i]>0.f?g_out[i]:0.f);}
__global__ void add_kernel(const float* x,float* y,int N){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<N) y[i]+=x[i];}
__global__ void zero_kernel(float* t,int N){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<N) t[i]=0.f;}

// Backward conv (data)
template<int IC,int OC>
__global__ void conv3_bwd_data(const float* __restrict__ g_out,
                               const float* __restrict__ w,
                               float*       __restrict__ g_in){
    int ic=blockIdx.z;
    int y =blockIdx.y*blockDim.y+threadIdx.y;
    int x =blockIdx.x*blockDim.x+threadIdx.x;
    if(y>=H||x>=W) return;
    float s=0.f;
    for(int oc=0;oc<OC;++oc)
      for(int ky=0;ky<3;++ky)
        for(int kx=0;kx<3;++kx){
            int yy=y-ky+1,xx=x-kx+1;
            if(yy<0||yy>=H||xx<0||xx>=W) continue;
            s+=g_out[off(oc,yy,xx,OC)]*get_w<IC,OC>(w,oc,ic,ky,kx);
        }
    g_in[off(ic,y,x,IC)]=s;
}

// Backward conv (weights)
template<int IC,int OC>
__global__ void conv3_bwd_w(const float* __restrict__ in,
                            const float* __restrict__ g_out,
                            float*       __restrict__ g_w){
    int oc=blockIdx.x, ic=blockIdx.y;
    int k=threadIdx.x, ky=k/3, kx=k%3;
    float sum=0.f;
    for(int y=0;y<H;++y) for(int x=0;x<W;++x){
        int yy=y+ky-1,xx=x+kx-1;
        if(yy<0||yy>=H||xx<0||xx>=W) continue;
        sum+=in[off(ic,yy,xx,IC)]*g_out[off(oc,y,x,OC)];
    }
    atomicAdd(&g_w[(((oc*IC+ic)*K)+ky)*K+kx],sum);
}

// SGD
__global__ void sgd(float* w,const float* g,float lr,int N){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<N) w[i]-=lr*g[i];}

// Loss + grad
__global__ void loss_mse_grad(const float* pred,const float* tgt,
                              float* grad,float* loss,int N){
    __shared__ float buf[256]; float l=0.f;
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<N){ float d=pred[i]-tgt[i]; grad[i]=2.f*d/N; l=d*d; }
    buf[threadIdx.x]=l; __syncthreads();
    for(int s=128;s; s>>=1){ if(threadIdx.x<s) buf[threadIdx.x]+=buf[threadIdx.x+s]; __syncthreads();}
    if(threadIdx.x==0) atomicAdd(loss,buf[0]);
}

// RNG init & reverse‑step kernels
__global__ void rng_init(curandStatePhilox4_32_10_t* st,int n,int seed){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) curand_init(seed,i,0,&st[i]);}
__global__ void reverse_step_kernel(float* xt,const float* eps_pred,
                                    curandStatePhilox4_32_10_t* st,int n,
                                    float inv_sqrt_alpha,float coeff,float sigma){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n){
        float z=curand_normal(&st[i]);
        xt[i]=inv_sqrt_alpha*(xt[i]-coeff*eps_pred[i])+sigma*z;
    }
}

// Utilities
template<class T> inline void fill_rand(std::vector<T>& v,std::mt19937& r,float a,float b){
    std::uniform_real_distribution<float> d(a,b); for(auto&x:v) x=d(r);}
template<class T> inline void fill_gauss(std::vector<T>& v,std::mt19937& r,float s){
    std::normal_distribution<float> d(0.f,s); for(auto&x:v) x=d(r);}

int main(){
    // Host buffers
    std::mt19937 rng(42);
    std::vector<std::vector<float>> h_x(BATCH,std::vector<float>(PIX_IN));
    std::vector<std::vector<float>> h_eps(BATCH,std::vector<float>(PIX_OUT));
    for(auto& img:h_x)  fill_rand(img,rng,-1.f,1.f);
    for(auto& eps:h_eps) fill_gauss(eps,rng,1.f);

    // Weights & grads
    size_t W1=CINTM*CIN*K*K, W2=COUT*CINTM*K*K;
    std::vector<float> h_w1(W1),h_w2(W2);
    fill_gauss(h_w1,rng,0.05f); fill_gauss(h_w2,rng,0.05f);

    // Memory allocation on device
    float *d_x,*d_mid,*d_y,*d_tgt,*d_grad_y,*d_w1,*d_w2,*d_gw1,*d_gw2;
    cudaMalloc(&d_x,PIX_IN*sizeof(float));
    cudaMalloc(&d_mid,PIX_MID*sizeof(float));
    cudaMalloc(&d_y,PIX_OUT*sizeof(float));
    cudaMalloc(&d_tgt,PIX_OUT*sizeof(float));
    cudaMalloc(&d_grad_y,PIX_OUT*sizeof(float));
    cudaMalloc(&d_w1,W1*sizeof(float));
    cudaMalloc(&d_w2,W2*sizeof(float));
    cudaMalloc(&d_gw1,W1*sizeof(float));
    cudaMalloc(&d_gw2,W2*sizeof(float));
    cudaMemcpy(d_w1,h_w1.data(),W1*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_w2,h_w2.data(),W2*sizeof(float),cudaMemcpyHostToDevice);

    // Streams & events
    cudaStream_t s_cpy,s_cmp; cudaStreamCreate(&s_cpy); cudaStreamCreate(&s_cmp);
    cudaEvent_t ev0,ev_h2d,ev_k,ev_d2h,ev_end; cudaEventCreate(&ev0);
    cudaEventCreate(&ev_h2d); cudaEventCreate(&ev_k); cudaEventCreate(&ev_d2h);
    cudaEventCreate(&ev_end);

    // Training with 1 mini‑batch
    cudaEventRecord(ev0);
    cudaMemcpyAsync(d_x,h_x[0].data(),PIX_IN*sizeof(float),cudaMemcpyHostToDevice,s_cpy);
    cudaMemcpyAsync(d_tgt,h_eps[0].data(),PIX_OUT*sizeof(float),cudaMemcpyHostToDevice,s_cpy);
    cudaEventRecord(ev_h2d,s_cpy);

    zero_kernel<<<(W1+255)/256,256,0,s_cmp>>>(d_gw1,W1);
    zero_kernel<<<(W2+255)/256,256,0,s_cmp>>>(d_gw2,W2);

    dim3 g1((W+TILE-1)/TILE,(H+TILE-1)/TILE,CINTM), g2((W+TILE-1)/TILE,(H+TILE-1)/TILE,COUT);
    dim3 blk(TILE+2,TILE+2);

    conv3_fwd<CIN,CINTM><<<g1,blk,0,s_cmp>>>(d_x,d_w1,d_mid);
    relu_fwd<<<(PIX_MID+255)/256,256,0,s_cmp>>>(d_mid,PIX_MID);
    conv3_fwd<CINTM,COUT><<<g2,blk,0,s_cmp>>>(d_mid,d_w2,d_y);
    add_kernel<<<(PIX_OUT+255)/256,256,0,s_cmp>>>(d_x,d_y,PIX_OUT);

    float* d_loss; cudaMalloc(&d_loss,sizeof(float)); cudaMemsetAsync(d_loss,0,4,s_cmp);
    loss_mse_grad<<<(PIX_OUT+255)/256,256,0,s_cmp>>>(d_y,d_tgt,d_grad_y,d_loss,PIX_OUT);

    conv3_bwd_w<CINTM,COUT><<<dim3(COUT,CINTM),9,0,s_cmp>>>(d_mid,d_grad_y,d_gw2);
    conv3_bwd_data<CINTM,COUT><<<dim3((W+15)/16,(H+15)/16,CINTM),dim3(16,16),0,s_cmp>>>(d_grad_y,d_w2,d_mid);
    relu_bwd<<<(PIX_MID+255)/256,256,0,s_cmp>>>(d_mid,d_mid,d_mid,PIX_MID);
    conv3_bwd_w<CIN,CINTM><<<dim3(CINTM,CIN),9,0,s_cmp>>>(d_x,d_mid,d_gw1);
    conv3_bwd_data<CIN,CINTM><<<dim3((W+15)/16,(H+15)/16,CIN),dim3(16,16),0,s_cmp>>>(d_mid,d_w1,d_grad_y);

    sgd<<<(W1+255)/256,256,0,s_cmp>>>(d_w1,d_gw1,LR,W1);
    sgd<<<(W2+255)/256,256,0,s_cmp>>>(d_w2,d_gw2,LR,W2);

    float loss_val; cudaMemcpyAsync(&loss_val,d_loss,4,cudaMemcpyDeviceToHost,s_cpy);
    cudaEventRecord(ev_k,s_cmp); cudaEventRecord(ev_d2h,s_cpy);
    cudaEventRecord(ev_end); cudaEventSynchronize(ev_end);

    float ms_h2d,ms_k,ms_d2h,ms_tot;
    cudaEventElapsedTime(&ms_h2d,ev0,ev_h2d);
    cudaEventElapsedTime(&ms_k,ev_h2d,ev_k);
    cudaEventElapsedTime(&ms_d2h,ev_k,ev_d2h);
    cudaEventElapsedTime(&ms_tot,ev0,ev_end);

    // Image synthesis (1 000 steps)
    std::vector<float> h_latent(PIX_IN); fill_gauss(h_latent,rng,1.f);
    float* d_xt; cudaMalloc(&d_xt,PIX_IN*sizeof(float));
    cudaMemcpyAsync(d_xt,h_latent.data(),PIX_IN*sizeof(float),cudaMemcpyHostToDevice,s_cpy);
    cudaStreamSynchronize(s_cpy);
    curandStatePhilox4_32_10_t* d_states; cudaMalloc(&d_states,PIX_IN*sizeof(curandStatePhilox4_32_10_t));
    rng_init<<<(PIX_IN+255)/256,256,0,s_cmp>>>(d_states,PIX_IN,1234);

    float ms_sK=0.f; cudaEvent_t e1,e2; cudaEventCreate(&e1); cudaEventCreate(&e2);
    cudaEventRecord(e1);
    for(int t=T_STEPS-1;t>=0;--t){
        // eps_pred
        conv3_fwd<CIN,CINTM><<<g1,blk,0,s_cmp>>>(d_xt,d_w1,d_mid);
        relu_fwd<<<(PIX_MID+255)/256,256,0,s_cmp>>>(d_mid,PIX_MID);
        conv3_fwd<CINTM,COUT><<<g2,blk,0,s_cmp>>>(d_mid,d_w2,d_y);
        add_kernel<<<(PIX_OUT+255)/256,256,0,s_cmp>>>(d_xt,d_y,PIX_OUT);

        float ac=sched.ac[t], ac_prev=(t? sched.ac[t-1]:1.f), beta=sched.beta[t];
        float sqrt_one_minus=sqrtf(1.f-ac);
        float coeff=beta/sqrt_one_minus;
        float inv_sqrt_alpha=1.f/sqrtf(1.f-beta);
        float sigma=sqrtf((1.f-ac_prev)/(1.f-ac)*beta);

        reverse_step_kernel<<<(PIX_IN+255)/256,256,0,s_cmp>>>
            (d_xt,d_y,d_states,PIX_IN,inv_sqrt_alpha,coeff,sigma);
    }
    cudaEventRecord(e2); cudaEventSynchronize(e2); cudaEventElapsedTime(&ms_sK,e1,e2);
    cudaMemcpyAsync(h_latent.data(),d_xt,PIX_IN*sizeof(float),cudaMemcpyDeviceToHost,s_cpy);
    cudaStreamSynchronize(s_cpy);

    std::cout<<"GPU timings (ms):\n"
    <<"  Host-to-Device copy time:      : "<<ms_h2d<<"\n"
    <<"  Kernel execution time:         : "<<ms_k<<"\n"
    <<"  Device-to-Host copy time:      : "<<ms_d2h<<"\n"
    <<"  Total GPU time:                : "<<ms_tot<<"\n";
    std::cout<<"GPU loss after 1 update : "<<loss_val<<"\n";
    std::cout<<"GPU kernel time (1000 steps) : "<<ms_sK<<" ms\n";

}
