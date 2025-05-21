#include <cuda_runtime.h>
#include <cufft.h>
#include <vector>
#include <random>
#include <iostream>
#include <iomanip>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// CUDA helpers
#define CUDA_CHECK(x)  do{ cudaError_t rc=(x); if(rc!=cudaSuccess){ \
  std::cerr<<"CUDA "<<cudaGetErrorString(rc)<<" @"<<__LINE__<<"\n"; std::exit(EXIT_FAILURE);} }while(0)
#define CUFFT_CHECK(x) do{ cufftResult rc=(x); if(rc!=CUFFT_SUCCESS){ \
  std::cerr<<"cuFFT "<<rc<<" @"<<__LINE__<<"\n"; std::exit(EXIT_FAILURE);} }while(0)

// float2 helpers & overloads
__host__ __device__ inline float2 f2(float re,float im){ return make_float2(re,im); }
__host__ __device__ inline float2 operator+(float2 a,float2 b){ return f2(a.x+b.x,a.y+b.y); }
__host__ __device__ inline float2 operator-(float2 a,float2 b){ return f2(a.x-b.x,a.y-b.y); }
__host__ __device__ inline float2 operator*(float2 a,float2 b){ return f2(a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x); }
__host__ __device__ inline float2& operator+=(float2& a,const float2& b){ a.x+=b.x; a.y+=b.y; return a; }  // ★ NEW
__host__ __device__ inline float2& operator-=(float2& a,const float2& b){ a.x-=b.x; a.y-=b.y; return a; }  // ★ NEW
__host__ __device__ inline float2 cexpj(float th){ float s,c; sincosf(th,&s,&c); return f2(c,s); }

// Struct
struct Times{ float h2d=0,kern=0,d2h=0; };

// 0. Naïve GPU DFT
__global__ void dft_naive(const float* x,float2* X,int N){
    int k=blockIdx.x*blockDim.x+threadIdx.x; if(k>=N) return;
    float2 acc{0,0}; float w=-2.f*M_PI/N;
    for(int n=0;n<N;++n) acc+=f2(x[n],0)*cexpj(w*k*n);
    X[k]=acc;
}
Times run_naive(const std::vector<float>& h,std::vector<float2>& out){
    int N=h.size(); float *d_x; float2* d_X; CUDA_CHECK(cudaMalloc(&d_x,sizeof(float)*N));
    CUDA_CHECK(cudaMalloc(&d_X,sizeof(float2)*N));
    cudaEvent_t e0,e1,e2,e3; cudaEventCreate(&e0);cudaEventCreate(&e1);
    cudaEventCreate(&e2); cudaEventCreate(&e3);
    cudaEventRecord(e0); CUDA_CHECK(cudaMemcpy(d_x,h.data(),sizeof(float)*N,cudaMemcpyHostToDevice)); cudaEventRecord(e1);
    int th=256,bl=(N+th-1)/th; dft_naive<<<bl,th>>>(d_x,d_X,N); cudaEventRecord(e2);
    out.resize(N); CUDA_CHECK(cudaMemcpy(out.data(),d_X,sizeof(float2)*N,cudaMemcpyDeviceToHost)); cudaEventRecord(e3); cudaEventSynchronize(e3);
    Times t; cudaEventElapsedTime(&t.h2d,e0,e1); cudaEventElapsedTime(&t.kern,e1,e2); cudaEventElapsedTime(&t.d2h,e2,e3);
    cudaFree(d_x); cudaFree(d_X); return t;
}

// 1. Shared-memory tiled DFT
constexpr int TILE = 256;
__global__ void dft_shared(const float* x,float2* X,int N){
    extern __shared__ float sh[];
    int k=blockIdx.x*blockDim.x+threadIdx.x; if(k>=N) return;
    float2 acc{0,0}; float w=-2.f*M_PI/N;
    for(int base=0;base<N;base+=TILE){
        int tid=threadIdx.x; if(tid<TILE && base+tid<N) sh[tid]=x[base+tid]; __syncthreads();
        for(int j=0;j<TILE && base+j<N;++j) acc+=f2(sh[j],0)*cexpj(w*k*(base+j));
        __syncthreads();
    }
    X[k]=acc;
}
Times run_shared(const std::vector<float>& h,std::vector<float2>& out){
    int N=h.size(); float *d_x; float2 *d_X; CUDA_CHECK(cudaMalloc(&d_x,sizeof(float)*N));
    CUDA_CHECK(cudaMalloc(&d_X,sizeof(float2)*N));
    cudaEvent_t e0,e1,e2,e3; cudaEventCreate(&e0);cudaEventCreate(&e1);cudaEventCreate(&e2);cudaEventCreate(&e3);
    cudaEventRecord(e0); CUDA_CHECK(cudaMemcpy(d_x,h.data(),sizeof(float)*N,cudaMemcpyHostToDevice)); cudaEventRecord(e1);
    int th=256,bl=(N+th-1)/th; dft_shared<<<bl,th,TILE*sizeof(float)>>>(d_x,d_X,N); cudaEventRecord(e2);
    out.resize(N); CUDA_CHECK(cudaMemcpy(out.data(),d_X,sizeof(float2)*N,cudaMemcpyDeviceToHost)); cudaEventRecord(e3); cudaEventSynchronize(e3);
    Times t; cudaEventElapsedTime(&t.h2d,e0,e1); cudaEventElapsedTime(&t.kern,e1,e2); cudaEventElapsedTime(&t.d2h,e2,e3);
    cudaFree(d_x); cudaFree(d_X); return t;
}

// 2. Mixed-radix 2/3/5 FFT
__device__ void bf3(float2&a,float2&b,float2&c){
    const float2 w1=f2(-0.5f,-0.8660254039f), w2=f2(-0.5f,0.8660254039f);
    float2 t1=a+b+c, t2=a+b*w1+c*w2, t3=a+b*w2+c*w1; a=t1; b=t2; c=t3;
}
__device__ void bf5(float2&a,float2&b,float2&c,float2&d,float2&e){
    const float tau=0.3090169944f, sin72=0.9510565163f;
    float2 t1=b+e, t2=b-e, t3=c+d, t4=c-d;
    float2 t5=a+t1+t3;
    float2 t6=a+f2(-0.25f,0)*t1+f2(-0.25f,0)*t3;
    float2 t7=f2(0,tau)*t2+f2(0,-tau)*t4;
    float2 t8=f2(-sin72,0)*t2+f2(sin72,0)*t4;
    b=t6+t7; e=t6-t7; c=t6+t8; d=t6-t8; a=t5;
}
__global__ void mixed_kernel(float2* d,int n,int radix,int stage){
    int tid=blockIdx.x*blockDim.x+threadIdx.x, grp=tid/stage, idx=tid%stage;
    if(radix==3){
        int base=grp*stage*3+idx; if(base+2*stage<n){ float2 a=d[base],b=d[base+stage],c=d[base+2*stage]; bf3(a,b,c);
            d[base]=a; d[base+stage]=b; d[base+2*stage]=c; }
    }else if(radix==5){
        int base=grp*stage*5+idx; if(base+4*stage<n){ float2 a=d[base],b=d[base+stage],c=d[base+2*stage],f=d[base+3*stage],e=d[base+4*stage];
            bf5(a,b,c,f,e); d[base]=a; d[base+stage]=b; d[base+2*stage]=c; d[base+3*stage]=f; d[base+4*stage]=e; }
    }
}
__device__ __forceinline__ float2 Wmix(int k,int N){ return cexpj(-2.f*M_PI*k/N); }
__global__ void bitrev(float2* d,int n,int lg){
    int tid=blockIdx.x*blockDim.x+threadIdx.x; if(tid<n){ unsigned r=__brev(tid)>>(32-lg); if(r>tid){ auto t=d[tid]; d[tid]=d[r]; d[r]=t; }}}
__global__ void radix2(float2* d,int n,int half){
    int tid=blockIdx.x*blockDim.x+threadIdx.x; if(tid>=n/2) return;
    int span=half*2, j=tid%half, base=(tid/half)*span, p=base+j;
    float2 u=d[p], v=d[p+half]*Wmix(j,span); d[p]=u+v; d[p+half]=u-v;
}
void engine_radix2(float2* d,int n){
    int th=256,lg=static_cast<int>(log2f(n)); bitrev<<<(n+th-1)/th,th>>>(d,n,lg);
    for(int s=1;s<=lg;++s){ int half=1<<(s-1); radix2<<<((n>>1)+th-1)/th,th>>>(d,n,half);}
}
void engine_mixed(float2* d,int n){
    int rem=n,stage=1,th=256,bl;
    while(rem%5==0){ bl=(n/(stage*5)+th-1)/th; mixed_kernel<<<bl,th>>>(d,n,5,stage); stage*=5; rem/=5;}
    while(rem%3==0){ bl=(n/(stage*3)+th-1)/th; mixed_kernel<<<bl,th>>>(d,n,3,stage); stage*=3; rem/=3;}
    if(rem>1) engine_radix2(d,n);
}
Times run_mixed(const std::vector<float>& h,std::vector<float2>& out){
    int N=h.size(); std::vector<float2> in(N); for(int i=0;i<N;++i) in[i]=f2(h[i],0);
    float2* d; CUDA_CHECK(cudaMalloc(&d,sizeof(float2)*N));
    cudaEvent_t e0,e1,e2,e3; cudaEventCreate(&e0);cudaEventCreate(&e1);cudaEventCreate(&e2);cudaEventCreate(&e3);
    cudaEventRecord(e0); CUDA_CHECK(cudaMemcpy(d,in.data(),sizeof(float2)*N,cudaMemcpyHostToDevice)); cudaEventRecord(e1);
    engine_mixed(d,N); cudaEventRecord(e2);
    out.resize(N); CUDA_CHECK(cudaMemcpy(out.data(),d,sizeof(float2)*N,cudaMemcpyDeviceToHost)); cudaEventRecord(e3); cudaEventSynchronize(e3);
    Times t; cudaEventElapsedTime(&t.h2d,e0,e1); cudaEventElapsedTime(&t.kern,e1,e2); cudaEventElapsedTime(&t.d2h,e2,e3); cudaFree(d); return t;
}

// 3. Split-radix FFT
__device__ __forceinline__ float2 Wsr(int k,int N){ return cexpj(-2.f*M_PI*k/N); }
__global__ void sr_post(float2* d,int N){
    int k=blockIdx.x*blockDim.x+threadIdx.x; if(k>=N/4) return;
    float2 a=d[2*k], b=d[2*k+1], c=d[k+N/2], e=d[k+N/2+1];
    float2 t1=b+e, t2=b-e; d[2*k]=a+c; d[2*k+1]=t1; d[k+N/2]=(a-c)*Wsr(k,N); d[k+N/2+1]=t2*Wsr(k,N)*f2(0,1);
}
void split_fft(float2* d,int N){
    int th=256,lg=static_cast<int>(log2f(N)); bitrev<<<(N+th-1)/th,th>>>(d,N,lg);
    for(int s=1;s<=lg;++s){ int half=1<<(s-1); radix2<<<((N>>1)+th-1)/th,th>>>(d,N,half);}
    sr_post<<<(N/4+th-1)/th,th>>>(d,N);
}
Times run_split(const std::vector<float>& h,std::vector<float2>& out){
    int N=h.size(); std::vector<float2> in(N); for(int i=0;i<N;++i) in[i]=f2(h[i],0);
    float2* d; CUDA_CHECK(cudaMalloc(&d,sizeof(float2)*N)); cudaEvent_t e0,e1,e2,e3;cudaEventCreate(&e0);cudaEventCreate(&e1);cudaEventCreate(&e2);cudaEventCreate(&e3);
    cudaEventRecord(e0); CUDA_CHECK(cudaMemcpy(d,in.data(),sizeof(float2)*N,cudaMemcpyHostToDevice)); cudaEventRecord(e1);
    split_fft(d,N); cudaEventRecord(e2);
    out.resize(N); CUDA_CHECK(cudaMemcpy(out.data(),d,sizeof(float2)*N,cudaMemcpyDeviceToHost)); cudaEventRecord(e3); cudaEventSynchronize(e3);
    Times t; cudaEventElapsedTime(&t.h2d,e0,e1); cudaEventElapsedTime(&t.kern,e1,e2); cudaEventElapsedTime(&t.d2h,e2,e3);
    cudaFree(d); return t;
}

// 4. Bluestein / Chirp-Z
int next_pow2(int n){ int p=1; while(p<n) p<<=1; return p; }
__global__ void blu_build(const float* x,float2* A,float2* B,int N,float piN){
    int n=blockIdx.x*blockDim.x+threadIdx.x; if(n<N){ float ang=piN*n*n; A[n]=f2(x[n],0)*cexpj( ang); B[n]=cexpj(-ang);} }
__global__ void blu_pad(float2* B,int N,int M){ int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=M) return; if(i>=N && i<M-N+1) B[i]=f2(0,0); if(i>0 && i<N) B[M-i]=B[i]; }
__global__ void pointmul(const float2* A,const float2* B,float2* C,int M){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<M) C[i]=A[i]*B[i]; }
__global__ void blu_final(const float2* y,float2* X,int N,int M,float piN){
    int k=blockIdx.x*blockDim.x+threadIdx.x; if(k<N){ X[k]=f2(y[k].x/M,y[k].y/M)*cexpj(-piN*k*k);} }
Times run_blu(const std::vector<float>& h,std::vector<float2>& out){
    int N=h.size(), M=next_pow2(2*N-1), th=256,bN=(N+th-1)/th,bM=(M+th-1)/th;
    float *d_x; float2 *d_A,*d_B,*d_C,*d_X; CUDA_CHECK(cudaMalloc(&d_x,sizeof(float)*N));
    CUDA_CHECK(cudaMalloc(&d_A,sizeof(float2)*M)); CUDA_CHECK(cudaMalloc(&d_B,sizeof(float2)*M));
    CUDA_CHECK(cudaMalloc(&d_C,sizeof(float2)*M)); CUDA_CHECK(cudaMalloc(&d_X,sizeof(float2)*N));
    cudaEvent_t e0,e1,e2; cudaEventCreate(&e0);cudaEventCreate(&e1);cudaEventCreate(&e2);
    cudaEventRecord(e0); CUDA_CHECK(cudaMemcpy(d_x,h.data(),sizeof(float)*N,cudaMemcpyHostToDevice));
    blu_build<<<bN,th>>>(d_x,d_A,d_B,N,M_PI/N); blu_pad<<<bM,th>>>(d_B,N,M);
    cufftHandle plan; CUFFT_CHECK(cufftPlan1d(&plan,M,CUFFT_C2C,1));
    CUFFT_CHECK(cufftExecC2C(plan,(cufftComplex*)d_A,(cufftComplex*)d_A,CUFFT_FORWARD));
    CUFFT_CHECK(cufftExecC2C(plan,(cufftComplex*)d_B,(cufftComplex*)d_B,CUFFT_FORWARD));
    pointmul<<<bM,th>>>(d_A,d_B,d_C,M);
    CUFFT_CHECK(cufftExecC2C(plan,(cufftComplex*)d_C,(cufftComplex*)d_C,CUFFT_INVERSE));
    blu_final<<<bN,th>>>(d_C,d_X,N,M,M_PI/N); cudaEventRecord(e1);
    out.resize(N); CUDA_CHECK(cudaMemcpy(out.data(),d_X,sizeof(float2)*N,cudaMemcpyDeviceToHost)); cudaEventRecord(e2); cudaEventSynchronize(e2);
    Times t; cudaEventElapsedTime(&t.kern,e0,e1); cudaEventElapsedTime(&t.d2h,e1,e2); t.h2d=0;
    cufftDestroy(plan); cudaFree(d_x); cudaFree(d_A); cudaFree(d_B); cudaFree(d_C); cudaFree(d_X); return t;
}

// 5. cuFFT
Times run_cufft(const std::vector<float>& h,std::vector<float2>& out){
    int N=h.size(); std::vector<float2> in(N); for(int i=0;i<N;++i) in[i]=f2(h[i],0);
    cufftComplex *d_in,*d_out; CUDA_CHECK(cudaMalloc(&d_in,sizeof(cufftComplex)*N));
    CUDA_CHECK(cudaMalloc(&d_out,sizeof(cufftComplex)*N)); cudaEvent_t e0,e1,e2;cudaEventCreate(&e0);cudaEventCreate(&e1);cudaEventCreate(&e2);
    cudaEventRecord(e0); CUDA_CHECK(cudaMemcpy(d_in,in.data(),sizeof(cufftComplex)*N,cudaMemcpyHostToDevice));
    cufftHandle plan; CUFFT_CHECK(cufftPlan1d(&plan,N,CUFFT_C2C,1));
    CUFFT_CHECK(cufftExecC2C(plan,d_in,d_out,CUFFT_FORWARD)); cudaEventRecord(e1);
    out.resize(N); CUDA_CHECK(cudaMemcpy(out.data(),d_out,sizeof(cufftComplex)*N,cudaMemcpyDeviceToHost)); cudaEventRecord(e2); cudaEventSynchronize(e2);
    Times t; cudaEventElapsedTime(&t.kern,e0,e1); cudaEventElapsedTime(&t.d2h,e1,e2); t.h2d=0;
    cufftDestroy(plan); cudaFree(d_in); cudaFree(d_out); return t;
}

int main(int argc,char**argv)
{
    const int N=(argc>1)?std::atoi(argv[1]):16384;
    std::cout<<"Vector length N = "<<N<<"\n\n";
    std::vector<float> sig(N); std::mt19937 rng(1); std::uniform_real_distribution<float> dist(-1,1);
    for(float& v:sig) v=dist(rng);

    struct A{ const char* name; Times(*run)(const std::vector<float>&,std::vector<float2>&);} algs[]={
        {"Naïve DFT   :",   run_naive },
        {"Shared DFT  :",  run_shared},
        {"Mixed-radix :", run_mixed },
        {"Split-radix :", run_split },
        {"Bluestein   :",   run_blu   },
        {"cuFFT       :",       run_cufft }
    };

    std::cout<<std::fixed<<std::setprecision(3);
    for(auto& a:algs){
        std::vector<float2> X; Times t=a.run(sig,X);
        double sum=0; for(auto& v:X) sum+=std::hypot(v.x,v.y);
        std::cout<<std::setw(12)<<std::left<<a.name<<"  Σ|X|="<<std::setw(12)<<sum
                 <<" H2D "<<std::setw(6)<<t.h2d
                 <<" Kern "<<std::setw(7)<<t.kern
                 <<" D2H "<<std::setw(6)<<t.d2h
                 <<" Total "<<t.h2d+t.kern+t.d2h<<" ms\n";
    }
    return 0;
}