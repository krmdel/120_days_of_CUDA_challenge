#include <cuda_runtime.h>
#include <cufft.h>
#include <cuComplex.h>
#include <device_launch_parameters.h>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Error-check helpers
#define CUDA_CHECK(x)  do{ cudaError_t rc=(x); if(rc!=cudaSuccess){           \
    std::cerr<<"CUDA "<<cudaGetErrorString(rc)<<" @ "<<__FILE__<<":"<<__LINE__;\
    std::exit(EXIT_FAILURE);} } while(0)
#define CUFFT_CHECK(x) do{ cufftResult rc=(x); if(rc!=CUFFT_SUCCESS){         \
    std::cerr<<"cuFFT "<<rc<<" @ "<<__FILE__<<":"<<__LINE__;                  \
    std::exit(EXIT_FAILURE);} } while(0)

// Device complex helpers
__device__ __forceinline__ float2 cAdd(float2 a,float2 b){return {a.x+b.x,a.y+b.y};}
__device__ __forceinline__ float2 cSub(float2 a,float2 b){return {a.x-b.x,a.y-b.y};}
__device__ __forceinline__ float2 cMul(float2 a,float2 b){return {a.x*b.x-a.y*b.y,
                                                                  a.x*b.y+a.y*b.x};}
__device__ __forceinline__ float2 cExp(float t){ float s,c; sincosf(t,&s,&c); return {c,s}; }
__device__ __forceinline__ float2 Wnk(int k,int N){ float s,c; sincosf(-2.f*M_PI*k/N,&s,&c); return {c,s}; }

// 32×32 transpose kernel
__global__ void transpose32(float2*dst,const float2*src,int rows,int cols)
{
    __shared__ float2 tile[32][33];                     // 1-col padding avoids bank conflicts
    int x=blockIdx.x*32+threadIdx.x;
    int y=blockIdx.y*32+threadIdx.y;
    if(x<cols && y<rows) tile[threadIdx.y][threadIdx.x]=src[y*cols+x];
    __syncthreads();
    x=blockIdx.y*32+threadIdx.x;
    y=blockIdx.x*32+threadIdx.y;
    if(x<rows && y<cols) dst[y*rows+x]=tile[threadIdx.x][threadIdx.y];
}

// 1-D FFT engines
__global__ void bitrev(float2*buf,int n,int lg)
{
    int tid=blockIdx.x*blockDim.x+threadIdx.x;
    if(tid>=n) return;
    unsigned r=__brev(tid)>>(32-lg);
    if(r>tid){ float2 t=buf[tid]; buf[tid]=buf[r]; buf[r]=t; }
}
__global__ void radix2_stage(float2*buf,int n,int half)
{
    int tid=blockIdx.x*blockDim.x+threadIdx.x;
    if(tid>=n/2) return;
    int span=half*2;
    int j   =tid%half;
    int p   =(tid/half)*span+j;
    float2 u=buf[p];
    float2 v=cMul(buf[p+half], Wnk(j,span));
    buf[p]     =cAdd(u,v);
    buf[p+half]=cSub(u,v);
}
void fft_radix2(float2*buf,int n)
{
    int lg=static_cast<int>(log2f(n));
    int th=256, bl=(n+th-1)/th;
    bitrev<<<bl,th>>>(buf,n,lg);
    for(int s=1;s<=lg;++s){
        int half=1<<(s-1);
        bl=((n>>1)+th-1)/th;
        radix2_stage<<<bl,th>>>(buf,n,half);
    }
}

// Radix-3 / radix-5 butterflies
__device__ void bf3(float2&a,float2&b,float2&c){
    const float2 w1={-0.5f,-0.8660254039f};
    const float2 w2={-0.5f, 0.8660254039f};
    float2 t1=cAdd(a,cAdd(b,c));
    float2 t2=cAdd(a,cAdd(cMul(b,w1),cMul(c,w2)));
    float2 t3=cAdd(a,cAdd(cMul(b,w2),cMul(c,w1)));
    a=t1; b=t2; c=t3;
}
__device__ void bf5(float2&a,float2&b,float2&c,float2&d4,float2&e){
    const float tau=0.3090169944f, sin72=0.9510565163f;
    float2 t1=cAdd(b,e), t2=cSub(b,e), t3=cAdd(c,d4), t4=cSub(c,d4);
    float2 t5=cAdd(a,cAdd(t1,t3));
    float2 t6=cAdd(a, {-0.25f*(t1.x+t3.x), -0.25f*(t1.y+t3.y)});
    float2 t7=cAdd(cMul({0,tau},t2), cMul({0,-tau},t4));
    float2 t8=cAdd(cMul({-sin72,0},t2), cMul({sin72,0},t4));
    b=cAdd(t6,t7); e=cSub(t6,t7);
    c=cAdd(t6,t8); d4=cSub(t6,t8); a=t5;
}
__global__ void mixed_kernel(float2*buf,int n,int radix,int span)
{
    int tid=blockIdx.x*blockDim.x+threadIdx.x;
    int grp=tid/span, idx=tid%span;
    if(radix==3){
        int base=grp*span*3+idx;
        if(base+2*span<n){
            float2 a=buf[base], b=buf[base+span], c=buf[base+2*span];
            bf3(a,b,c); buf[base]=a; buf[base+span]=b; buf[base+2*span]=c;
        }
    }else{                              // radix-5
        int base=grp*span*5+idx;
        if(base+4*span<n){
            float2 a=buf[base], b=buf[base+span], c=buf[base+2*span],
                   d4=buf[base+3*span], e=buf[base+4*span];
            bf5(a,b,c,d4,e);
            buf[base]=a; buf[base+span]=b; buf[base+2*span]=c;
            buf[base+3*span]=d4; buf[base+4*span]=e;
        }
    }
}
void fft_mixed(float2*buf,int n)
{
    int rem=n, span=1, th=256, bl;
    while(rem%5==0){ bl=(n/(span*5)+th-1)/th;
        mixed_kernel<<<bl,th>>>(buf,n,5,span); span*=5; rem/=5; }
    while(rem%3==0){ bl=(n/(span*3)+th-1)/th;
        mixed_kernel<<<bl,th>>>(buf,n,3,span); span*=3; rem/=3; }
    if(rem>1) fft_radix2(buf,n);
}

// Split-radix post-processing
__global__ void split_post(float2*buf,int N)
{
    int k=blockIdx.x*blockDim.x+threadIdx.x; if(k>=N/4) return;
    float2 a=buf[2*k], b1=buf[2*k+1], c=buf[k+N/2], e=buf[k+N/2+1];
    float2 t1=cAdd(b1,e), t2=cSub(b1,e);
    buf[2*k]      =cAdd(a,c);
    buf[2*k+1]    =t1;
    buf[k+N/2]    =cMul(cSub(a,c), Wnk(k,N));
    buf[k+N/2+1]  =cMul(t2, {0,1});
}
void fft_split(float2*buf,int N)
{
    fft_radix2(buf,N);
    int th=256, bl=(N/4+th-1)/th;
    split_post<<<bl,th>>>(buf,N);
}

// Bluestein (Chirp-Z)
__global__ void blu_build(const float2*x,float2*A,float2*B,int N,float k)
{
    int n=blockIdx.x*blockDim.x+threadIdx.x; if(n>=N) return;
    float2 w=cExp( k*n*n); float2 wm=cExp(-k*n*n);
    A[n]=cMul(x[n], w); B[n]=wm;
}
__global__ void blu_pad(float2*B,int N,int M)
{
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=M) return;
    if(i>=N && i<M-N+1) B[i]={0,0};
    if(i>0 && i<N)      B[M-i]=B[i];
}
__global__ void blu_mul(const float2*A,const float2*B,float2*C,int M)
{
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<M) C[i]=cMul(A[i],B[i]);
}
__global__ void blu_final(const float2*y,float2*X,int N,int M,float k)
{
    int n=blockIdx.x*blockDim.x+threadIdx.x; if(n>=N) return;
    float2 v=y[n]; v.x/=M; v.y/=M;
    X[n]=cMul(v, cExp(-k*n*n));
}
int next_pow2(int v){ int p=1; while(p<v) p<<=1; return p; }
void fft_bluestein(float2*d,int N)
{
    const int M=next_pow2(2*N-1);
    const float k=M_PI/N;
    float2 *dA,*dB,*dC;
    CUDA_CHECK(cudaMalloc(&dA,sizeof(float2)*M));
    CUDA_CHECK(cudaMalloc(&dB,sizeof(float2)*M));
    CUDA_CHECK(cudaMalloc(&dC,sizeof(float2)*M));
    CUDA_CHECK(cudaMemset(dA,0,sizeof(float2)*M));
    CUDA_CHECK(cudaMemset(dB,0,sizeof(float2)*M));

    int th=256, blN=(N+th-1)/th, blM=(M+th-1)/th;
    blu_build<<<blN,th>>>(d,dA,dB,N,k);
    blu_pad  <<<blM,th>>>(dB,N,M);

    cufftHandle plan; CUFFT_CHECK(cufftPlan1d(&plan,M,CUFFT_C2C,1));
    CUFFT_CHECK(cufftExecC2C(plan,(cufftComplex*)dA,(cufftComplex*)dA,CUFFT_FORWARD));
    CUFFT_CHECK(cufftExecC2C(plan,(cufftComplex*)dB,(cufftComplex*)dB,CUFFT_FORWARD));
    blu_mul  <<<blM,th>>>(dA,dB,dC,M);
    CUFFT_CHECK(cufftExecC2C(plan,(cufftComplex*)dC,(cufftComplex*)dC,CUFFT_INVERSE));
    blu_final<<<blN,th>>>(dC,d,N,M,k);

    cufftDestroy(plan); cudaFree(dA); cudaFree(dB); cudaFree(dC);
}

// 2-D wrapper: rows → transpose → cols
using FFT1D = void(*)(float2*,int);
void fft2d_generic(float2*d,float2*tmp,int W,int H,FFT1D eng)
{
    for(int r=0;r<H;++r) eng(d+(size_t)r*W,W);
    dim3 b(32,32), g((W+31)/32,(H+31)/32);
    transpose32<<<g,b>>>(tmp,d,H,W);
    for(int r=0;r<W;++r) eng(tmp+(size_t)r*H,H);
    dim3 g2((H+31)/32,(W+31)/32);
    transpose32<<<g2,b>>>(d,tmp,W,H);
}

// Space-domain 2-D DFT kernels
__global__ void dft2d_naive_kernel(const float*src,float2*dst,int W,int H)
{
    int u=blockIdx.x*blockDim.x+threadIdx.x;
    int v=blockIdx.y*blockDim.y+threadIdx.y;
    if(u>=W||v>=H) return;
    float re=0,im=0;
    for(int y=0;y<H;++y)
        for(int x=0;x<W;++x){
            float ang=-2.f*M_PI*(u*x/(float)W+v*y/(float)H);
            float s,c; sincosf(ang,&s,&c);
            float val=src[y*W+x];
            re+=val*c; im+=val*s;
        }
    dst[v*W+u]={re,im};
}
__device__ __forceinline__ void rot_inc(float&c,float&s,float cd,float sd)
{ float t=c*cd - s*sd; s=c*sd + s*cd; c=t; }
constexpr int TILE=16;
__global__ void dft2d_tiled_kernel(const float*src,float2*dst,int W,int H)
{
    int u=blockIdx.x*blockDim.x+threadIdx.x;
    int v=blockIdx.y*blockDim.y+threadIdx.y;
    if(u>=W||v>=H) return;
    float kx=-2.f*M_PI*u/W, ky=-2.f*M_PI*v/H;
    float sdx,cdx,sdy,cdy; sincosf(kx,&sdx,&cdx); sincosf(ky,&sdy,&cdy);
    __shared__ float tile[TILE][TILE];
    float accR=0,accI=0;

    for(int y0=0;y0<H;y0+=TILE){
        float sy,cy; sincosf(ky*y0,&sy,&cy);
        for(int x0=0;x0<W;x0+=TILE){
            int gx=x0+threadIdx.x, gy=y0+threadIdx.y;
            tile[threadIdx.y][threadIdx.x]=(gx<W && gy<H)?src[gy*W+gx]:0;
            __syncthreads();

            float sx,cx; sincosf(kx*x0,&sx,&cx);
            float syRow=sy, cyRow=cy;
            #pragma unroll
            for(int ty=0;ty<TILE && y0+ty<H;++ty){
                float sxCol=sx, cxCol=cx;
                #pragma unroll
                for(int tx=0;tx<TILE && x0+tx<W;++tx){
                    float cosT=cxCol*cyRow - sxCol*syRow;
                    float sinT=cxCol*syRow + sxCol*cyRow;
                    float val=tile[ty][tx];
                    accR+=val*cosT; accI+=val*sinT;
                    rot_inc(cxCol,sxCol,cdx,sdx);
                }
                rot_inc(cyRow,syRow,cdy,sdy);
            }
            __syncthreads();
        }
    }
    dst[v*W+u]={accR,accI};
}

// Benchmark helper structures / functions
struct Res{ double sum; float h2d,ker,d2h; float total()const{return h2d+ker+d2h;} };

Res bench_space(const std::vector<float>&h,int W,int H,bool tiled)
{
    size_t N=(size_t)W*H; float *d_img; float2*d_out;
    CUDA_CHECK(cudaMalloc(&d_img,sizeof(float)*N));
    CUDA_CHECK(cudaMalloc(&d_out,sizeof(float2)*N));

    cudaEvent_t t0,t1,t2,t3,t4; cudaEventCreate(&t0);cudaEventCreate(&t1);
    cudaEventCreate(&t2);cudaEventCreate(&t3);cudaEventCreate(&t4);

    cudaEventRecord(t0);
    CUDA_CHECK(cudaMemcpy(d_img,h.data(),sizeof(float)*N,cudaMemcpyHostToDevice));
    cudaEventRecord(t1);

    dim3 b(16,16), g((W+15)/16,(H+15)/16);
    cudaEventRecord(t2);
    if(tiled) dft2d_tiled_kernel<<<g,b>>>(d_img,d_out,W,H);
    else      dft2d_naive_kernel<<<g,b>>>(d_img,d_out,W,H);
    cudaEventRecord(t3);

    std::vector<float2> host(N);
    CUDA_CHECK(cudaMemcpy(host.data(),d_out,sizeof(float2)*N,cudaMemcpyDeviceToHost));
    cudaEventRecord(t4); CUDA_CHECK(cudaDeviceSynchronize());

    float h2d,ker,d2h;
    cudaEventElapsedTime(&h2d,t0,t1);
    cudaEventElapsedTime(&ker,t2,t3);
    cudaEventElapsedTime(&d2h,t3,t4);

    double sum=0; for(const auto&c:host) sum+=std::hypot(c.x,c.y);

    cudaFree(d_img); cudaFree(d_out);
    cudaEventDestroy(t0);cudaEventDestroy(t1);cudaEventDestroy(t2);
    cudaEventDestroy(t3);cudaEventDestroy(t4);

    return {sum,h2d,ker,d2h};
}
Res bench_fft(const std::vector<float>&h,int W,int H,FFT1D eng)
{
    size_t N=(size_t)W*H;
    std::vector<float2> host(N); for(size_t i=0;i<N;++i) host[i]={h[i],0};

    float2 *d,*tmp; CUDA_CHECK(cudaMalloc(&d,sizeof(float2)*N));
    CUDA_CHECK(cudaMalloc(&tmp,sizeof(float2)*N));

    cudaEvent_t t0,t1,t2,t3; cudaEventCreate(&t0);cudaEventCreate(&t1);
    cudaEventCreate(&t2);cudaEventCreate(&t3);

    cudaEventRecord(t0);
    CUDA_CHECK(cudaMemcpy(d,host.data(),sizeof(float2)*N,cudaMemcpyHostToDevice));
    cudaEventRecord(t1);

    cudaEventRecord(t2);
    fft2d_generic(d,tmp,W,H,eng);
    cudaEventRecord(t3);

    CUDA_CHECK(cudaMemcpy(host.data(),d,sizeof(float2)*N,cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize());

    float h2d,ker; cudaEventElapsedTime(&h2d,t0,t1); cudaEventElapsedTime(&ker,t1,t3);
    double sum=0; for(const auto&c:host) sum+=std::hypot(c.x,c.y);

    cudaFree(d); cudaFree(tmp);
    cudaEventDestroy(t0);cudaEventDestroy(t1);cudaEventDestroy(t2);cudaEventDestroy(t3);

    return {sum,h2d,ker,0};
}
Res bench_cufft(const std::vector<float>&h,int W,int H)
{
    size_t N=(size_t)W*H;
    std::vector<cuFloatComplex> host(N);
    for(size_t i=0;i<N;++i) host[i]=make_cuFloatComplex(h[i],0);

    cuFloatComplex*d; CUDA_CHECK(cudaMalloc(&d,sizeof(cuFloatComplex)*N));

    cudaEvent_t t0,t1,t2,t3,t4; cudaEventCreate(&t0);cudaEventCreate(&t1);
    cudaEventCreate(&t2);cudaEventCreate(&t3);cudaEventCreate(&t4);

    cudaEventRecord(t0);
    CUDA_CHECK(cudaMemcpy(d,host.data(),sizeof(cuFloatComplex)*N,cudaMemcpyHostToDevice));
    cudaEventRecord(t1);

    cufftHandle plan; CUFFT_CHECK(cufftPlan2d(&plan,H,W,CUFFT_C2C));
    cudaEventRecord(t2);
    CUFFT_CHECK(cufftExecC2C(plan,d,d,CUFFT_FORWARD));
    cudaEventRecord(t3);

    CUDA_CHECK(cudaMemcpy(host.data(),d,sizeof(cuFloatComplex)*N,cudaMemcpyDeviceToHost));
    cudaEventRecord(t4); CUDA_CHECK(cudaDeviceSynchronize());

    float h2d,ker,d2h;
    cudaEventElapsedTime(&h2d,t0,t1);
    cudaEventElapsedTime(&ker,t2,t3);
    cudaEventElapsedTime(&d2h,t3,t4);

    double sum=0; for(const auto&c:host) sum+=std::hypot(c.x,c.y);

    cufftDestroy(plan); cudaFree(d);
    cudaEventDestroy(t0);cudaEventDestroy(t1);cudaEventDestroy(t2);
    cudaEventDestroy(t3);cudaEventDestroy(t4);

    return {sum,h2d,ker,d2h};
}

int main(int argc,char**argv)
{
    int W=(argc>1)?std::atoi(argv[1]):256;
    int H=(argc>2)?std::atoi(argv[2]):256;
    size_t N=(size_t)W*H;
    std::cout<<"Image "<<W<<"×"<<H<<" ("<<N<<" px)\n\n";

    // Generate random image
    std::vector<float> img(N);
    std::mt19937 rng(42); std::uniform_real_distribution<float> dist(0,1);
    for(float &v:img) v=dist(rng);

    auto show=[&](const char*name,const Res&r){
        std::cout<<std::setw(18)<<std::left<<name
                 <<" Σ|X|="<<std::setw(13)<<std::scientific<<r.sum
                 <<std::fixed
                 <<" H2D "<<std::setw(7)<<r.h2d
                 <<" Kern "<<std::setw(8)<<r.ker
                 <<" D2H "<<std::setw(7)<<r.d2h
                 <<" Total "<<r.total()<<" ms\n";
    };

    show("Naïve DFT",        bench_space(img,W,H,false));
    show("Shared DFT",       bench_space(img,W,H,true ));
    show("Radix-2 FFT",      bench_fft  (img,W,H,fft_radix2   ));
    show("Mixed-radix FFT",  bench_fft  (img,W,H,fft_mixed    ));
    show("Split-radix FFT",  bench_fft  (img,W,H,fft_split    ));
    show("Bluestein FFT",    bench_fft  (img,W,H,fft_bluestein));
    show("cuFFT (library)",  bench_cufft(img,W,H));

    return 0;
}