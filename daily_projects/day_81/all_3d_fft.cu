#include <cuda_runtime.h>
#include <cufft.h>
#include <cuComplex.h>
#include <device_launch_parameters.h>

#include <vector>
#include <random>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <cstdlib>
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Error helpers
#define CUDA_CHECK(x)  do{ cudaError_t rc=(x); if(rc!=cudaSuccess){            \
    std::cerr<<"CUDA "<<cudaGetErrorString(rc)<<" @"<<__FILE__<<":"<<__LINE__;\
    std::exit(EXIT_FAILURE);} }while(0)
#define CUFFT_CHECK(x) do{ cufftResult rc=(x); if(rc!=CUFFT_SUCCESS){          \
    std::cerr<<"cuFFT "<<rc<<" @"<<__FILE__<<":"<<__LINE__<<"\n";              \
    std::exit(EXIT_FAILURE);} }while(0)

// float2 helpers
__host__ __device__ inline float2 f2(float r,float i){ return make_float2(r,i); }
__host__ __device__ inline float2 operator+(float2 a,float2 b){ return f2(a.x+b.x,a.y+b.y); }
__host__ __device__ inline float2 operator-(float2 a,float2 b){ return f2(a.x-b.x,a.y-b.y); }
__host__ __device__ inline float2 operator*(float2 a,float2 b){
    return f2(a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x);
}
__host__ __device__ inline float2& operator+=(float2& a,const float2& b)
{ a.x+=b.x; a.y+=b.y; return a; }
__host__ __device__ inline float2 cexpj(float th)
{ float s,c; sincosf(th,&s,&c); return f2(c,s); }

// Device helpers
__device__ __forceinline__ float2 cMul(float2 a,float2 b){ return a*b; }
__device__ __forceinline__ float2 Wnk(int k,int N){ return cexpj(-2.f*M_PI*k/N); }

// 32×32 transpose
__global__ void transpose32(float2* dst,const float2* src,int rows,int cols)
{
    __shared__ float2 tile[32][33];
    int x=blockIdx.x*32+threadIdx.x,
        y=blockIdx.y*32+threadIdx.y;
    if(x<cols && y<rows) tile[threadIdx.y][threadIdx.x]=src[y*cols+x];
    __syncthreads();
    x=blockIdx.y*32+threadIdx.x;
    y=blockIdx.x*32+threadIdx.y;
    if(x<rows && y<cols) dst[y*rows+x]=tile[threadIdx.x][threadIdx.y];
}

// Radix-2 FFT kernels
__global__ void bitrev(float2* d,int n,int lg)
{
    int tid=blockIdx.x*blockDim.x+threadIdx.x; if(tid>=n) return;
    unsigned r=__brev(tid)>>(32-lg); if(r>tid){ float2 t=d[tid]; d[tid]=d[r]; d[r]=t; }
}
__global__ void radix2_stage(float2* d,int n,int half)
{
    int tid=blockIdx.x*blockDim.x+threadIdx.x; if(tid>=n/2) return;
    int span=half*2, j=tid%half, base=(tid/half)*span;
    float2 u=d[base+j], v=d[base+j+half]*Wnk(j,span);
    d[base+j]       =u+v;
    d[base+j+half]  =u-v;
}
void fft_radix2(float2* d,int n)
{
    int th=256, lg=static_cast<int>(log2f(n));
    bitrev<<<(n+th-1)/th,th>>>(d,n,lg);
    for(int s=1;s<=lg;++s){
        int half=1<<(s-1);
        radix2_stage<<<((n>>1)+th-1)/th,th>>>(d,n,half);
    }
}

// Mixed-radix 2/3/5
__device__ void bf3(float2&a,float2&b,float2&c){
    const float2 w1=f2(-0.5f,-0.8660254039f), w2=f2(-0.5f,0.8660254039f);
    float2 t1=a+b+c;
    float2 t2=a+b*w1+c*w2;
    float2 t3=a+b*w2+c*w1;
    a=t1; b=t2; c=t3;
}
__device__ void bf5(float2&a,float2&b,float2&c,float2&d,float2&e){
    const float tau=0.3090169944f, s72=0.9510565163f;
    float2 t1=b+e, t2=b-e, t3=c+d, t4=c-d;
    float2 t5=a+t1+t3;
    float2 t6=a+f2(-0.25f,0)*(t1+t3);
    float2 t7=f2(0,tau)*t2+f2(0,-tau)*t4;
    float2 t8=f2(-s72,0)*t2+f2(s72,0)*t4;
    b=t6+t7; e=t6-t7; c=t6+t8; d=t6-t8; a=t5;
}
__global__ void mixed_kernel(float2* d,int n,int radix,int span)
{
    int tid=blockIdx.x*blockDim.x+threadIdx.x;
    int grp=tid/span, idx=tid%span;
    if(radix==3){
        int base=grp*span*3+idx; if(base+2*span<n){
            float2 a=d[base],b=d[base+span],c=d[base+2*span]; bf3(a,b,c);
            d[base]=a; d[base+span]=b; d[base+2*span]=c;
        }
    }else{                                              // radix-5
        int base=grp*span*5+idx; if(base+4*span<n){
            float2 a=d[base],b=d[base+span],c=d[base+2*span],
                   f=d[base+3*span],e=d[base+4*span];
            bf5(a,b,c,f,e);
            d[base]=a; d[base+span]=b; d[base+2*span]=c;
            d[base+3*span]=f; d[base+4*span]=e;
        }
    }
}
void fft_mixed(float2* d,int n)
{
    int rem=n, span=1, th=256;
    while(rem%5==0){
        mixed_kernel<<<(n/(span*5)+th-1)/th,th>>>(d,n,5,span);
        span*=5; rem/=5;
    }
    while(rem%3==0){
        mixed_kernel<<<(n/(span*3)+th-1)/th,th>>>(d,n,3,span);
        span*=3; rem/=3;
    }
    if(rem>1) fft_radix2(d,n);
}

// Split-radix post stage
__global__ void split_post(float2* d,int N)
{
    int k=blockIdx.x*blockDim.x+threadIdx.x; if(k>=N/4) return;
    float2 a=d[2*k], b=d[2*k+1], c=d[k+N/2], e=d[k+N/2+1];
    float2 t1=b+e, t2=b-e;
    d[2*k]      =a+c;
    d[2*k+1]    =t1;
    d[k+N/2]    =(a-c)*Wnk(k,N);
    d[k+N/2+1]  =t2*Wnk(k,N)*f2(0,1);
}
void fft_split(float2* d,int N)
{
    fft_radix2(d,N);
    split_post<<<(N/4+255)/256,256>>>(d,N);
}

// Bluestein / Chirp-Z (1-D)
__global__ void blu_build(const float2* x,float2* A,float2* B,int N,float k)
{
    int n=blockIdx.x*blockDim.x+threadIdx.x; if(n<N){
        float2 w=cexpj(k*n*n);
        A[n]=x[n]*w; B[n]=cexpj(-k*n*n);
    }
}
__global__ void blu_pad(float2* B,int N,int M){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=M) return;
    if(i>=N && i<M-N+1) B[i]=f2(0,0);
    if(i>0 && i<N)      B[M-i]=B[i];
}
__global__ void blu_mul(const float2* A,const float2* B,float2* C,int M){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<M) C[i]=A[i]*B[i];
}
__global__ void blu_final(const float2* y,float2* X,int N,int M,float k){
    int n=blockIdx.x*blockDim.x+threadIdx.x; if(n<N){
        float2 v=y[n]; v.x/=M; v.y/=M; X[n]=v*cexpj(-k*n*n);
    }
}
int next_pow2(int v){ int p=1; while(p<v) p<<=1; return p; }
void fft_bluestein(float2* d,int N)
{
    const int M=next_pow2(2*N-1); float k=M_PI/N;
    float2 *dA,*dB,*dC; CUDA_CHECK(cudaMalloc(&dA,sizeof(float2)*M));
    CUDA_CHECK(cudaMalloc(&dB,sizeof(float2)*M));
    CUDA_CHECK(cudaMalloc(&dC,sizeof(float2)*M));
    CUDA_CHECK(cudaMemset(dA,0,sizeof(float2)*M));
    CUDA_CHECK(cudaMemset(dB,0,sizeof(float2)*M));

    dim3 B(256);
    blu_build<<<(N+255)/256,B>>>(d,dA,dB,N,k);
    blu_pad  <<<(M+255)/256,B>>>(dB,N,M);

    cufftHandle plan; CUFFT_CHECK(cufftPlan1d(&plan,M,CUFFT_C2C,1));
    CUFFT_CHECK(cufftExecC2C(plan,(cufftComplex*)dA,(cufftComplex*)dA,CUFFT_FORWARD));
    CUFFT_CHECK(cufftExecC2C(plan,(cufftComplex*)dB,(cufftComplex*)dB,CUFFT_FORWARD));
    blu_mul  <<<(M+255)/256,B>>>(dA,dB,dC,M);
    CUFFT_CHECK(cufftExecC2C(plan,(cufftComplex*)dC,(cufftComplex*)dC,CUFFT_INVERSE));
    blu_final<<<(N+255)/256,B>>>(dC,d,N,M,k);

    cufftDestroy(plan); cudaFree(dA); cudaFree(dB); cudaFree(dC);
}

// 3-D separable helper
#define IDX3(x,y,z)  ( (size_t(z)*H + y)*W + x )

using FFT1D = void(*)(float2*,int);

static void fft3d_generic(float2* vol,float2* scratch,int W,int H,int D,FFT1D eng)
{
    // X-axis
    for(int z=0; z<D; ++z){
        for(int y=0; y<H; ++y){
            eng(vol+IDX3(0,y,z), W);
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Y-axis
    for(int z=0; z<D; ++z)
        for(int x=0; x<W; ++x){
            for(int y=0; y<H; ++y) scratch[y]=vol[IDX3(x,y,z)];
            eng(scratch,H);
            CUDA_CHECK(cudaDeviceSynchronize());
            for(int y=0; y<H; ++y) vol[IDX3(x,y,z)]=scratch[y];
        }

    // Z-axis
    for(int y=0; y<H; ++y)
        for(int x=0; x<W; ++x){
            for(int z=0; z<D; ++z) scratch[z]=vol[IDX3(x,y,z)];
            eng(scratch,D);
            CUDA_CHECK(cudaDeviceSynchronize());
            for(int z=0; z<D; ++z) vol[IDX3(x,y,z)]=scratch[z];
        }
}

// Device helper for space kernels
__device__ __forceinline__ void rot_inc(float& c,float& s,float cd,float sd){
    float t=c*cd - s*sd; s=c*sd + s*cd; c=t;
}
constexpr int TILE = 16;

// Naive slice
__global__ void dft2d_naive_kernel(const float* src,float2* dst,int W,int H)
{
    int u=blockIdx.x*blockDim.x+threadIdx.x;
    int v=blockIdx.y*blockDim.y+threadIdx.y;
    if(u>=W||v>=H) return;
    float re=0,im=0;
    for(int y=0;y<H;++y)
        for(int x=0;x<W;++x){
            float ang=-2.f*M_PI*(u* x /(float)W + v*y/(float)H);
            float s,c; __sincosf(ang,&s,&c);
            float val=src[y*W+x];
            re+=val*c; im+=val*s;
        }
    dst[v*W+u]=f2(re,im);
}

// Tiled slice
__global__ void dft2d_tiled_slice(const float* src,float2* dst,int W,int H)
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
            for(int ty=0; ty<TILE && y0+ty<H; ++ty){
                float sxCol=sx, cxCol=cx;
                #pragma unroll
                for(int tx=0; tx<TILE && x0+tx<W; ++tx){
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
    dst[v*W+u]=f2(accR,accI);
}

static void run_space_slice(const float* d_img,float2* d_out,
                            int W,int H,int D,bool tiled)
{
    dim3 B(16,16), G((W+15)/16,(H+15)/16);
    for(int z=0; z<D; ++z){
        const float* src=d_img + size_t(z)*W*H;
        float2*      dst=d_out + size_t(z)*W*H;
        if(tiled) dft2d_tiled_slice<<<G,B>>>(src,dst,W,H);
        else      dft2d_naive_kernel <<<G,B>>>(src,dst,W,H);
    }
}

// Result struct
struct Res{ double sum; float h2d,ker,d2h; float total()const{return h2d+ker+d2h;} };

// Space-domain benchmark
static Res bench_space3(const std::vector<float>& vol,int W,int H,int D,bool tiled)
{
    size_t N=size_t(W)*H*D; float *d_in; float2* d_out;
    CUDA_CHECK(cudaMalloc(&d_in ,sizeof(float)*N));
    CUDA_CHECK(cudaMalloc(&d_out,sizeof(float2)*N));

    cudaEvent_t t0,t1,t2,t3; cudaEventCreate(&t0);cudaEventCreate(&t1);
    cudaEventCreate(&t2);cudaEventCreate(&t3);

    cudaEventRecord(t0);
    CUDA_CHECK(cudaMemcpy(d_in,vol.data(),sizeof(float)*N,cudaMemcpyHostToDevice));
    cudaEventRecord(t1);

    run_space_slice(d_in,d_out,W,H,D,tiled);
    cudaEventRecord(t2);

    std::vector<float2> host(N);
    CUDA_CHECK(cudaMemcpy(host.data(),d_out,sizeof(float2)*N,cudaMemcpyDeviceToHost));
    cudaEventRecord(t3); cudaDeviceSynchronize();

    float h2d,ker,d2h; cudaEventElapsedTime(&h2d,t0,t1);
    cudaEventElapsedTime(&ker,t1,t2); cudaEventElapsedTime(&d2h,t2,t3);

    double sum=0; for(auto& c:host) sum+=std::hypot(c.x,c.y);

    cudaFree(d_in); cudaFree(d_out);
    cudaEventDestroy(t0);cudaEventDestroy(t1);cudaEventDestroy(t2);cudaEventDestroy(t3);
    return {sum,h2d,ker,d2h};
}

// Separable-FFT benchmark  (Unified Memory + D2H timing)
static Res bench_fft3(const std::vector<float>& vol,int W,int H,int D,FFT1D eng)
{
    const size_t N=size_t(W)*H*D;
    float2* d_vol;    CUDA_CHECK(cudaMallocManaged(&d_vol, sizeof(float2)*N));
    int maxDim = std::max({W,H,D});
    float2* scratch;  CUDA_CHECK(cudaMallocManaged(&scratch, sizeof(float2)*maxDim));

    for(size_t i=0;i<N;++i) d_vol[i]=f2(vol[i],0);

    cudaEvent_t t0,t1,t2,t3; cudaEventCreate(&t0);cudaEventCreate(&t1);
    cudaEventCreate(&t2);cudaEventCreate(&t3);

    cudaEventRecord(t0);
    CUDA_CHECK(cudaMemPrefetchAsync(d_vol ,sizeof(float2)*N, 0));
    CUDA_CHECK(cudaMemPrefetchAsync(scratch,sizeof(float2)*maxDim,0));
    cudaEventRecord(t1);

    fft3d_generic(d_vol,scratch,W,H,D,eng);
    cudaEventRecord(t2);

    CUDA_CHECK(cudaMemPrefetchAsync(d_vol ,sizeof(float2)*N, cudaCpuDeviceId));
    cudaEventRecord(t3);  cudaDeviceSynchronize();

    double sum=0; for(size_t i=0;i<N;++i) sum+=std::hypot(d_vol[i].x,d_vol[i].y);

    float h2d,ker,d2h;
    cudaEventElapsedTime(&h2d,t0,t1);
    cudaEventElapsedTime(&ker,t1,t2);
    cudaEventElapsedTime(&d2h,t2,t3);

    cudaFree(d_vol); cudaFree(scratch);
    cudaEventDestroy(t0);cudaEventDestroy(t1);cudaEventDestroy(t2);cudaEventDestroy(t3);
    return {sum,h2d,ker,d2h};
}

// cuFFT 3-D benchmark
static Res bench_cufft3(const std::vector<float>& vol,int W,int H,int D)
{
    size_t N=size_t(W)*H*D;
    std::vector<cufftComplex> host(N);
    for(size_t i=0;i<N;++i) host[i]=make_cuFloatComplex(vol[i],0);

    cufftComplex* d; CUDA_CHECK(cudaMalloc(&d,sizeof(cufftComplex)*N));
    cudaEvent_t t0,t1,t2,t3; cudaEventCreate(&t0);cudaEventCreate(&t1);
    cudaEventCreate(&t2);cudaEventCreate(&t3);

    cudaEventRecord(t0);
    CUDA_CHECK(cudaMemcpy(d,host.data(),sizeof(cufftComplex)*N,cudaMemcpyHostToDevice));
    cudaEventRecord(t1);

    cufftHandle plan; CUFFT_CHECK(cufftPlan3d(&plan,D,H,W,CUFFT_C2C));
    cudaEventRecord(t2);
    CUFFT_CHECK(cufftExecC2C(plan,d,d,CUFFT_FORWARD));
    cudaEventRecord(t3); CUFFT_CHECK(cufftDestroy(plan));

    CUDA_CHECK(cudaMemcpy(host.data(),d,sizeof(cufftComplex)*N,cudaMemcpyDeviceToHost));
    cudaDeviceSynchronize();

    float h2d,ker,d2h;
    cudaEventElapsedTime(&h2d,t0,t1);
    cudaEventElapsedTime(&ker,t2,t3);
    cudaEventElapsedTime(&d2h,t3,t3);
    double sum=0; for(auto& c:host) sum+=std::hypot(c.x,c.y);

    cudaFree(d);
    cudaEventDestroy(t0);cudaEventDestroy(t1);cudaEventDestroy(t2);cudaEventDestroy(t3);
    return {sum,h2d,ker,d2h};
}

int main(int argc,char**argv)
{
    int W=(argc>1)?std::atoi(argv[1]):256;
    int H=(argc>2)?std::atoi(argv[2]):256;
    int D=(argc>3)?std::atoi(argv[3]):32;
    size_t N=size_t(W)*H*D;
    std::cout<<"Volume "<<W<<"×"<<H<<"×"<<D<<" ("<<N<<" voxels)\n\n";

    std::vector<float> vol(N);
    std::mt19937 rng(42); std::uniform_real_distribution<float> dist(0,1);
    for(float& v:vol) v=dist(rng);

    auto prn=[&](const char* n,const Res& r){
        std::cout<<std::setw(18)<<std::left<<n
                 <<" Σ|X|="<<std::setw(13)<<std::scientific<<r.sum
                 <<std::fixed
                 <<" H2D "<<std::setw(8)<<r.h2d
                 <<" Kern "<<std::setw(9)<<r.ker
                 <<" D2H "<<std::setw(8)<<r.d2h
                 <<" Total "<<r.total()<<" ms\n";
    };

    prn("Naïve DFT",        bench_space3(vol,W,H,D,false));
    prn("Shared DFT",       bench_space3(vol,W,H,D,true ));
    prn("Radix-2 FFT",      bench_fft3  (vol,W,H,D,fft_radix2   ));
    prn("Mixed-radix FFT",  bench_fft3  (vol,W,H,D,fft_mixed    ));
    prn("Split-radix FFT",  bench_fft3  (vol,W,H,D,fft_split    ));
    prn("Bluestein FFT",    bench_fft3  (vol,W,H,D,fft_bluestein));
    prn("cuFFT 3-D",        bench_cufft3(vol,W,H,D));

    return 0;
}