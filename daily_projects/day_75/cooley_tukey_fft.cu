#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>

#ifndef SIZE
#define SIZE 16384
#endif
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// CUDA helpers
#define CUDA_CHECK(x) \
    do { cudaError_t rc=(x); if(rc!=cudaSuccess){                             \
        std::fprintf(stderr,"CUDA %s @ %s:%d\n", cudaGetErrorString(rc),      \
                     __FILE__,__LINE__); std::exit(EXIT_FAILURE);} } while(0)

__device__ __forceinline__ float2 operator+(float2 a,float2 b){
    return make_float2(a.x+b.x , a.y+b.y);
}
__device__ __forceinline__ float2 operator-(float2 a,float2 b){
    return make_float2(a.x-b.x , a.y-b.y);
}
__device__ __forceinline__ float2 operator*(float2 a,float2 b){
    return make_float2(a.x*b.x - a.y*b.y , a.x*b.y + a.y*b.x);
}
__device__ __forceinline__ float2 W(int k,int N){
    float s,c; __sincosf(-2.f*M_PI*k/N,&s,&c);
    return make_float2(c,s);
}

// Bit-reverse (shared)
__global__ void bitrev(float2* d,int n,int log2n){
    int tid=blockIdx.x*blockDim.x+threadIdx.x;
    if(tid>=n) return;
    unsigned r=__brev(tid)>>(32-log2n);
    if(r>tid){ float2 t=d[tid]; d[tid]=d[r]; d[r]=t; }
}

// Engine 0  : iterative radix-2 DIF
__global__ void radix2_stage(float2* d,int n,int half){
    int tid = blockIdx.x*blockDim.x+threadIdx.x;
    int span = half*2;
    if(tid>=n/2) return;
    int j   = tid % half;
    int base= (tid/half)*span;
    int p   = base + j;
    float2 u=d[p];
    float2 v=d[p+half]*W(j,span);
    d[p]       = u+v;
    d[p+half]  = u-v;
}
void engine_radix2(float2* d,int n){
    int lg=static_cast<int>(log2f(n)), th=256;
    int bl=(n+th-1)/th;
    bitrev<<<bl,th>>>(d,n,lg);
    for(int s=1;s<=lg;++s){
        int half=1<<(s-1);
        int work=n>>1;
        bl=(work+th-1)/th;
        radix2_stage<<<bl,th>>>(d,n,half);
    }
}

// Engine 1  : mixed-radix 2/3/5
__device__ void bf3(float2&a,float2&b,float2&c){
    const float2 w1=make_float2(-0.5f,-0.8660254039f);
    const float2 w2=make_float2(-0.5f, 0.8660254039f);
    float2 t1=a+b+c, t2=a+b*w1+c*w2, t3=a+b*w2+c*w1;
    a=t1; b=t2; c=t3;
}
__device__ void bf5(float2&a,float2&b,float2&c,float2&d,float2&e){
    const float tau=0.3090169944f, sin72=0.9510565163f;
    float2 t1=b+e, t2=b-e, t3=c+d, t4=c-d;
    float2 t5=a+t1+t3;
    float2 t6=a+make_float2(-0.25f,0)*t1 + make_float2(-0.25f,0)*t3;
    float2 t7=make_float2(0, tau)*t2  + make_float2(0,-tau)*t4;
    float2 t8=make_float2(-sin72,0)*t2 + make_float2(sin72,0)*t4;
    b=t6+t7; e=t6-t7; c=t6+t8; d=t6-t8; a=t5;
}
__global__ void mixed_kernel(float2* d,int n,int radix,int stage){
    int tid=blockIdx.x*blockDim.x+threadIdx.x;
    int grp=tid/stage, idx=tid%stage;
    if(radix==3){
        int base=grp*stage*3+idx;
        if(base+2*stage<n){
            float2 a=d[base],b=d[base+stage],c=d[base+2*stage];
            bf3(a,b,c);
            d[base]=a; d[base+stage]=b; d[base+2*stage]=c;
        }
    }else if(radix==5){
        int base=grp*stage*5+idx;
        if(base+4*stage<n){
            float2 a=d[base],b=d[base+stage],c=d[base+2*stage],
                   f=d[base+3*stage],e=d[base+4*stage];
            bf5(a,b,c,f,e);
            d[base]=a; d[base+stage]=b; d[base+2*stage]=c;
            d[base+3*stage]=f; d[base+4*stage]=e;
        }
    }
}
void engine_mixed(float2* d,int n){
    int rem=n, stage=1, th=256, bl;
    while(rem%5==0){
        bl=(n/(stage*5)+th-1)/th;
        mixed_kernel<<<bl,th>>>(d,n,5,stage);
        stage*=5; rem/=5;
    }
    while(rem%3==0){
        bl=(n/(stage*3)+th-1)/th;
        mixed_kernel<<<bl,th>>>(d,n,3,stage);
        stage*=3; rem/=3;
    }
    if(rem>1) engine_radix2(d,n);
}

// Engine 2  : split-radix (radix-2 + twiddle save)
__global__ void split_post(float2* d,int N){
    int k = blockIdx.x*blockDim.x+threadIdx.x;
    if(k>=N/4) return;
    float2 a = d[2*k];
    float2 b = d[2*k+1];
    float2 c = d[k+N/2];
    float2 e = d[k+N/2+1];
    float2 t1 = b + e;
    float2 t2 = b - e;
    d[2*k]           = a + c;
    d[2*k+1]         = t1;
    d[k+N/2]         = (a - c) * W(k, N);
    d[k+N/2+1]       = t2      * W(k, N) * make_float2(0,1);
}
void engine_splitradix(float2* d,int N){
    // Step 1 : radix-2 baseline
    engine_radix2(d,N);

    // Step 2 : twiddle-saving shuffle on lowest stage
    int th=256, bl=(N/4 + th-1)/th;
    split_post<<<bl,th>>>(d,N);
}

// Engine 3  : four-step Stockham
__global__ void stockham(float2* src,float2* dst,int N,int L,int m){
    int tid=blockIdx.x*blockDim.x+threadIdx.x;
    if(tid>=N/2) return;
    int i=tid/m, j=tid%m;
    int p=i*L+j;
    float2 a=src[p], b=src[p+m], tw=W(i,L*2), t=b*tw;
    dst[i*2*m + j]     = a+t;
    dst[i*2*m + j + m] = a-t;
}
void engine_fourstep(float2* d,int N){
    float2 *ping=d,*pong; CUDA_CHECK(cudaMalloc(&pong,sizeof(float2)*N));
    int stages=static_cast<int>(log2f(N)), th=256;
    for(int s=0;s<stages;++s){
        int m=1<<s, L=N>>(s+1);
        int bl=((N>>1)+th-1)/th;
        stockham<<<bl,th>>>(ping,pong,N,L,m);
        std::swap(ping,pong);
    }
    if(ping!=d) CUDA_CHECK(cudaMemcpy(d,ping,sizeof(float2)*N,cudaMemcpyDeviceToDevice));
    cudaFree(pong);
}

// Engine 4  : six-step (transpose)
__global__ void transpose(float2* dst,const float2* src,int rows,int cols){
    __shared__ float2 tile[32][33];
    int x=blockIdx.x*32+threadIdx.x, y=blockIdx.y*32+threadIdx.y;
    if(x<cols && y<rows) tile[threadIdx.y][threadIdx.x]=src[y*cols+x];
    __syncthreads();
    x=blockIdx.y*32+threadIdx.x; y=blockIdx.x*32+threadIdx.y;
    if(x<rows && y<cols) dst[y*rows+x]=tile[threadIdx.x][threadIdx.y];
}
void engine_sixstep(float2* d,int N){
    const int n1=32, n2=N/n1;
    float2* tmp; CUDA_CHECK(cudaMalloc(&tmp,sizeof(float2)*N));

    for(int r=0;r<n2;++r) engine_radix2(d+r*n1,n1);

    dim3 g((n1+31)/32,(n2+31)/32), b(32,32);
    transpose<<<g,b>>>(tmp,d,n2,n1);
    for(int c=0;c<n1;++c) engine_radix2(tmp+c*n2,n2);
    transpose<<<g,b>>>(d,tmp,n1,n2);
    cudaFree(tmp);
}

// Time helper
float bench(int id,const std::vector<float2>& h,int N){
    float2* d; CUDA_CHECK(cudaMalloc(&d,sizeof(float2)*N));
    CUDA_CHECK(cudaMemcpy(d,h.data(),sizeof(float2)*N,cudaMemcpyHostToDevice));
    cudaEvent_t s,e; cudaEventCreate(&s); cudaEventCreate(&e);
    cudaEventRecord(s);

    switch(id){
        case 0: engine_radix2    (d,N); break;
        case 1: engine_mixed     (d,N); break;
        case 2: engine_splitradix(d,N); break;
        case 3: engine_fourstep  (d,N); break;
        case 4: engine_sixstep   (d,N); break;
    }

    cudaEventRecord(e); cudaEventSynchronize(e);
    float ms; cudaEventElapsedTime(&ms,s,e);
    cudaEventDestroy(s); cudaEventDestroy(e); cudaFree(d);
    return ms;
}

int main(){
    const int N=SIZE;
    std::cout<<"Signal length: "<<N<<"\n";

    std::vector<float2> sig(N);
    for(int i=0;i<N;++i){ sig[i].x=std::sin(2*M_PI*i/64); sig[i].y=0; }

    const char* name[5]={
        "Iterative radix-2",
        "Mixed-radix 2/3/5",
        "Split-radix",
        "Four-step Stockham",
        "Six-step"
    };
    for(int id=0;id<5;++id){
        float t=bench(id,sig,N);
        std::cout<<"Engine "<<id<<" ("<<name[id]<<") : "<<t<<" ms\n";
    }
    return 0;
}