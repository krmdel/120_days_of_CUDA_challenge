#include <cuda_runtime.h>

#include <vector>
#include <random>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <cstdlib>

#define MAX_TAPS 1024
#define THREADS   256
#define TILE      128            // Outputs per block for shared-mem kernel

// Helpers
#define CUDA_CHECK(x)  do{ cudaError_t rc=(x); if(rc!=cudaSuccess){            \
    std::cerr<<"CUDA "<<cudaGetErrorString(rc)<<" @"<<__FILE__<<":"<<__LINE__;\
    std::exit(EXIT_FAILURE);} }while(0)

// Taps in constant memory
__constant__ float d_taps[MAX_TAPS];

// Naïve kernel: one thread -> one decimated output
__global__ void fir_decim_naive(const float* __restrict__ x,
                                float*       __restrict__ y,
                                int N,int T,int D,int OUT)
{
    int m = blockIdx.x*blockDim.x + threadIdx.x;
    if(m >= OUT) return;
    int n = m * D;

    float acc=0.f;
    #pragma unroll 4
    for(int k=0;k<T;++k){
        int idx = n - k;
        if(idx>=0) acc += d_taps[k]*x[idx];
    }
    y[m]=acc;
}

// Shared-memory tiled kernel
template<int TILE_OUT>
__global__ void fir_decim_tiled(const float* __restrict__ x,
                                float*       __restrict__ y,
                                int N,int T,int D,int OUT)
{
    extern __shared__ float s_in[];                // TILE*D + T -1 samples
    const int m0   = blockIdx.x * TILE_OUT;        // First out index this block
    const int n0   = m0 * D;
    const int halo = T-1;
    const int span = TILE_OUT*D + halo;            // Samples per block

    // Co-operative load
    for(int off=threadIdx.x; off<span; off+=TILE_OUT){
        int g = n0 - halo + off;
        s_in[off] = (g>=0 && g<N) ? x[g] : 0.f;
    }
    __syncthreads();

    int m = m0 + threadIdx.x;
    if(m>=OUT) return;

    int local_idx = threadIdx.x*D + halo;
    float acc=0.f;
    #pragma unroll 8
    for(int k=0;k<T;++k)
        acc += d_taps[k]*s_in[local_idx-k];
    y[m]=acc;
}

// Timing struct
struct Times{ float h2d,ker,d2h; float total()const{return h2d+ker+d2h;} };

// Run + time helper
Times run_kernel(const float* d_x,float* d_y,
                 int N,int T,int D,int OUT,
                 void (*launch)(const float*,float*,int,int,int,int,int),
                 size_t shBytes=0)
{
    cudaEvent_t e0,e1,e2,e3; for(auto& e:{&e0,&e1,&e2,&e3}) cudaEventCreate(e);
    cudaEventRecord(e0);
    launch(d_x,d_y,N,T,D,OUT,(int)shBytes);
    cudaEventRecord(e1);

    std::vector<float> h_tmp(OUT);
    CUDA_CHECK(cudaMemcpyAsync(h_tmp.data(),d_y,sizeof(float)*OUT,
                               cudaMemcpyDeviceToHost));
    cudaEventRecord(e2);
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEventRecord(e3);

    float ker_ms,d2h_ms;
    cudaEventElapsedTime(&ker_ms,e0,e1);
    cudaEventElapsedTime(&d2h_ms,e1,e2);
    for(auto e:{e0,e1,e2,e3}) cudaEventDestroy(e);
    return {0.f,ker_ms,d2h_ms};
}

int main(int argc,char**argv)
{
    const int N = (argc>1)?std::atoi(argv[1]):1'048'576;
    const int T = (argc>2)?std::atoi(argv[2]):63;
    const int D = (argc>3)?std::atoi(argv[3]):4;
    if(T>MAX_TAPS){ std::cerr<<"T > MAX_TAPS\n"; return 1; }

    const int OUT = (N + D - 1) / D;
    std::cout<<"Input signal lenght: "<<N<<",  taps: "<<T<<",  decim: "<<D
             <<",  output: "<<OUT<<"\n\n";

    // Host
    std::vector<float> h_x(N), h_h(T);
    std::mt19937 rng(0); std::uniform_real_distribution<float> dist(-1,1);
    for(float& v:h_x) v=dist(rng);
    for(float& v:h_h) v=dist(rng);

    // Device
    float *d_x,*d_y_naive,*d_y_tile;
    CUDA_CHECK(cudaMalloc(&d_x,sizeof(float)*N));
    CUDA_CHECK(cudaMalloc(&d_y_naive,sizeof(float)*OUT));
    CUDA_CHECK(cudaMalloc(&d_y_tile ,sizeof(float)*OUT));

    CUDA_CHECK(cudaMemcpyToSymbol(d_taps,h_h.data(),sizeof(float)*T));

    // Single H2D copy for input
    cudaEvent_t eh0,eh1; cudaEventCreate(&eh0); cudaEventCreate(&eh1);
    cudaEventRecord(eh0);
    CUDA_CHECK(cudaMemcpy(d_x,h_x.data(),sizeof(float)*N,cudaMemcpyHostToDevice));
    cudaEventRecord(eh1); CUDA_CHECK(cudaDeviceSynchronize());
    float h2d_ms; cudaEventElapsedTime(&h2d_ms,eh0,eh1);
    cudaEventDestroy(eh0); cudaEventDestroy(eh1);

    // Naïve engine
    auto launch_naive = [](const float* x,float* y,int N,int T,int D,int OUT,int){
        dim3 grid((OUT+THREADS-1)/THREADS), block(THREADS);
        fir_decim_naive<<<grid,block>>>(x,y,N,T,D,OUT);
    };
    Times t_naive = run_kernel(d_x,d_y_naive,N,T,D,OUT,launch_naive);

    // Tiled engine
    constexpr int SH_TILE = TILE;
    size_t shBytes = (SH_TILE*D + T - 1u)*sizeof(float);
    auto launch_tile = [](const float* x,float* y,int N,int T,int D,int OUT,int bytes){
        dim3 grid((OUT+SH_TILE-1)/SH_TILE), block(SH_TILE);
        fir_decim_tiled<SH_TILE><<<grid,block,bytes>>>(x,y,N,T,D,OUT);
    };
    Times t_tile  = run_kernel(d_x,d_y_tile ,N,T,D,OUT,launch_tile,shBytes);

    t_naive.h2d = t_tile.h2d = h2d_ms;

    auto show=[&](const char* n,const Times& t){
        std::cout<<std::setw(18)<<std::left<<n
                 <<" H2D "<<std::setw(7)<<t.h2d
                 <<" Kern "<<std::setw(9)<<t.ker
                 <<" D2H "<<std::setw(7)<<t.d2h
                 <<" Total "<<t.total()<<" ms\n";
    };
    std::cout<<std::fixed<<std::setprecision(3);
    show("Naïve decim:",       t_naive);
    show("Shared-mem decim:",  t_tile );

    cudaFree(d_x); cudaFree(d_y_naive); cudaFree(d_y_tile);
    return 0;
}