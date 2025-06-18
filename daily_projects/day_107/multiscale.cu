#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <numeric>
#include <cmath>
#include <chrono>

#define CUDA_CHECK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){                 \
    printf("CUDA error %s (%s:%d)\n",cudaGetErrorString(e),__FILE__,__LINE__);   \
    std::exit(1);} }while(0)

template<class TP> inline double ms(TP a,TP b){
    return std::chrono::duration<double,std::milli>(b-a).count(); }

// CPU reference (memset)
void cpuFill(std::vector<uint8_t>& img,uint8_t v){
    std::fill(img.begin(),img.end(),v);
}

// Fill kernel
__global__ void fillKernel(uint8_t* p,size_t pitch,int W,int H,uint8_t v){
    int x=blockIdx.x*32 + threadIdx.x;
    int y=blockIdx.y*8  + threadIdx.y;
    if(x<W && y<H) p[y*pitch + x] = v;
}

int main(){
    const int W0 = 4096, H0 = 4096;                // largest level
    const int S  = 4;                              // scales
    // Host pyramids
    std::vector<int> Ws(S), Hs(S);
    Ws[0]=W0; Hs[0]=H0;
    for(int i=1;i<S;++i){ Ws[i]=Ws[i-1]/2; Hs[i]=Hs[i-1]/2; }

    std::vector<std::vector<uint8_t>> h_in(S), h_out(S);
    for(int i=0;i<S;++i){
        h_in[i].assign(size_t(Ws[i])*Hs[i], uint8_t(10+i));
        h_out[i].resize(h_in[i].size());
    }

    // CPU reference
    auto c0 = std::chrono::high_resolution_clock::now();
    for(auto& v:h_out) cpuFill(v,42);
    auto c1 = std::chrono::high_resolution_clock::now();
    double cpu_ms = ms(c0,c1);

    // Memory pool
    cudaMemPool_t pool;
    CUDA_CHECK(cudaDeviceGetDefaultMemPool(&pool,0));
    size_t threshold = size_t(W0)*H0;              // ≥ largest buffer
    CUDA_CHECK(cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold,
                                       &threshold));
    CUDA_CHECK(cudaDeviceSetMemPool(0,pool));

    cudaStream_t stream; CUDA_CHECK(cudaStreamCreate(&stream));

    // Timing containers
    std::vector<double> allocVec, h2dVec, kerVec, d2hVec, totVec;

    cudaEvent_t eLoopBeg,eLoopEnd; CUDA_CHECK(cudaEventCreate(&eLoopBeg));
    CUDA_CHECK(cudaEventCreate(&eLoopEnd));

    for(int run=0; run<100; ++run){
        CUDA_CHECK(cudaEventRecord(eLoopBeg,stream));

        double tAlloc=0,tH2D=0,tKer=0,tD2H=0;

        for(int s=0;s<S;++s){
            size_t bytes = size_t(Ws[s])*Hs[s];

            // Declare events and create them one-by-one
            cudaEvent_t a0,a1,h0,h1,k0,k1,d0,d1;
            CUDA_CHECK(cudaEventCreate(&a0)); CUDA_CHECK(cudaEventCreate(&a1));
            CUDA_CHECK(cudaEventCreate(&h0)); CUDA_CHECK(cudaEventCreate(&h1));
            CUDA_CHECK(cudaEventCreate(&k0)); CUDA_CHECK(cudaEventCreate(&k1));
            CUDA_CHECK(cudaEventCreate(&d0)); CUDA_CHECK(cudaEventCreate(&d1));

            // mallocAsync
            uint8_t* d; size_t pitch = Ws[s];          // contiguous
            CUDA_CHECK(cudaEventRecord(a0,stream));
            CUDA_CHECK(cudaMallocAsync((void**)&d,bytes,stream));
            CUDA_CHECK(cudaEventRecord(a1,stream));

            // H2D
            CUDA_CHECK(cudaEventRecord(h0,stream));
            CUDA_CHECK(cudaMemcpyAsync(d,h_in[s].data(),bytes,
                                       cudaMemcpyHostToDevice,stream));
            CUDA_CHECK(cudaEventRecord(h1,stream));

            // Kernel
            dim3 blk(32,8), grd((Ws[s]+31)/32,(Hs[s]+7)/8);
            CUDA_CHECK(cudaEventRecord(k0,stream));
            fillKernel<<<grd,blk,0,stream>>>(d,pitch,Ws[s],Hs[s],42);
            CUDA_CHECK(cudaEventRecord(k1,stream));

            // D2H
            CUDA_CHECK(cudaEventRecord(d0,stream));
            CUDA_CHECK(cudaMemcpyAsync(h_out[s].data(),d,bytes,
                                       cudaMemcpyDeviceToHost,stream));
            CUDA_CHECK(cudaEventRecord(d1,stream));

            // Free
            CUDA_CHECK(cudaFreeAsync(d,stream));

            // Sync & accumulate
            CUDA_CHECK(cudaStreamSynchronize(stream));
            float t;
            CUDA_CHECK(cudaEventElapsedTime(&t,a0,a1)); tAlloc+=t;
            CUDA_CHECK(cudaEventElapsedTime(&t,h0,h1)); tH2D +=t;
            CUDA_CHECK(cudaEventElapsedTime(&t,k0,k1)); tKer +=t;
            CUDA_CHECK(cudaEventElapsedTime(&t,d0,d1)); tD2H +=t;

            for(auto ev:{a0,a1,h0,h1,k0,k1,d0,d1}) CUDA_CHECK(cudaEventDestroy(ev));
        }

        CUDA_CHECK(cudaEventRecord(eLoopEnd,stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        float loop_ms; CUDA_CHECK(cudaEventElapsedTime(&loop_ms,eLoopBeg,eLoopEnd));

        allocVec.push_back(tAlloc);
        h2dVec .push_back(tH2D);
        kerVec .push_back(tKer);
        d2hVec .push_back(tD2H);
        totVec .push_back(loop_ms);
    }

    // Stats after warm-up
    auto stats=[&](const std::vector<double>& v){
        double m=0,s=0; int n=v.size()-1;
        for(size_t i=1;i<v.size();++i) m+=v[i]; m/=n;
        for(size_t i=1;i<v.size();++i){ double d=v[i]-m; s+=d*d; }
        return std::pair<double,double>(m,std::sqrt(s/n));
    };
    auto [aMean,aStd] = stats(allocVec);
    auto [hMean,_ ]   = stats(h2dVec);
    auto [kMean,__]   = stats(kerVec);
    auto [dMean,___]  = stats(d2hVec);
    auto [tMean,____] = stats(totVec);

    printf("Runs (cold+warm)            : %zu\n",totVec.size());
    printf("Alloc  mean ± σ  (warm)     : %.4f ± %.4f ms  (<- expect <0.05)\n",aMean,aStd);
    printf("H2D   mean                  : %.3f ms\n",hMean);
    printf("Kernel mean                 : %.3f ms\n",kMean);
    printf("D2H   mean                  : %.3f ms\n",dMean);
    printf("GPU total mean              : %.3f ms\n",tMean);
    printf("CPU reference               : %.3f ms\n",cpu_ms);

    CUDA_CHECK(cudaStreamDestroy(stream));
    return 0;
}