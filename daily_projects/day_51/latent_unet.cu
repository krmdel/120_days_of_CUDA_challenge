#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cmath>

#define cudaCheck(e) \
  do{ cudaError_t _e=(e); if(_e!=cudaSuccess){ \
       std::cerr<<"CUDA "<<cudaGetErrorString(_e)<<" @ "<<__FILE__<<":"<<__LINE__<<"\n"; \
       std::exit(EXIT_FAILURE);} }while(0)

// Parameters
constexpr int H   = 64;
constexpr int W   = 64;
constexpr int CIN = 4;
constexpr int CM1 = 8;   // channels after first conv
constexpr int COUT= 4;   // output channels
constexpr int K   = 3;

constexpr int PIX_IN   = H*W*CIN;
constexpr int PIX_MID  = H*W*CM1;
constexpr int PIX_OUT  = H*W*COUT;

constexpr int TILE = 16;                 // 16×16 spatial tiles

// CPU functions

// Indexing helpers
inline float idx(const float* t,int c,int y,int x,int C,int H,int W)
{ return t[(c*H + y)*W + x]; }

// Set value at index
void set_idx(float* t,int c,int y,int x,int C,int H,int W,float v)
{ t[(c*H + y)*W + x]=v; }

// 3 × 3 convolution, padding = 1, stride = 1
void conv3x3_cpu(const float* in, float* out,
                 const float* w, int IC,int OC)
{
    for(int oc=0; oc<OC; ++oc)
      for(int y=0; y<H; ++y)
        for(int x=0; x<W; ++x)
        {
            float s=0.f;
            for(int ic=0; ic<IC; ++ic)
              for(int ky=-1; ky<=1; ++ky)
                for(int kx=-1; kx<=1; ++kx)
                {
                    int yy=y+ky, xx=x+kx;
                    if(yy<0||yy>=H||xx<0||xx>=W) continue;
                    float v = idx(in,ic,yy,xx,IC,H,W);
                    float wv = w[((oc*IC+ic)*K + (ky+1))*K + (kx+1)];
                    s += v*wv;
                }
            set_idx(out,oc,y,x,OC,H,W,s);
        }
}

void relu_cpu(float* t,int CH){ for(int i=0;i<CH*H*W;++i) t[i]=fmaxf(t[i],0.f); }

void add_residual_cpu(const float* x,float* y) // y += x (COUT==CIN==4)
{ for(int i=0;i<PIX_OUT;++i) y[i]+=x[i]; }


// CUDA kernels

__constant__ float d_weights1[CM1*CIN*K*K];
__constant__ float d_weights2[COUT*CM1*K*K];

// shared‑memory tiled conv ‑ each blk processes one output channel slice
template<int IC,int OC>
__global__ void conv3x3_kernel(const float* __restrict__ in,
                               float*       __restrict__ out)
{
    __shared__ float tile[TILE+2][TILE+2][IC];

    const int oc = blockIdx.z;            // output channel handled by block
    const int by = blockIdx.y*TILE;
    const int bx = blockIdx.x*TILE;
    const int ty = threadIdx.y;
    const int tx = threadIdx.x;

    // load input patch
    for(int ic=0; ic<IC; ++ic){
        int iy = by + ty - 1;
        int ix = bx + tx - 1;
        float v = 0.f;
        if(iy>=0 && iy<H && ix>=0 && ix<W)
            v = in[(ic*H + iy)*W + ix];
        tile[ty][tx][ic] = v;
    }
    __syncthreads();

    // compute only for valid central 16×16 threads
    if(ty < TILE && tx < TILE){
        int y = by + ty;
        int x = bx + tx;
        if(y<H && x<W){
            float sum=0.f;
            for(int ic=0; ic<IC; ++ic){
                #pragma unroll
                for(int ky=0; ky<3; ++ky)
                  #pragma unroll
                  for(int kx=0; kx<3; ++kx)
                {
                    float v = tile[ty+ky][tx+kx][ic];
                    float w = (OC==CM1? d_weights1 : d_weights2)
                              [((oc*IC+ic)*K + ky)*K + kx];
                    sum += v*w;
                }
            }
            out[(oc*H + y)*W + x] = sum;
        }
    }
}

__global__ void relu_kernel(float* t,int CH)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int N = CH*H*W;
    if(i<N) t[i]=fmaxf(t[i],0.f);
}

__global__ void add_residual_kernel(const float* x,float* y) // y += x
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i<PIX_OUT) y[i]+=x[i];
}

int main()
{
    const size_t B_IN  = PIX_IN  *sizeof(float);
    const size_t B_MID = PIX_MID *sizeof(float);
    const size_t B_OUT = PIX_OUT *sizeof(float);
    const size_t B_W1  = CM1*CIN*K*K*sizeof(float);
    const size_t B_W2  = COUT*CM1*K*K*sizeof(float);

    // Host memory allocation
    float *h_x ,*h_mid,*h_y_cpu,*h_y_gpu;
    float *h_w1,*h_w2;
    cudaCheck(cudaHostAlloc(&h_x ,  B_IN ,cudaHostAllocDefault));
    cudaCheck(cudaHostAlloc(&h_mid,B_MID,cudaHostAllocDefault));
    cudaCheck(cudaHostAlloc(&h_y_cpu,B_OUT,cudaHostAllocDefault));
    cudaCheck(cudaHostAlloc(&h_y_gpu,B_OUT,cudaHostAllocDefault));
    h_w1 = new float[CM1*CIN*K*K];
    h_w2 = new float[COUT*CM1*K*K];

    std::mt19937 rng(123);
    std::normal_distribution<float> g(0.f,0.05f);
    std::uniform_real_distribution<float> u(-1.f,1.f);
    for(int i=0;i<PIX_IN; ++i)  h_x[i]=u(rng);
    for(int i=0;i<CM1*CIN*K*K;++i) h_w1[i]=g(rng);
    for(int i=0;i<COUT*CM1*K*K;++i) h_w2[i]=g(rng);

    // Device memory allocation
    float *d_x,*d_mid,*d_y;
    cudaCheck(cudaMalloc(&d_x  ,B_IN ));
    cudaCheck(cudaMalloc(&d_mid,B_MID));
    cudaCheck(cudaMalloc(&d_y  ,B_OUT));

    // Copy weights to constant memory once
    cudaCheck(cudaMemcpyToSymbol(d_weights1,h_w1,B_W1));
    cudaCheck(cudaMemcpyToSymbol(d_weights2,h_w2,B_W2));

    // Streams & events
    cudaStream_t s_cpy,s_cmp; cudaCheck(cudaStreamCreate(&s_cpy));
    cudaCheck(cudaStreamCreate(&s_cmp));

    cudaEvent_t ev0,ev_h2d,ev_k,ev_d2h,ev_end;
    cudaCheck(cudaEventCreate(&ev0));  cudaCheck(cudaEventCreate(&ev_h2d));
    cudaCheck(cudaEventCreate(&ev_k)); cudaCheck(cudaEventCreate(&ev_d2h));
    cudaCheck(cudaEventCreate(&ev_end));

    // CPU inference
    auto cpu0=std::chrono::high_resolution_clock::now();
    conv3x3_cpu(h_x,h_mid, h_w1, CIN, CM1);
    relu_cpu(h_mid,CM1);
    conv3x3_cpu(h_mid,h_y_cpu, h_w2, CM1, COUT);
    add_residual_cpu(h_x,h_y_cpu);            // residual add
    auto cpu1=std::chrono::high_resolution_clock::now();
    double cpu_ms=std::chrono::duration<double,std::milli>(cpu1-cpu0).count();

    // GPU inference
    cudaEventRecord(ev0);

    cudaMemcpyAsync(d_x,h_x,B_IN,cudaMemcpyHostToDevice,s_cpy);
    cudaEventRecord(ev_h2d,s_cpy);

    // Conv1
    dim3 grid((W+TILE-1)/TILE,(H+TILE-1)/TILE,CM1);
    dim3 block(TILE+2,TILE+2);                // extra halo cols/rows
    conv3x3_kernel<CIN,CM1><<<grid,block,0,s_cmp>>>(d_x,d_mid);
    // ReLU
    relu_kernel<<<(PIX_MID+255)/256,256,0,s_cmp>>>(d_mid,CM1);
    // Conv2
    dim3 grid2((W+TILE-1)/TILE,(H+TILE-1)/TILE,COUT);
    conv3x3_kernel<CM1,COUT><<<grid2,block,0,s_cmp>>>(d_mid,d_y);
    // Residual add
    add_residual_kernel<<<(PIX_OUT+255)/256,256,0,s_cmp>>>(d_x,d_y);

    cudaEventRecord(ev_k,s_cmp);

    cudaMemcpyAsync(h_y_gpu,d_y,B_OUT,cudaMemcpyDeviceToHost,s_cpy);
    cudaEventRecord(ev_d2h,s_cpy);

    cudaEventRecord(ev_end);
    cudaCheck(cudaEventSynchronize(ev_end));

    // Timings
    float ms_h2d,ms_k,ms_d2h,ms_tot;
    cudaEventElapsedTime(&ms_h2d,ev0,ev_h2d);
    cudaEventElapsedTime(&ms_k,  ev_h2d,ev_k);
    cudaEventElapsedTime(&ms_d2h,ev_k,ev_d2h);
    cudaEventElapsedTime(&ms_tot,ev0,ev_end);

    std::cout<<"CPU inference time (ms) : "<<cpu_ms <<" ms\n";
    std::cout<<"GPU timings (ms):\n"
    <<"  Host-to-Device copy time:      : "<<ms_h2d<<"\n"
    <<"  Kernel execution time:         : "<<ms_k<<"\n"
    <<"  Device-to-Host copy time:      : "<<ms_d2h<<"\n"
    <<"  Total GPU time:                : "<<ms_tot<<"\n";

    // Cleanup
    cudaFree(d_x);   cudaFree(d_mid); cudaFree(d_y);
    cudaFreeHost(h_x); cudaFreeHost(h_mid); cudaFreeHost(h_y_cpu); cudaFreeHost(h_y_gpu);
    delete[] h_w1;   delete[] h_w2;
    cudaStreamDestroy(s_cpy); cudaStreamDestroy(s_cmp);
    cudaEventDestroy(ev0); cudaEventDestroy(ev_h2d); cudaEventDestroy(ev_k);
    cudaEventDestroy(ev_d2h); cudaEventDestroy(ev_end);
}
