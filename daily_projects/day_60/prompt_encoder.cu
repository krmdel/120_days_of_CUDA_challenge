
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <cstdlib>
#define CK(x) do{ cudaError_t e=(x); if(e){                                 \
                   printf("CUDA %s:%d %s\n",__FILE__,__LINE__,              \
                          cudaGetErrorString(e)); exit(1);} }while(0)

// Parameters 
constexpr int IMG  = 1024;
constexpr int GRID = 64;                          // 1024/16
constexpr int EMB  = 256;

constexpr int MAX_POINTS = 10;                    // synthetic
constexpr int TOKENS     = MAX_POINTS + 2;        // + box corners

// Helpers
inline dim3 g1(int n){ return dim3((n+255)/256,1,1); }

// Kernels

// Generate sparse token embeddings
__global__ void point_box_embed(
        const float* __restrict__ xy,         // [TOKENS,2]
        const int*   __restrict__ typ,        // token type id
        const float* __restrict__ Wpos,       // [2,256]
        const float* __restrict__ typeTab,    // [4,256]
        float*       __restrict__ out)        // [TOKENS,256]
{
    int t = blockIdx.x;       // token idx
    int d = threadIdx.x;      // embed dim
    if(d>=EMB) return;
    float x = xy[t*2+0];
    float y = xy[t*2+1];
    float e = x*Wpos[d] + y*Wpos[EMB+d] + typeTab[typ[t]*EMB + d];
    out[t*EMB+d] = e;
}

//   Down-sample 1024² mask → 64² (average 16×16 blocks)
__global__ void mask_downsample(
        const float* __restrict__ in,   // [1024,1024]
        float*       __restrict__ out)  // [64,64]
{
    int oy = blockIdx.y, ox = blockIdx.x;
    float sum = 0.f;
    for(int dy=0; dy<16; ++dy)
      for(int dx=0; dx<16; ++dx){
          int iy = oy*16 + dy;
          int ix = ox*16 + dx;
          sum += in[iy*IMG + ix];
      }
    out[oy*GRID + ox] = sum / 256.f;
}

//   Project scalar mask cell → 256-d vector
__global__ void mask_project(
        const float* __restrict__ m64,   // [4096]
        const float* __restrict__ Wm,    // [256]
        const float* __restrict__ Bm,    // [256]
        float*       __restrict__ out)   // [4096,256]
{
    int cell = blockIdx.x;
    int d    = threadIdx.x;
    if(d>=EMB) return;
    float v = m64[cell] * Wm[d] + Bm[d];
    out[cell*EMB + d] = v;
}

int main(){
    // Synthetic host inputs
    // points (random coords [0,1], label 0=BG 1=FG)
    std::vector<float> hPts(MAX_POINTS*2);
    std::vector<int>   hLab(MAX_POINTS);
    for(int i=0;i<MAX_POINTS;++i){
        hPts[i*2+0] = rand()/float(RAND_MAX);      // x
        hPts[i*2+1] = rand()/float(RAND_MAX);      // y
        hLab[i]     = rand()%2;                    // BG/FG
    }
    // one box (TL + BR)
    float x0 = 0.2f, y0 = 0.3f, x1 = 0.7f, y1 = 0.8f;

    // mask 1024² random binary
    std::vector<float> hMask(IMG*IMG);
    for(auto&v:hMask) v = (rand()%2);

    // Arrange token arrays
    std::vector<float> hTokXY(TOKENS*2);
    std::vector<int>   hTokType(TOKENS);

    // points first
    for(int i=0;i<MAX_POINTS;++i){
        hTokXY[i*2+0]=hPts[i*2+0];
        hTokXY[i*2+1]=hPts[i*2+1];
        hTokType[i]  = hLab[i];               // 0=BG,1=FG
    }
    // Box corners
    hTokXY[MAX_POINTS*2+0] = x0; hTokXY[MAX_POINTS*2+1]=y0;
    hTokType[MAX_POINTS]   = 2;               // TL corner id=2
    hTokXY[MAX_POINTS*2+2] = x1; hTokXY[MAX_POINTS*2+3]=y1;
    hTokType[MAX_POINTS+1] = 3;               // BR corner id=3

    // Device buffers
    float *dTokXY,*dMask,*dPosW,*dTypeTab,*dTokens,*dM64,*dDense;
    int   *dTokType;
    CK(cudaMalloc(&dTokXY, TOKENS*2*sizeof(float)));
    CK(cudaMalloc(&dTokType, TOKENS*sizeof(int)));
    CK(cudaMalloc(&dMask, IMG*IMG*sizeof(float)));

    // Learnable weights
    CK(cudaMalloc(&dPosW , 2*EMB*sizeof(float)));        // [2,256]
    CK(cudaMalloc(&dTypeTab, 4*EMB*sizeof(float)));      // 4 token types
    CK(cudaMalloc(&dM64 , GRID*GRID*sizeof(float)));
    CK(cudaMalloc(&dDense, GRID*GRID*EMB*sizeof(float))); // mask embedding

    std::vector<float> hPosW(2*EMB), hTypeTab(4*EMB), hWm(EMB), hBm(EMB);
    auto rnd=[](){return (rand()/float(RAND_MAX))*0.02f-0.01f;};
    for(float&v:hPosW)    v=rnd();
    for(float&v:hTypeTab) v=rnd();
    for(float&v:hWm)      v=rnd();
    for(float&v:hBm)      v=rnd();
    float *dWm,*dBm;
    CK(cudaMalloc(&dWm,EMB*sizeof(float)));
    CK(cudaMalloc(&dBm,EMB*sizeof(float)));
    CK(cudaMemcpy(dPosW   ,hPosW   .data(),hPosW.size()   *sizeof(float),cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dTypeTab,hTypeTab.data(),hTypeTab.size()*sizeof(float),cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dWm,hWm.data(),EMB*sizeof(float),cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dBm,hBm.data(),EMB*sizeof(float),cudaMemcpyHostToDevice));

    // Tokens out
    CK(cudaMalloc(&dTokens, TOKENS*EMB*sizeof(float)));

    // Events
    cudaEvent_t h2d0,h2d1,k0,k1,d2h0,d2h1;
    for(auto ev:{&h2d0,&h2d1,&k0,&k1,&d2h0,&d2h1}) cudaEventCreate(ev);

    // Host to device copy
    cudaEventRecord(h2d0);
    CK(cudaMemcpyAsync(dTokXY, hTokXY.data(), hTokXY.size()*sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMemcpyAsync(dTokType, hTokType.data(), hTokType.size()*sizeof(int), cudaMemcpyHostToDevice));
    CK(cudaMemcpyAsync(dMask, hMask.data(), hMask.size()*sizeof(float), cudaMemcpyHostToDevice));
    cudaEventRecord(h2d1);
    CK(cudaStreamSynchronize(0));

    // Kernels
    cudaEventRecord(k0);

    // Sparse embeddings
    point_box_embed<<<TOKENS,EMB>>>(
        dTokXY,dTokType,dPosW,dTypeTab,dTokens);

    // Dense mask embedding
    dim3 gs(GRID,GRID);
    mask_downsample<<<gs,1>>>(dMask,dM64);
    mask_project<<<GRID*GRID,EMB>>>(dM64,dWm,dBm,dDense);

    cudaEventRecord(k1);
    CK(cudaStreamSynchronize(0));

    // Devide to host copy
    float peekToken, peekDense;
    cudaEventRecord(d2h0);
    CK(cudaMemcpyAsync(&peekToken,dTokens,sizeof(float),cudaMemcpyDeviceToHost));
    CK(cudaMemcpyAsync(&peekDense,dDense+EMB, sizeof(float),cudaMemcpyDeviceToHost));
    cudaEventRecord(d2h1);
    CK(cudaStreamSynchronize(0));

    // Timings
    float tH2D,tKer,tD2H,tTot;
    CK(cudaEventElapsedTime(&tH2D,h2d0,h2d1));
    CK(cudaEventElapsedTime(&tKer,k0,k1));
    CK(cudaEventElapsedTime(&tD2H,d2h0,d2h1));
    CK(cudaEventElapsedTime(&tTot,h2d0,d2h1));

    printf("GPU timings (ms):\n Host-to-Device copy time:      :  %.2f ms\n Kernel execution time:         :  %.2f ms\n Device-to-Host copy time:      :  %.2f ms\n Total GPU time:                :  %.2f ms\n",
        tH2D,tKer,tD2H,tTot);

    return 0;
}

// Kernel bodies
__global__ void patch_embed(const float*,const float*,const float*,float*){}
__global__ void layernorm(float*,int){}
__global__ void add_vec(float*,const float*,int){}
__global__ void linear(const float*,const float*,const float*,float*,int,int,int){}
__global__ void attn(const float*,const float*,const float*,float*,int){}
__global__ void mlp(const float*,const float*,const float*,const float*,const float*,float*,int){}
