#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <cstdlib>
#define CK(x) do{ cudaError_t e=(x); if(e){                                 \
                 printf("CUDA %s:%d %s\n",__FILE__,__LINE__,                \
                        cudaGetErrorString(e)); exit(1);} }while(0)

// Parameters
constexpr int GRID  = 64;                      // 1024/16
constexpr int PATCH = 256/GRID;                // upsample factor = 4
constexpr int FTOK  = GRID*GRID;               // 4 096
constexpr int EMB   = 256;
constexpr int MTOK  = 2;                       // #mask tokens

// Helpers
inline dim3 g1(int n){ return dim3((n+255)/256,1,1); }

// Layer-norm on feature tokens
__global__ void layernorm_feat(float* f){
    int p=blockIdx.x; float m=0,v=0;
    for(int d=0;d<EMB;++d) m+=f[p*EMB+d];  m/=EMB;
    for(int d=0;d<EMB;++d){ float z=f[p*EMB+d]-m; v+=z*z; }
    float inv=rsqrtf(v/EMB+1e-5f);
    for(int d=0;d<EMB;++d) f[p*EMB+d]=(f[p*EMB+d]-m)*inv;
}

// 4× nearest-neighbor upsample 64×64×C → 256×256×C
__global__ void upsample4(
        const float* __restrict__ in,   // [4096,256]
        float*       __restrict__ out)  // [65536,256]
{
    int oy = blockIdx.y, ox = blockIdx.x;   // 0..255
    int d  = threadIdx.x;                   // channel
    if(d>=EMB) return;
    int iy = oy / PATCH;                    // source 0..63
    int ix = ox / PATCH;
    int src = (iy*GRID + ix)*EMB + d;
    int dst = (oy*256 + ox)*EMB + d;
    out[dst] = in[src];
}

// Dot-product mask projection: one kernel per mask-token
__global__ void proj_mask(
        const float* __restrict__ up,      // [65536,256]
        const float* __restrict__ mTok,    // 256-d
        float*       __restrict__ mask)    // [65536]
{
    int pix = blockIdx.x*256 + threadIdx.x; if(pix>=256*256) return;
    float acc=0;
    for(int d=0; d<EMB; ++d)
        acc += up[pix*EMB+d]*mTok[d];
    mask[pix] = acc;
}

int main(){
    // Synthetic inputs
    std::vector<float> hFeat(FTOK*EMB);
    std::vector<float> hTok (MTOK*EMB);     // take 2 mask-tokens
    for(float&v:hFeat) v=(rand()/float(RAND_MAX))*0.1f-0.05f;
    for(float&v:hTok)  v=(rand()/float(RAND_MAX))*0.1f-0.05f;

    // Device buffers
    float *dF,*dTok; CK(cudaMalloc(&dF,hFeat.size()*sizeof(float)));
    CK(cudaMalloc(&dTok,hTok.size()*sizeof(float)));

    float *dFeatUp;   // 256×256×256
    CK(cudaMalloc(&dFeatUp,256*256*EMB*sizeof(float)));

    float *dMask0,*dMask1;
    CK(cudaMalloc(&dMask0,256*256*sizeof(float)));
    CK(cudaMalloc(&dMask1,256*256*sizeof(float)));

    // Events
    cudaEvent_t H0,H1,K0,K1,D0,D1;
    for(auto ev:{&H0,&H1,&K0,&K1,&D0,&D1}) cudaEventCreate(ev);

    // Host to device copy
    cudaEventRecord(H0);
    CK(cudaMemcpyAsync(dF  ,hFeat.data(),hFeat.size()*sizeof(float),cudaMemcpyHostToDevice));
    CK(cudaMemcpyAsync(dTok,hTok .data(),hTok .size()*sizeof(float),cudaMemcpyHostToDevice));
    cudaEventRecord(H1); CK(cudaStreamSynchronize(0));

    // Kernels
    cudaEventRecord(K0);

    layernorm_feat<<<FTOK,1>>>(dF);

    dim3 upsGrid(256,256);
    upsample4<<<upsGrid,EMB>>>(dF,dFeatUp);

    proj_mask<<<g1(256*256),256>>>(dFeatUp,dTok+0*EMB,dMask0);
    proj_mask<<<g1(256*256),256>>>(dFeatUp,dTok+1*EMB,dMask1);

    cudaEventRecord(K1); CK(cudaStreamSynchronize(0));

    // Device to host copy
    std::vector<float> hMask0(256*256), hMask1(256*256);
    cudaEventRecord(D0);
    CK(cudaMemcpyAsync(hMask0.data(),dMask0,hMask0.size()*sizeof(float),cudaMemcpyDeviceToHost));
    CK(cudaMemcpyAsync(hMask1.data(),dMask1,hMask1.size()*sizeof(float),cudaMemcpyDeviceToHost));
    cudaEventRecord(D1); CK(cudaStreamSynchronize(0));

    // Timing output
    float msH,msK,msD,msT;
    cudaEventElapsedTime(&msH,H0,H1);
    cudaEventElapsedTime(&msK,K0,K1);
    cudaEventElapsedTime(&msD,D0,D1);
    cudaEventElapsedTime(&msT,H0,D1);

    printf("GPU timings (ms):\n Host-to-Device copy time:      :  %.2f ms\n Kernel execution time:         :  %.2f ms\n Device-to-Host copy time:      :  %.2f ms\n Total GPU time:                :  %.2f ms\n",
    msH,msK,msD,msT);

    return 0;
}
