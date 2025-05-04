#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>
#include <vector>
#include <cstdlib>
#include <cmath>

// Helpers
#define CK(x)  do{ cudaError_t e=(x); if(e){                             \
                    printf("CUDA %s:%d %s\n",__FILE__,__LINE__,          \
                           cudaGetErrorString(e)); exit(1);} }while(0)
#define CB(x)  do{ cublasStatus_t s=(x); if(s!=CUBLAS_STATUS_SUCCESS){   \
                    printf("cuBLAS %s:%d status %d\n",__FILE__,__LINE__,s); \
                    exit(1);} }while(0)

// Parameters
constexpr int IMG   = 1024;
constexpr int PATCH = 16;
constexpr int GRID  = IMG / PATCH;               // 64
constexpr int TOK   = GRID * GRID;               // 4096
constexpr int CIN   = 3;
constexpr int PVEC  = CIN * PATCH * PATCH;       // 768

constexpr int EMB   = 256;
constexpr int HEADS = 8;
constexpr int HDIM  = EMB / HEADS;

constexpr int MAX_POINTS = 10;
constexpr int PTOK       = MAX_POINTS + 2;        // 12
constexpr int MTOK       = 2;
constexpr int DTOK       = TOK + PTOK + MTOK;     // 4 110

inline dim3 g1(int n){ return dim3((n+255)/256,1,1); }

// Kernels

// Vector addition
__global__ void add_vec(float* a,const float* b,int n){
    int i = blockIdx.x*256 + threadIdx.x;
    if(i<n) a[i] += b[i];
}
__global__ void layernorm(float* x,int tokens){
    int t=blockIdx.x;   float m=0,v=0;
    for(int d=0; d<EMB; ++d) m += x[t*EMB+d];
    m /= EMB;
    for(int d=0; d<EMB; ++d){
        float z = x[t*EMB+d]-m;
        v += z*z;
    }
    float inv = rsqrtf(v/EMB + 1e-5f);
    for(int d=0; d<EMB; ++d)
        x[t*EMB+d] = (x[t*EMB+d]-m)*inv;
}

// Image → im2col  (4096 × 768)
__global__ void patch2col(
        const float* __restrict__ img,   // [3,1024,1024]
        float*       __restrict__ col)   // [4096,768]
{
    int p = blockIdx.x;   // patch id 0..4095
    int k = threadIdx.x;  // 0..767
    if(k>=PVEC) return;

    int c  = k / (PATCH*PATCH);
    int r  = (k % (PATCH*PATCH)) / PATCH;
    int c2 =  k % PATCH;

    int gy = p / GRID, gx = p % GRID;
    int iy = gy*PATCH + r;
    int ix = gx*PATCH + c2;
    int idx = ((c*IMG)+iy)*IMG + ix;
    col[p*PVEC + k] = img[idx];
}

// Scaled-dot-product attention
__device__ void softmax(float* v,int n){
    float m=-1e20f,s=0.f;
    for(int i=0;i<n;++i) m=fmaxf(m,v[i]);
    for(int i=0;i<n;++i){ v[i]=expf(v[i]-m); s+=v[i]; }
    for(int i=0;i<n;++i) v[i]/=s;
}

__global__ void attn(
        const float* __restrict__ Q,
        const float* __restrict__ K,
        const float* __restrict__ V,
        float*       __restrict__ O,
        int T)
{
    int t = blockIdx.x;          // query token
    int h = blockIdx.y;          // head
    int d = threadIdx.x;         // 0..31
    if(d>=HDIM) return;

    extern __shared__ float buf[];   // length = T
    float* prob = buf;

    if(d==0){
        for(int j=0;j<T;++j){
            float s = 0.f;
            for(int k=0;k<HDIM;++k){
                s += Q[t*EMB + h*HDIM + k] *
                     K[j*EMB + h*HDIM + k];
            }
            prob[j] = s * rsqrtf((float)HDIM);
        }
        softmax(prob,T);
    }
    __syncthreads();

    float acc = 0.f;
    for(int j=0;j<T;++j)
        acc += prob[j] * V[j*EMB + h*HDIM + d];
    O[t*EMB + h*HDIM + d] = acc;
}

// GELU element-wise
__global__ void gelu_kernel(float* x,int n){
    int i = blockIdx.x*256 + threadIdx.x;
    if(i<n){
        float v = x[i];
        x[i] = 0.5f*v*(1.f+tanhf(0.79788456f*(v+0.044715f*v*v*v)));
    }
}

// Prompt encoder
__global__ void point_embed(
        const float* __restrict__ xy,
        const int*   __restrict__ typ,
        const float* __restrict__ Wpos,
        const float* __restrict__ typeTab,
        float*       __restrict__ out)
{
    int t = blockIdx.x, d = threadIdx.x;
    if(d>=EMB) return;
    float x = xy[t*2+0], y = xy[t*2+1];
    out[t*EMB+d] = x*Wpos[d] + y*Wpos[EMB+d] + typeTab[typ[t]*EMB + d];
}

__global__ void mask_downsample(
        const float* __restrict__ in,
        float*       __restrict__ out)        // 64×64 scalar grid
{
    int oy = blockIdx.y, ox = blockIdx.x;
    float s=0.f;
    #pragma unroll
    for(int dy=0;dy<16;++dy)
      for(int dx=0;dx<16;++dx){
          int iy = oy*16 + dy;
          int ix = ox*16 + dx;
          s += in[iy*IMG + ix];
      }
    out[oy*GRID + ox] = s / 256.f;
}

__global__ void mask_project(
        const float* __restrict__ m64,
        const float* __restrict__ W,
        const float* __restrict__ B,
        float*       __restrict__ out)
{
    int cell = blockIdx.x, d = threadIdx.x;
    if(d>=EMB) return;
    out[cell*EMB+d] = m64[cell]*W[d] + B[d];
}

// Mask decoder
__global__ void upsample4(
        const float* __restrict__ in,   // [4096,256]
        float*       __restrict__ out)  // [256×256,256]
{
    int oy = blockIdx.y, ox = blockIdx.x;   // 0..255
    int d  = threadIdx.x;
    if(d>=EMB) return;
    int iy = oy/4, ix = ox/4;
    int src = (iy*GRID + ix)*EMB + d;
    int dst = (oy*256 + ox)*EMB + d;
    out[dst] = in[src];
}

__global__ void proj_mask(
        const float* __restrict__ featUp,
        const float* __restrict__ maskTok,
        float*       __restrict__ maskOut)
{
    int pix = blockIdx.x*256 + threadIdx.x;      // 0..65535
    if(pix >= 256*256) return;
    float acc=0.f;
    for(int d=0; d<EMB; ++d)
        acc += featUp[pix*EMB + d] * maskTok[d];
    maskOut[pix] = acc;
}

int main(){

    // Synthetic host data
    float* hImg;  CK(cudaMallocHost(&hImg, CIN*IMG*IMG*sizeof(float)));
    for(size_t i=0;i<CIN*IMG*IMG;++i)
        hImg[i]=(rand()/float(RAND_MAX))*2.f-1.f;

    std::vector<float> hMask(IMG*IMG);
    for(auto&v:hMask) v = rand()%2;

    std::vector<float> hXY(PTOK*2);
    std::vector<int>   hType(PTOK);
    for(int i=0;i<MAX_POINTS;++i){
        hXY[i*2+0]=rand()/float(RAND_MAX);
        hXY[i*2+1]=rand()/float(RAND_MAX);
        hType[i]=rand()%2;
    }
    hXY[MAX_POINTS*2+0]=0.2f; hXY[MAX_POINTS*2+1]=0.3f; hType[MAX_POINTS]=2;
    hXY[MAX_POINTS*2+2]=0.7f; hXY[MAX_POINTS*2+3]=0.8f; hType[MAX_POINTS+1]=3;

    // Device buffers
    float *dImg,*dCol,*dFeat;
    CK(cudaMalloc(&dImg ,CIN*IMG*IMG*sizeof(float)));
    CK(cudaMalloc(&dCol ,TOK*PVEC*sizeof(float)));
    CK(cudaMalloc(&dFeat,TOK*EMB*sizeof(float)));

    float *dXY,*dMask,*dMask64,*dPTok,*dDense;
    int   *dType;
    CK(cudaMalloc(&dXY ,PTOK*2*sizeof(float)));
    CK(cudaMalloc(&dType,PTOK*sizeof(int)));
    CK(cudaMalloc(&dMask ,IMG*IMG*sizeof(float)));
    CK(cudaMalloc(&dMask64,TOK*sizeof(float)));
    CK(cudaMalloc(&dPTok ,PTOK*EMB*sizeof(float)));
    CK(cudaMalloc(&dDense,TOK*EMB*sizeof(float)));

    float *dSeq,*dQ,*dK,*dV,*dCtx,*dOut;
    CK(cudaMalloc(&dSeq ,DTOK*EMB*sizeof(float)));
    CK(cudaMalloc(&dQ   ,DTOK*EMB*sizeof(float)));
    CK(cudaMalloc(&dK   ,DTOK*EMB*sizeof(float)));
    CK(cudaMalloc(&dV   ,DTOK*EMB*sizeof(float)));
    CK(cudaMalloc(&dCtx ,DTOK*EMB*sizeof(float)));
    CK(cudaMalloc(&dOut ,DTOK*EMB*sizeof(float)));

    float *dFeatUp,*dMask0,*dMask1;
    CK(cudaMalloc(&dFeatUp,256*256*EMB*sizeof(float)));
    CK(cudaMalloc(&dMask0 ,256*256*sizeof(float)));
    CK(cudaMalloc(&dMask1 ,256*256*sizeof(float)));

    // Random weights
    auto rnd=[](){return (rand()/float(RAND_MAX))*0.02f-0.01f;};

    std::vector<float> hWp(PVEC*EMB), hPos(TOK*EMB);
    for(auto&v:hWp)  v=rnd();   for(auto&v:hPos) v=rnd();
    float *dWp,*dPos; CK(cudaMalloc(&dWp ,hWp .size()*sizeof(float)));
    CK(cudaMalloc(&dPos,hPos.size()*sizeof(float)));
    CK(cudaMemcpy(dWp ,hWp .data(),hWp .size()*sizeof(float),cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dPos,hPos.data(),hPos.size()*sizeof(float),cudaMemcpyHostToDevice));

    std::vector<float> hW(EMB*EMB), hB(EMB,0.f),
                       hW1(EMB*4*EMB), hB1(EMB*4,0.f),
                       hW2(EMB*4*EMB), hB2(EMB,0.f);
    for(auto&v:hW)  v=rnd();
    for(auto&v:hW1) v=rnd();
    for(auto&v:hW2) v=rnd();

    float *dW,*dB,*dW1,*dB1,*dW2,*dB2;
    CK(cudaMalloc(&dW ,hW .size()*sizeof(float)));
    CK(cudaMalloc(&dB ,hB .size()*sizeof(float)));
    CK(cudaMalloc(&dW1,hW1.size()*sizeof(float)));
    CK(cudaMalloc(&dB1,hB1.size()*sizeof(float)));
    CK(cudaMalloc(&dW2,hW2.size()*sizeof(float)));
    CK(cudaMalloc(&dB2,hB2.size()*sizeof(float)));
    CK(cudaMemcpy(dW ,hW .data(),hW .size()*sizeof(float),cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dB ,hB .data(),hB .size()*sizeof(float),cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dW1,hW1.data(),hW1.size()*sizeof(float),cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dB1,hB1.data(),hB1.size()*sizeof(float),cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dW2,hW2.data(),hW2.size()*sizeof(float),cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dB2,hB2.data(),hB2.size()*sizeof(float),cudaMemcpyHostToDevice));

    std::vector<float> hPosW(2*EMB),hTypeTab(4*EMB),hWm(EMB),hBm(EMB);
    for(auto&v:hPosW)    v=rnd();
    for(auto&v:hTypeTab) v=rnd();
    for(auto&v:hWm)      v=rnd();
    for(auto&v:hBm)      v=rnd();
    float *dPosW,*dTypeTab,*dWm,*dBm;
    CK(cudaMalloc(&dPosW   ,hPosW   .size()*sizeof(float)));
    CK(cudaMalloc(&dTypeTab,hTypeTab.size()*sizeof(float)));
    CK(cudaMalloc(&dWm,EMB*sizeof(float)));
    CK(cudaMalloc(&dBm,EMB*sizeof(float)));
    CK(cudaMemcpy(dPosW   ,hPosW   .data(),hPosW   .size()*sizeof(float),cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dTypeTab,hTypeTab.data(),hTypeTab.size()*sizeof(float),cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dWm,hWm.data(),EMB*sizeof(float),cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dBm,hBm.data(),EMB*sizeof(float),cudaMemcpyHostToDevice));

    // cuBLAS
    cublasHandle_t hC; CB(cublasCreate(&hC));
    CB(cublasSetPointerMode(hC,CUBLAS_POINTER_MODE_HOST));
    const float alpha=1.f, beta=0.f;

    // Timers
    cudaEvent_t eH0,eH1,eK0,eK1,eD0,eD1;
    for(auto ev:{&eH0,&eH1,&eK0,&eK1,&eD0,&eD1}) cudaEventCreate(ev);

    // Host to device copy
    cudaEventRecord(eH0);
    CK(cudaMemcpyAsync(dImg ,hImg ,CIN*IMG*IMG*sizeof(float),cudaMemcpyHostToDevice));
    CK(cudaMemcpyAsync(dMask,hMask.data(),hMask.size()*sizeof(float),cudaMemcpyHostToDevice));
    CK(cudaMemcpyAsync(dXY  ,hXY  .data(),hXY  .size()*sizeof(float),cudaMemcpyHostToDevice));
    CK(cudaMemcpyAsync(dType,hType.data(),hType.size()*sizeof(int),cudaMemcpyHostToDevice));
    cudaEventRecord(eH1); CK(cudaStreamSynchronize(0));

    // Kernels & GEMMs
    cudaEventRecord(eK0);

    // Image encoder (im2col + GEMM)
    patch2col<<<TOK,PVEC>>>(dImg,dCol);
    CB(cublasSgemm(hC,
        CUBLAS_OP_N,CUBLAS_OP_N,
        EMB, TOK, PVEC,
        &alpha,
        dWp, EMB,
        dCol, PVEC,
        &beta,
        dFeat, EMB));
    add_vec<<<g1(TOK*EMB),256>>>(dFeat,dPos,TOK*EMB);
    layernorm<<<TOK,1>>>(dFeat,TOK);

    // Prompt encoder
    point_embed<<<PTOK,EMB>>>(dXY,dType,dPosW,dTypeTab,dPTok);
    dim3 g64(GRID,GRID);
    mask_downsample<<<g64,1>>>(dMask,dMask64);
    mask_project<<<TOK,EMB>>>(dMask64,dWm,dBm,dDense);

    // Build full sequence  [feat | prompt | 2×0]
    CK(cudaMemcpyAsync(dSeq,dFeat,TOK*EMB*sizeof(float),cudaMemcpyDeviceToDevice));
    CK(cudaMemcpyAsync(dSeq+TOK*EMB,dPTok,PTOK*EMB*sizeof(float),cudaMemcpyDeviceToDevice));
    CK(cudaMemsetAsync(dSeq+(TOK+PTOK)*EMB,0,MTOK*EMB*sizeof(float)));

    layernorm<<<DTOK,1>>>(dSeq,DTOK);

    // Q K V linear projections (GEMM)
    CB(cublasSgemm(hC,CUBLAS_OP_N,CUBLAS_OP_N,
        EMB, DTOK, EMB,&alpha,dW,EMB,dSeq,EMB,&beta,dQ,EMB));
    CB(cublasSgemm(hC,CUBLAS_OP_N,CUBLAS_OP_N,
        EMB, DTOK, EMB,&alpha,dW,EMB,dSeq,EMB,&beta,dK,EMB));
    CB(cublasSgemm(hC,CUBLAS_OP_N,CUBLAS_OP_N,
        EMB, DTOK, EMB,&alpha,dW,EMB,dSeq,EMB,&beta,dV,EMB));

    // Attention
    attn<<<dim3(DTOK,HEADS),HDIM,DTOK*sizeof(float)>>>(dQ,dK,dV,dCtx,DTOK);
    add_vec<<<g1(DTOK*EMB),256>>>(dCtx,dSeq,DTOK*EMB);

    // MLP = Ctx·W1 -> GELU -> ·W2
    CB(cublasSgemm(hC,CUBLAS_OP_N,CUBLAS_OP_N,
        EMB*4, DTOK, EMB,&alpha,dW1,EMB*4,dCtx,EMB,&beta,dQ,EMB*4));
    add_vec<<<g1(DTOK*EMB*4),256>>>(dQ,dB1,DTOK*EMB*4);
    gelu_kernel<<<g1(DTOK*EMB*4),256>>>(dQ,DTOK*EMB*4);
    CB(cublasSgemm(hC,CUBLAS_OP_N,CUBLAS_OP_N,
        EMB, DTOK, EMB*4,&alpha,dW2,EMB,dQ,EMB*4,&beta,dOut,EMB));
    add_vec<<<g1(DTOK*EMB),256>>>(dOut,dCtx,DTOK*EMB);

    // Mask decoder
    float* dMaskTok0 = dOut + (DTOK-MTOK)*EMB;
    float* dMaskTok1 = dMaskTok0 + EMB;

    layernorm<<<TOK,1>>>(dFeat,TOK);
    dim3 gr(256,256); upsample4<<<gr,EMB>>>(dFeat,dFeatUp);

    proj_mask<<<g1(256*256),256>>>(dFeatUp,dMaskTok0,dMask0);
    proj_mask<<<g1(256*256),256>>>(dFeatUp,dMaskTok1,dMask1);

    cudaEventRecord(eK1); CK(cudaStreamSynchronize(0));

    // Device to host copy
    std::vector<float> hMask0(256*256),hMask1(256*256);
    cudaEventRecord(eD0);
    CK(cudaMemcpyAsync(hMask0.data(),dMask0,hMask0.size()*sizeof(float),cudaMemcpyDeviceToHost));
    CK(cudaMemcpyAsync(hMask1.data(),dMask1,hMask1.size()*sizeof(float),cudaMemcpyDeviceToHost));
    cudaEventRecord(eD1); CK(cudaStreamSynchronize(0));

    // Timings
    float msH,msK,msD,msT;
    cudaEventElapsedTime(&msH,eH0,eH1);
    cudaEventElapsedTime(&msK,eK0,eK1);
    cudaEventElapsedTime(&msD,eD0,eD1);
    cudaEventElapsedTime(&msT,eH0,eD1);

    printf("GPU timings (ms):\n Host-to-Device copy time:      :  %.2f ms\n Kernel execution time:         :  %.2f ms\n Device-to-Host copy time:      :  %.2f ms\n Total GPU time:                :  %.2f ms\n",
    msH,msK,msD,msT);

    cublasDestroy(hC);
    return 0;
}
