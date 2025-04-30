#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <cmath>

#define CK(stmt)  do{ cudaError_t e=(stmt);                        \
        if(e){ printf("CUDA %s:%d %s\n",__FILE__,__LINE__,         \
                 cudaGetErrorString(e)); exit(1);} }while(0)

// Model sizes
constexpr int IMG   = 1024;              // input side
constexpr int PATCH = 16;                // ViT-H/16 patch
constexpr int GRID  = IMG / PATCH;       // 64
constexpr int TOK   = GRID * GRID;       // 4096 tokens
constexpr int CIN   = 3;
constexpr int PVEC  = CIN * PATCH * PATCH;   // 768
constexpr int EMB   = 256;               // token dim
constexpr int HEADS = 8;
constexpr int HDIM  = EMB / HEADS;       // 32

// Helpers
inline dim3 g1(int n){ return dim3((n+255)/256,1,1); }

// Kernels
__global__ void patch_embed(
        const float* __restrict__ img,   // [3,1024,1024]
        const float* __restrict__ Wp,    // [768,256]
        const float* __restrict__ pos,   // [4096,256]
        float*       __restrict__ tok)   // [4096,256]
{
    int p = blockIdx.x;                  // patch index
    int d = threadIdx.x;                 // embed dim
    if(d >= EMB) return;
    int gy = p / GRID, gx = p % GRID;

    float acc = 0.f;
    for(int c=0;c<CIN;++c)
      for(int py=0; py<PATCH; ++py)
        for(int px=0; px<PATCH; ++px){
            int iy = gy*PATCH + py;
            int ix = gx*PATCH + px;
            int imgIdx = ((c*IMG)+iy)*IMG + ix;
            int k = ((c*PATCH+py)*PATCH + px);
            acc += img[imgIdx] * Wp[k*EMB+d];
        }
    tok[p*EMB+d] = acc + pos[p*EMB+d];
}

// Addition
__global__ void add_vec(float* a,const float* b,int n){
    int i = blockIdx.x*256 + threadIdx.x;
    if(i<n) a[i] += b[i];
}

// Mean-std layer-norm
__global__ void layernorm(float* x,int tokens){
    int t = blockIdx.x;
    float m=0,v=0;
    for(int d=0; d<EMB; ++d) m += x[t*EMB+d];
    m /= EMB;
    for(int d=0; d<EMB; ++d){
        float z = x[t*EMB+d]-m; v += z*z;
    }
    float inv = rsqrtf(v/EMB + 1e-5f);
    for(int d=0; d<EMB; ++d)
        x[t*EMB+d] = (x[t*EMB+d]-m)*inv;
}

// Linear
__global__ void linear(
        const float* __restrict__ in,
        const float* __restrict__ W,
        const float* __restrict__ B,
        float*       __restrict__ out,
        int T,int OD,int ID)
{
    int t = blockIdx.x, d = threadIdx.x;
    if(d>=OD) return;
    float acc = B?B[d]:0.f;
    for(int k=0;k<ID;++k) acc += in[t*ID+k]*W[k*OD+d];
    out[t*OD+d] = acc;
}

// Softmax
__device__ void softmax(float* v,int n){
    float m=-1e20f,s=0;
    for(int i=0;i<n;++i) m = fmaxf(m,v[i]);
    for(int i=0;i<n;++i){ v[i]=expf(v[i]-m); s+=v[i]; }
    for(int i=0;i<n;++i) v[i]/=s;
}

// Scaled dot-prod attention (one head per block.y)
__global__ void attn(
        const float* __restrict__ Q,
        const float* __restrict__ K,
        const float* __restrict__ V,
        float*       __restrict__ O,
        int T)
{
    int t = blockIdx.x;      // query token
    int h = blockIdx.y;      // head
    int d = threadIdx.x;     // dim in head
    if(d >= HDIM) return;

    extern __shared__ float buf[];      // T floats
    float* prob = buf;                  // alias

    /* Thread 0 computes softmax weights */
    if(d==0){
        for(int j=0;j<T;++j){
            float dot=0.f;
            for(int k=0;k<HDIM;++k){
                float q = Q[t*EMB + h*HDIM + k];
                float kk= K[j*EMB + h*HDIM + k];
                dot += q*kk;
            }
            prob[j] = dot * rsqrtf((float)HDIM);
        }
        softmax(prob,T);
    }
    __syncthreads();

    float val = 0.f;
    for(int j=0;j<T;++j){
        float v = V[j*EMB + h*HDIM + d];
        val += prob[j] * v;
    }
    O[t*EMB + h*HDIM + d] = val;
}

// GELU
__device__ __forceinline__
float gelu(float x){ return 0.5f*x*(1.f+tanhf(0.79788456f*(x+0.044715f*x*x*x))); }

// 2-layer MLP
__global__ void mlp(
        const float* __restrict__ in,
        const float* __restrict__ W1,const float* __restrict__ B1,
        const float* __restrict__ W2,const float* __restrict__ B2,
        float*       __restrict__ out,
        int T)
{
    int t=blockIdx.x, d=threadIdx.x; if(d>=EMB) return;

    /* Hidden = GELU(in·W1 + b1)  (width = 4*EMB) */
    float h=0.f;
    for(int k=0;k<EMB;++k)
        h += in[t*EMB+k]*W1[k*EMB*4 + d];
    h = gelu(h + B1[d]);

    /* Out = hidden·W2 + b2 */
    float o=0.f;
    for(int k=0;k<EMB*4;++k)
        o += h * W2[k*EMB + d];
    out[t*EMB+d] = o + B2[d];
}

int main(){
    
    /* Host random image (pinned) */
    float *hImg; CK(cudaMallocHost(&hImg, CIN*IMG*IMG*sizeof(float)));
    for(size_t i=0;i<CIN*IMG*IMG;++i)
        hImg[i]=(rand()/float(RAND_MAX))*2.f-1.f;

    /* Device tensors */
    float *dImg,*dTok; CK(cudaMalloc(&dImg,CIN*IMG*IMG*sizeof(float)));
    CK(cudaMalloc(&dTok,TOK*EMB*sizeof(float)));

    /* Patch projection + pos emb */
    std::vector<float> hWp(PVEC*EMB), hPos(TOK*EMB);
    for(float&v:hWp)  v=(rand()/float(RAND_MAX))*0.02f-0.01f;
    for(float&v:hPos) v=(rand()/float(RAND_MAX))*0.02f-0.01f;
    float *dWp,*dPos;
    CK(cudaMalloc(&dWp ,hWp .size()*sizeof(float)));
    CK(cudaMalloc(&dPos,hPos.size()*sizeof(float)));
    CK(cudaMemcpy(dWp ,hWp .data(),hWp .size()*sizeof(float),cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dPos,hPos.data(),hPos.size()*sizeof(float),cudaMemcpyHostToDevice));

    /* Shared linear / MLP weights (random init) */
    auto randv=[](){return (rand()/float(RAND_MAX))*0.02f-0.01f;};

    std::vector<float> hW(EMB*EMB);    for(float&v:hW) v=randv();
    std::vector<float> hW1(EMB*4*EMB); for(float&v:hW1)v=randv();
    std::vector<float> hW2(EMB*4*EMB); for(float&v:hW2)v=randv();
    std::vector<float> hB(EMB,0.f),  hB1(EMB*4,0.f), hB2(EMB,0.f);

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

    /* Workspace */
    float *dQ,*dK,*dV,*dCtx,*dOut;
    CK(cudaMalloc(&dQ  ,TOK*EMB*sizeof(float)));
    CK(cudaMalloc(&dK  ,TOK*EMB*sizeof(float)));
    CK(cudaMalloc(&dV  ,TOK*EMB*sizeof(float)));
    CK(cudaMalloc(&dCtx,TOK*EMB*sizeof(float)));
    CK(cudaMalloc(&dOut,TOK*EMB*sizeof(float)));

    /* Events */
    cudaEvent_t eH0,eH1,eK0,eK1,eD0,eD1;
    for(auto ev:{&eH0,&eH1,&eK0,&eK1,&eD0,&eD1}) cudaEventCreate(ev);

    /* Host to device copy */
    cudaEventRecord(eH0);
    CK(cudaMemcpyAsync(dImg,hImg,CIN*IMG*IMG*sizeof(float),cudaMemcpyHostToDevice));
    cudaEventRecord(eH1);
    CK(cudaStreamSynchronize(0));

    /* Kernels */
    cudaEventRecord(eK0);

    patch_embed<<<TOK,EMB>>>(dImg,dWp,dPos,dTok);
    layernorm<<<TOK,1>>>(dTok,TOK);

    linear<<<TOK,EMB>>>(dTok,dW,dB,dQ,TOK,EMB,EMB);
    linear<<<TOK,EMB>>>(dTok,dW,dB,dK,TOK,EMB,EMB);
    linear<<<TOK,EMB>>>(dTok,dW,dB,dV,TOK,EMB,EMB);

    attn<<<dim3(TOK,HEADS),TOK*sizeof(float)>>>(dQ,dK,dV,dCtx,TOK);
    add_vec<<<g1(TOK*EMB),256>>>(dCtx,dTok,TOK*EMB);

    layernorm<<<TOK,1>>>(dCtx,TOK);
    mlp<<<TOK,EMB>>>(dCtx,dW1,dB1,dW2,dB2,dOut,TOK);
    add_vec<<<g1(TOK*EMB),256>>>(dOut,dCtx,TOK*EMB);

    cudaEventRecord(eK1);
    CK(cudaStreamSynchronize(0));

    /* Device to host copy */
    float peek;
    cudaEventRecord(eD0);
    CK(cudaMemcpyAsync(&peek,dOut,sizeof(float),cudaMemcpyDeviceToHost));
    cudaEventRecord(eD1);
    CK(cudaStreamSynchronize(0));

    /* Timings */
    float tH,tK,tD,tTot;
    CK(cudaEventElapsedTime(&tH ,eH0,eH1));
    CK(cudaEventElapsedTime(&tK ,eK0,eK1));
    CK(cudaEventElapsedTime(&tD ,eD0,eD1));
    CK(cudaEventElapsedTime(&tTot,eH0,eD1));

    printf("GPU timings (ms):\n Host-to-Device copy time:      :  %.2f ms\n Kernel execution time:         :  %.2f ms\n Device-to-Host copy time:      :  %.2f ms\n Total GPU time:                :  %.2f ms\n",
           tH,tK,tD,tTot);
     
    return 0;
}