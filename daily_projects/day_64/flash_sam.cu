#ifdef _WIN32
#pragma comment(lib, "cublas.lib")
#pragma comment(lib, "cublasLt.lib")
#endif

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>
#include <vector>
#include <cstdlib>
#include <cmath>
#include <cfloat>

// Helper macros
#define CK(expr) do{ cudaError_t e=(expr);                                    \
    if(e){printf("CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));  \
    exit(1);} }while(0)

#define CB(expr) do{ cublasStatus_t s=(expr);                                 \
    if(s!=CUBLAS_STATUS_SUCCESS){                                             \
        printf("cuBLAS %s:%d status %d\n",__FILE__,__LINE__,s); exit(1);}     \
    }while(0)

// Parameters
constexpr int IMG   = 1024;
constexpr int PATCH = 16;
constexpr int GRID  = IMG / PATCH;              // 64
constexpr int TOK   = GRID * GRID;              // 4096

constexpr int CIN   = 3;
constexpr int PVEC  = CIN * PATCH * PATCH;      // 768
constexpr int EMB   = 256;
constexpr int HEADS = 8;
constexpr int HDIM  = EMB / HEADS;              // 32

constexpr int PTOK = 2;   // keep it minimal: just 2 mask tokens
constexpr int DTOK = TOK + PTOK;                // 4110 tokens

inline dim3 g1(int n){ return dim3((n+255)/256,1,1); }

// Baseline helpers
__global__ void add_vec(float* a,const float* b,int n){
    int i = blockIdx.x*256 + threadIdx.x;
    if(i<n) a[i]+=b[i];
}
__global__ void layernorm(float* x,int T){
    int t=blockIdx.x; float m=0,v=0;
    for(int d=0;d<EMB;++d) m+=x[t*EMB+d];
    m/=EMB;
    for(int d=0;d<EMB;++d){ float z=x[t*EMB+d]-m; v+=z*z; }
    float inv=rsqrtf(v/EMB+1e-5f);
    for(int d=0;d<EMB;++d) x[t*EMB+d]=(x[t*EMB+d]-m)*inv;
}
__global__ void patch2col(const float* img,float* col){
    int p=blockIdx.x, k=threadIdx.x; if(k>=PVEC) return;
    int c=k/(PATCH*PATCH);
    int r=(k%(PATCH*PATCH))/PATCH;
    int c2=k%PATCH;
    int gy=p/GRID, gx=p%GRID;
    int iy=gy*PATCH+r, ix=gx*PATCH+c2;
    col[p*PVEC+k]=img[((c*IMG)+iy)*IMG+ix];
}

// Baseline attention
__device__ void softmax(float* v,int n){
    float m=-FLT_MAX,s=0.f;
    for(int i=0;i<n;++i) m=fmaxf(m,v[i]);
    for(int i=0;i<n;++i){ v[i]=expf(v[i]-m); s+=v[i]; }
    for(int i=0;i<n;++i) v[i]/=s;
}
__global__ void attn_naive(const float* Q,const float* K,const float* V,
                        float* O,int T){
    int t=blockIdx.x, h=blockIdx.y, d=threadIdx.x; if(d>=HDIM) return;
    extern __shared__ float prob[];
    if(d==0){
        for(int j=0;j<T;++j){
            float s=0.f;
            for(int k=0;k<HDIM;++k)
                s+=Q[t*EMB+h*HDIM+k]*K[j*EMB+h*HDIM+k];
            prob[j]=s*rsqrtf((float)HDIM);
        }
        softmax(prob,T);
    }
    __syncthreads();
    float acc=0.f;
    for(int j=0;j<T;++j) acc+=prob[j]*V[j*EMB+h*HDIM+d];
    O[t*EMB+h*HDIM+d]=acc;
}

// Flash Attention (CTA-tiled)
constexpr int MAX_D   = HDIM;  // 32
constexpr int BLOCK_M = 32;
constexpr int BLOCK_K = 64;

__global__ void flash_attn(const float* Q,const float* K,const float* V,
                        float* O,int L,int d){
    int q0=blockIdx.x*BLOCK_M; if(q0>=L) return;

    extern __shared__ float sm[];
    float* shQ=sm;
    float* shK=shQ + BLOCK_M*MAX_D;
    float* shV=shK + BLOCK_K*MAX_D;

    // load local queries
    for(int idx=threadIdx.x; idx<BLOCK_M*d; idx+=blockDim.x){
        int r=idx/d, c=idx%d;
        if(q0+r<L) shQ[r*MAX_D+c]=Q[(q0+r)*d+c];
    }
    __syncthreads();

    int qIdx=threadIdx.x; if(qIdx>=BLOCK_M) return;

    float out[MAX_D]={0};
    float m=-FLT_MAX, l=0.f;

    for(int k0=0;k0<L;k0+=BLOCK_K){
        int tile=min(BLOCK_K,L-k0);
        for(int idx=threadIdx.x; idx<tile*d; idx+=blockDim.x){
            int r=idx/d, c=idx%d;
            shK[r*MAX_D+c]=K[(k0+r)*d+c];
            shV[r*MAX_D+c]=V[(k0+r)*d+c];
        }
        __syncthreads();

        if(q0+qIdx<L){
            const float* qv=shQ+qIdx*MAX_D;
            for(int j=0;j<tile;++j){
                float s=0.f;
                #pragma unroll
                for(int kk=0;kk<MAX_D;++kk) s+=qv[kk]*shK[j*MAX_D+kk];

                float m_new=fmaxf(m,s);
                float exp_s=__expf(s-m_new);
                float l_scale=__expf(m-m_new);
                float l_new=l*l_scale+exp_s;
                float c1=exp_s/l_new;
                float c2=(l*l_scale)/l_new;
                #pragma unroll
                for(int kk=0;kk<MAX_D;++kk)
                    out[kk]=out[kk]*c2 + c1*shV[j*MAX_D+kk];
                m=m_new; l=l_new;
            }
        }
        __syncthreads();
    }
    if(q0+qIdx<L)
        for(int kk=0;kk<d;++kk) O[(q0+qIdx)*d+kk]=out[kk];
}

// Buffer & util helpers
struct Buffers{
    float *dImg,*dCol,*dFeat;
    float *dSeq,*dQ,*dK,*dV,*dCtx;
    float *dWp,*dPos,*dW,*dB;
} B;

void alloc_buffers(){
    CK(cudaMalloc(&B.dImg ,CIN*IMG*IMG*sizeof(float)));
    CK(cudaMalloc(&B.dCol ,TOK*PVEC*sizeof(float)));
    CK(cudaMalloc(&B.dFeat,TOK*EMB*sizeof(float)));

    CK(cudaMalloc(&B.dSeq ,DTOK*EMB*sizeof(float)));
    CK(cudaMalloc(&B.dQ   ,DTOK*EMB*sizeof(float)));
    CK(cudaMalloc(&B.dK   ,DTOK*EMB*sizeof(float)));
    CK(cudaMalloc(&B.dV   ,DTOK*EMB*sizeof(float)));
    CK(cudaMalloc(&B.dCtx ,DTOK*EMB*sizeof(float)));

    CK(cudaMalloc(&B.dWp ,PVEC*EMB*sizeof(float)));
    CK(cudaMalloc(&B.dPos,TOK *EMB*sizeof(float)));
    CK(cudaMalloc(&B.dW  ,EMB*EMB*sizeof(float)));
    CK(cudaMalloc(&B.dB  ,EMB*sizeof(float)));
}
inline float frand(){ return (rand()/float(RAND_MAX))*2.f-1.f; }

void init_weights(){
    std::vector<float> tmp;
    tmp.resize(PVEC*EMB); for(auto&v:tmp)v=frand();
    CK(cudaMemcpy(B.dWp ,tmp.data(),tmp.size()*sizeof(float),cudaMemcpyHostToDevice));
    tmp.resize(TOK*EMB); for(auto&v:tmp)v=frand();
    CK(cudaMemcpy(B.dPos,tmp.data(),tmp.size()*sizeof(float),cudaMemcpyHostToDevice));
    tmp.resize(EMB*EMB); for(auto&v:tmp)v=frand();
    CK(cudaMemcpy(B.dW  ,tmp.data(),tmp.size()*sizeof(float),cudaMemcpyHostToDevice));
    tmp.resize(EMB); for(auto&v:tmp)v=0.f;
    CK(cudaMemcpy(B.dB  ,tmp.data(),tmp.size()*sizeof(float),cudaMemcpyHostToDevice));
}

struct Timing{ float h2d,ker,d2h,tot; };

Timing run(bool flash,cublasHandle_t hc,const float* hImg,float* hOut){
    const float alpha=1.f,beta=0.f;
    cudaEvent_t e0,e1,e2,e3,e4,e5;
    for(auto p:{&e0,&e1,&e2,&e3,&e4,&e5}) cudaEventCreate(&*p);

    // Host to device copy
    cudaEventRecord(e0);
    CK(cudaMemcpyAsync(B.dImg,hImg,CIN*IMG*IMG*sizeof(float),
                    cudaMemcpyHostToDevice));
    cudaEventRecord(e1); CK(cudaStreamSynchronize(0));

    // Kernels
    cudaEventRecord(e2);
    patch2col<<<TOK,PVEC>>>(B.dImg,B.dCol);
    CB(cublasSgemm(hc,CUBLAS_OP_N,CUBLAS_OP_N,
        EMB,TOK,PVEC,&alpha,
        B.dWp,EMB,B.dCol,PVEC,&beta,B.dFeat,EMB));
    add_vec<<<g1(TOK*EMB),256>>>(B.dFeat,B.dPos,TOK*EMB);
    layernorm<<<TOK,1>>>(B.dFeat,TOK);

    // Build seq = features + 2 zeros */
    CK(cudaMemcpyAsync(B.dSeq,B.dFeat,TOK*EMB*sizeof(float),
                    cudaMemcpyDeviceToDevice));
    CK(cudaMemsetAsync(B.dSeq+TOK*EMB,0,PTOK*EMB*sizeof(float)));
    layernorm<<<DTOK,1>>>(B.dSeq,DTOK);

    // Linear Q,K,V
    CB(cublasSgemm(hc,CUBLAS_OP_N,CUBLAS_OP_N,
        EMB,DTOK,EMB,&alpha,B.dW,EMB,B.dSeq,EMB,&beta,B.dQ,EMB));
    CB(cublasSgemm(hc,CUBLAS_OP_N,CUBLAS_OP_N,
        EMB,DTOK,EMB,&alpha,B.dW,EMB,B.dSeq,EMB,&beta,B.dK,EMB));
    CB(cublasSgemm(hc,CUBLAS_OP_N,CUBLAS_OP_N,
        EMB,DTOK,EMB,&alpha,B.dW,EMB,B.dSeq,EMB,&beta,B.dV,EMB));

    // Attention
    if(!flash){
        attn_naive<<<dim3(DTOK,HEADS),HDIM,DTOK*sizeof(float)>>>(
            B.dQ,B.dK,B.dV,B.dCtx,DTOK);
    }else{
        size_t smem=(BLOCK_M+2*BLOCK_K)*MAX_D*sizeof(float);
        dim3 grid((DTOK+BLOCK_M-1)/BLOCK_M);
        for(int h=0;h<HEADS;++h){
            flash_attn<<<grid,BLOCK_M,smem>>>(
                B.dQ + h*HDIM,
                B.dK + h*HDIM,
                B.dV + h*HDIM,
                B.dCtx+ h*HDIM,
                DTOK,HDIM);
        }
    }
    cudaEventRecord(e3); CK(cudaStreamSynchronize(0));

    // Device to host copy
    cudaEventRecord(e4);
    CK(cudaMemcpyAsync(hOut,B.dCtx,DTOK*EMB*sizeof(float),
                    cudaMemcpyDeviceToHost));
    cudaEventRecord(e5); CK(cudaStreamSynchronize(0));

    Timing t;
    cudaEventElapsedTime(&t.h2d,e0,e1);
    cudaEventElapsedTime(&t.ker,e2,e3);
    cudaEventElapsedTime(&t.d2h,e4,e5);
    cudaEventElapsedTime(&t.tot,e0,e5);
    return t;
}

int main(){
    // Host image
    std::vector<float> hImg(CIN*IMG*IMG);
    for(auto&v:hImg) v=frand();
    std::vector<float> hOut(DTOK*EMB);

    alloc_buffers();
    init_weights();

    cublasHandle_t hc; CB(cublasCreate(&hc));
    CB(cublasSetPointerMode(hc,CUBLAS_POINTER_MODE_HOST));

    Timing tBase = run(false,hc,hImg.data(),hOut.data());
    Timing tFlash= run(true ,hc,hImg.data(),hOut.data());

    printf("Baseline\n"
        "  Host-to-Device copy time : %6.2f ms\n"
        "  Kernel execution time    : %6.2f ms\n"
        "  Device-to-Host copy time : %6.2f ms\n"
        "  Total GPU time           : %6.2f ms\n\n",
        tBase.h2d,tBase.ker,tBase.d2h,tBase.tot);

    printf("Flash Attention CTA-tiled\n"
        "  Host-to-Device copy time : %6.2f ms\n"
        "  Kernel execution time    : %6.2f ms\n"
        "  Device-to-Host copy time : %6.2f ms\n"
        "  Total GPU time           : %6.2f ms\n\n",
        tFlash.h2d,tFlash.ker,tFlash.d2h,tFlash.tot);

    cublasDestroy(hc);
    return 0;
}

