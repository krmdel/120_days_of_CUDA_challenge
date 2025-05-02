#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <cstdlib>
#define CK(x) do{ cudaError_t e=(x); if(e){                                 \
                 printf("CUDA %s:%d %s\n",__FILE__,__LINE__,                \
                        cudaGetErrorString(e)); exit(1);} }while(0)

// Parameters
constexpr int GRID  = 64;
constexpr int FTOK  = GRID*GRID;     // 4 096 image patches
constexpr int PTOK  = 14;            // prompt tokens
constexpr int OTOK  = 2;             // output tokens
constexpr int TTOK  = PTOK + OTOK;   // 16 total decoder tokens
constexpr int EMB   = 256;
constexpr int HEAD  = 8;
constexpr int HDIM  = EMB/HEAD;

// Helpers
inline dim3 g1(int n){ return dim3((n+255)/256,1,1); }

// Kernels

// Vector addition
__global__ void add_vec(float* a,const float* b,int n){
    int i=blockIdx.x*256+threadIdx.x; if(i<n) a[i]+=b[i];
}

// Token LN
__global__ void ln_tok(float* t){
    int id=blockIdx.x; float m=0,v=0;
    for(int d=0;d<EMB;++d) m+=t[id*EMB+d];  m/=EMB;
    for(int d=0;d<EMB;++d){ float z=t[id*EMB+d]-m; v+=z*z; }
    float inv=rsqrtf(v/EMB+1e-5f);
    for(int d=0;d<EMB;++d) t[id*EMB+d]=(t[id*EMB+d]-m)*inv;
}
// Feature LN
__global__ void ln_feat(float* f){
    int id=blockIdx.x; float m=0,v=0;
    for(int d=0;d<EMB;++d) m+=f[id*EMB+d];  m/=EMB;
    for(int d=0;d<EMB;++d){ float z=f[id*EMB+d]-m; v+=z*z; }
    float inv=rsqrtf(v/EMB+1e-5f);
    for(int d=0;d<EMB;++d) f[id*EMB+d]=(f[id*EMB+d]-m)*inv;
}

// Linear
__global__ void linear(const float* in,const float* W,float* out,int M){
    int m=blockIdx.x, d=threadIdx.x; if(d>=EMB) return;
    float acc=0; for(int k=0;k<EMB;++k) acc+=in[m*EMB+k]*W[k*EMB+d];
    out[m*EMB+d]=acc;
}

// Softmax
__device__ void softmax(float* s,int n){
    float m=-1e9f,sum=0; for(int i=0;i<n;++i) m=max(m,s[i]);
    for(int i=0;i<n;++i){ s[i]=expf(s[i]-m); sum+=s[i]; }
    for(int i=0;i<n;++i) s[i]/=sum;
}

// Simple attention
template<int QLEN,int KLEN>
__global__ void attn(const float* Q,const float* K,const float* V,float* O){
    int q=blockIdx.x, h=blockIdx.y, d=threadIdx.x; if(d>=HDIM) return;
    extern __shared__ float probs[];        // KLEN floats

    if(d==0){
        for(int k=0;k<KLEN;++k){
            float dot=0;
            for(int u=0;u<HDIM;++u)
                dot += Q[q*EMB+h*HDIM+u]*K[k*EMB+h*HDIM+u];
            probs[k]=dot*rsqrtf((float)HDIM);
        }
        softmax(probs,KLEN);
    }
    __syncthreads();

    float ctx=0;
    for(int k=0;k<KLEN;++k)
        ctx+=probs[k]*V[k*EMB+h*HDIM+d];
    O[q*EMB+h*HDIM+d]=ctx;
}

// GELU & MLP
__device__ float gelu(float x){ return 0.5f*x*(1+erff(x*0.70710678f)); }
__global__ void mlp(const float* in,const float* W1,const float* W2,float* out){
    int t=blockIdx.x,d=threadIdx.x; if(d>=EMB) return;
    float h=0; for(int k=0;k<EMB;++k) h+=in[t*EMB+k]*W1[k*EMB+d];
    h=gelu(h);
    float o=0; for(int k=0;k<EMB;++k) o+=h*W2[k*EMB+d];
    out[t*EMB+d]=o;
}

int main(){

    // Host inputs
    std::vector<float> hFeat(FTOK*EMB), hTok(TTOK*EMB);
    for(float&v:hFeat) v=(rand()/float(RAND_MAX))*0.1f-0.05f;
    for(float&v:hTok)  v=(rand()/float(RAND_MAX))*0.1f-0.05f;

    // Device buffers
    float *dF,*dT; CK(cudaMalloc(&dF,hFeat.size()*sizeof(float)));
    CK(cudaMalloc(&dT,hTok .size()*sizeof(float)));

    // Single random weights
    std::vector<float> hW(EMB*EMB);
    for(float&v:hW) v=(rand()/float(RAND_MAX))*0.02f-0.01f;
    float *dW; CK(cudaMalloc(&dW,hW.size()*sizeof(float)));
    CK(cudaMemcpy(dW,hW.data(),hW.size()*sizeof(float),cudaMemcpyHostToDevice));

    // Work buffers
    float *dQ,*dK,*dV,*dBuf1,*dBuf2;
    CK(cudaMalloc(&dQ,FTOK*EMB*sizeof(float)));
    CK(cudaMalloc(&dK,FTOK*EMB*sizeof(float)));
    CK(cudaMalloc(&dV,FTOK*EMB*sizeof(float)));
    CK(cudaMalloc(&dBuf1,FTOK*EMB*sizeof(float)));
    CK(cudaMalloc(&dBuf2,FTOK*EMB*sizeof(float)));

    // Events
    cudaEvent_t H0,H1,K0,K1,D0,D1;
    for(auto ev:{&H0,&H1,&K0,&K1,&D0,&D1}) cudaEventCreate(ev);

    // Host to decice copy
    cudaEventRecord(H0);
    CK(cudaMemcpyAsync(dF,hFeat.data(),hFeat.size()*sizeof(float),cudaMemcpyHostToDevice));
    CK(cudaMemcpyAsync(dT,hTok .data(),hTok .size()*sizeof(float),cudaMemcpyHostToDevice));
    cudaEventRecord(H1); CK(cudaStreamSynchronize(0));

    // Kernels
    cudaEventRecord(K0);

    ln_tok <<<TTOK,1>>>(dT);

    // Self-attention
    linear<<<TTOK,EMB>>>(dT,dW,dQ,TTOK);
    linear<<<TTOK,EMB>>>(dT,dW,dK,TTOK);
    linear<<<TTOK,EMB>>>(dT,dW,dV,TTOK);
    attn<TTOK,TTOK><<<dim3(TTOK,HEAD),TTOK*sizeof(float)>>>(dQ,dK,dV,dBuf1);
    add_vec<<<g1(TTOK*EMB),256>>>(dBuf1,dT,TTOK*EMB);

    // Tokens → image
    ln_tok <<<TTOK,1>>>(dBuf1);
    ln_feat<<<FTOK,1>>>(dF);
    linear<<<TTOK,EMB>>>(dBuf1,dW,dQ,TTOK);
    linear<<<FTOK,EMB>>>(dF,dW,dK,FTOK);
    linear<<<FTOK,EMB>>>(dF,dW,dV,FTOK);
    attn<TTOK,FTOK><<<dim3(TTOK,HEAD),FTOK*sizeof(float)>>>(dQ,dK,dV,dBuf2);
    add_vec<<<g1(TTOK*EMB),256>>>(dBuf2,dBuf1,TTOK*EMB);

    // Image → tokens
    ln_tok <<<TTOK,1>>>(dBuf2);
    ln_feat<<<FTOK,1>>>(dF);
    linear<<<FTOK,EMB>>>(dF,dW,dK,FTOK);
    linear<<<FTOK,EMB>>>(dF,dW,dV,FTOK);
    linear<<<TTOK,EMB>>>(dBuf2,dW,dQ,TTOK);
    attn<FTOK,TTOK><<<dim3(FTOK,HEAD),TTOK*sizeof(float)>>>(dK,dQ,dV,dBuf1);
    add_vec<<<g1(FTOK*EMB),256>>>(dBuf1,dF,FTOK*EMB);

    // Token MLP
    mlp<<<TTOK,EMB>>>(dBuf2,dW,dW,dT);   // reuse dW

    cudaEventRecord(K1); CK(cudaStreamSynchronize(0));

    // Device to host
    std::vector<float> hBackF(FTOK*EMB), hBackT(TTOK*EMB);
    cudaEventRecord(D0);
    CK(cudaMemcpyAsync(hBackF.data(),dBuf1,hBackF.size()*sizeof(float),cudaMemcpyDeviceToHost));
    CK(cudaMemcpyAsync(hBackT.data(),dT ,  hBackT.size()*sizeof(float),cudaMemcpyDeviceToHost));
    cudaEventRecord(D1); CK(cudaStreamSynchronize(0));

    // Timing output
    float msH,msK,msD,msTot;
    cudaEventElapsedTime(&msH,H0,H1);
    cudaEventElapsedTime(&msK,K0,K1);
    cudaEventElapsedTime(&msD,D0,D1);
    cudaEventElapsedTime(&msTot,H0,D1);

    printf("GPU timings (ms):\n Host-to-Device copy time:      :  %.2f ms\n Kernel execution time:         :  %.2f ms\n Device-to-Host copy time:      :  %.2f ms\n Total GPU time:                :  %.2f ms\n",
        msH,msK,msD,msTot);

    return 0;
}
