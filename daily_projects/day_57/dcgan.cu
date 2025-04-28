#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#define CK(x) do{cudaError_t e=(x); if(e){printf("CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} }while(0)

// Hyperparameters
constexpr int   BATCH = 64, Z = 100, PIX = 28*28;
constexpr int   HG = 256,  HD = 256,  EPOCHS = 20;
constexpr float LR = 2e-4f;

// Helpers
inline dim3 vgrid(int n){ return dim3((n + 255) / 256, 1, 1); }
__global__ void fill(float* p,int n,float v){int i=blockIdx.x*256+threadIdx.x; if(i<n) p[i]=v;}
__global__ void relu(int n,const float* in,float* out){int i=blockIdx.x*256+threadIdx.x; if(i<n) out[i]=fmaxf(in[i],0);}
__global__ void relu_g(int n,const float* act,const float* gin,float* gout){int i=blockIdx.x*256+threadIdx.x; if(i<n) gout[i]=act[i]>0?gin[i]:0;}
__global__ void tanh_f(int n,const float* in,float* out){int i=blockIdx.x*256+threadIdx.x; if(i<n) out[i]=tanhf(in[i]);}
__global__ void tanh_g(int n,const float* act,const float* gin,float* gout){int i=blockIdx.x*256+threadIdx.x; if(i<n){float y=act[i]; gout[i]=gin[i]*(1-y*y);} }
__global__ void sigmoid(int n,float* x){int i=blockIdx.x*256+threadIdx.x; if(i<n) x[i]=1.f/(1+expf(-x[i]));}
__global__ void bce_grad(const float* p,const float* t,int n,float* g,float* loss){
  __shared__ float sh[256]; float l=0; int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<n){float q=fminf(fmaxf(p[i],1e-7f),1-1e-7f); l=-(t[i]*logf(q)+(1-t[i])*logf(1-q)); g[i]=q-t[i];}
  sh[threadIdx.x]=l; __syncthreads();
  for(int s=blockDim.x/2;s; s>>=1){ if(threadIdx.x<s) sh[threadIdx.x]+=sh[threadIdx.x+s]; __syncthreads(); }
  if(threadIdx.x==0) atomicAdd(loss,sh[0]);
}

// Tiny BLAS
__global__ void lin_fwd(const float* X,const float* W,const float* b,int M,int K,int N,float* Y){
  int m=blockIdx.y*blockDim.y+threadIdx.y, n=blockIdx.x*blockDim.x+threadIdx.x;
  if(m>=M||n>=N) return; float acc=0; for(int k=0;k<K;++k) acc+=X[m*K+k]*W[k*N+n]; Y[m*N+n]=acc+b[n];
}
__global__ void dW_db(const float* X,const float* dY,int M,int K,int N,float* dW,float* db){
  int k=blockIdx.y*blockDim.y+threadIdx.y, n=blockIdx.x*blockDim.x+threadIdx.x;
  if(k>=K||n>=N) return; float acc=0,bacc=0;
  for(int m=0;m<M;++m){acc+=X[m*K+k]*dY[m*N+n]; if(k==0) bacc+=dY[m*N+n];}
  dW[k*N+n]=acc; if(k==0) db[n]=bacc;
}
__global__ void dX(const float* dY,const float* W,int M,int K,int N,float* dX){
  int m=blockIdx.y*blockDim.y+threadIdx.y, k=blockIdx.x*blockDim.x+threadIdx.x;
  if(m>=M||k>=K) return; float acc=0; for(int n=0;n<N;++n) acc+=dY[m*N+n]*W[k*N+n]; dX[m*K+k]=acc;
}
__global__ void sgd(float* w,const float* dw,int n,float lr){
  int i=blockIdx.x*256+threadIdx.x; if(i<n) w[i]-=lr*dw[i]/BATCH;
}

// Layer wrapper
struct FC{
  int in,out; float *W,*b,*dW,*db;
  FC(int i,int o):in(i),out(o){
    CK(cudaMalloc(&W,i*o*sizeof(float))); CK(cudaMalloc(&b,o*sizeof(float)));
    CK(cudaMalloc(&dW,i*o*sizeof(float))); CK(cudaMalloc(&db,o*sizeof(float)));
    std::vector<float> tmp(i*o+o); float s=sqrtf(6.f/(i+o));
    for(auto&v:tmp) v=((rand()/float(RAND_MAX))*2-1)*s;
    CK(cudaMemcpy(W,tmp.data(),i*o*sizeof(float),cudaMemcpyHostToDevice));
    CK(cudaMemcpy(b,tmp.data()+i*o,o*sizeof(float),cudaMemcpyHostToDevice));
  }
  void step(){ sgd<<<vgrid(in*out),256>>>(W,dW,in*out,LR); sgd<<<vgrid(out),256>>>(b,db,out,LR); }
};

// Global buffers
float *d_z,*realX,*fakeX,*g_h,*g_h_grad,*d_h,*d_h_grad,*logit,*target,*d_loss_dev,*g_loss_dev;
void alloc(){
  CK(cudaMalloc(&d_z, BATCH*Z*sizeof(float)));
  CK(cudaMalloc(&realX,BATCH*PIX*sizeof(float)));
  CK(cudaMalloc(&fakeX,BATCH*PIX*sizeof(float)));
  CK(cudaMalloc(&g_h,  BATCH*HG*sizeof(float)));
  CK(cudaMalloc(&g_h_grad,BATCH*HG*sizeof(float)));
  CK(cudaMalloc(&d_h,  BATCH*HD*sizeof(float)));
  CK(cudaMalloc(&d_h_grad,BATCH*HD*sizeof(float)));
  CK(cudaMalloc(&logit,BATCH*sizeof(float)));
  CK(cudaMalloc(&target,BATCH*sizeof(float)));
  CK(cudaMalloc(&d_loss_dev,sizeof(float)));
  CK(cudaMalloc(&g_loss_dev,sizeof(float)));
}

// Forward helpers
void gen_fwd(const float* z,FC& g1,FC& g2){
  dim3 gf((HG+15)/16,(BATCH+15)/16); lin_fwd<<<gf,{16,16}>>>(z,g1.W,g1.b,BATCH,Z,HG,g_h);
  relu<<<vgrid(BATCH*HG),256>>>(BATCH*HG,g_h,g_h);
  dim3 go((PIX+15)/16,(BATCH+15)/16); lin_fwd<<<go,{16,16}>>>(g_h,g2.W,g2.b,BATCH,HG,PIX,fakeX);
  tanh_f<<<vgrid(BATCH*PIX),256>>>(BATCH*PIX,fakeX,fakeX);
}
void disc_fwd(const float* x,FC& d1,FC& d2){
  dim3 df((HD+15)/16,(BATCH+15)/16); lin_fwd<<<df,{16,16}>>>(x,d1.W,d1.b,BATCH,PIX,HD,d_h);
  relu<<<vgrid(BATCH*HD),256>>>(BATCH*HD,d_h,d_h);
  dim3 do_((1+15)/16,(BATCH+15)/16); lin_fwd<<<do_,{16,16}>>>(d_h,d2.W,d2.b,BATCH,HD,1,logit);
  sigmoid<<<vgrid(BATCH),256>>>(BATCH,logit);
}

// Training utilities
void disc_update(FC& d1,FC& d2,const float* X,int label){
  disc_fwd(X,d1,d2);
  fill<<<vgrid(BATCH),256>>>(target,BATCH,float(label));
  bce_grad<<<vgrid(BATCH),256>>>(logit,target,BATCH,target,d_loss_dev);

  dim3 gd2((1+15)/16,(HD+15)/16);
  dW_db<<<gd2,{16,16}>>>(d_h,target,BATCH,HD,1,d2.dW,d2.db);
  dX   <<<dim3((HD+15)/16,(BATCH+15)/16),{16,16}>>>(target,d2.W,BATCH,HD,1,d_h_grad);
  relu_g<<<vgrid(BATCH*HD),256>>>(BATCH*HD,d_h,d_h_grad,d_h_grad);
  dim3 gd1((HD+15)/16,(PIX+15)/16);
  dW_db<<<gd1,{16,16}>>>(X,d_h_grad,BATCH,PIX,HD,d1.dW,d1.db);
  d2.step(); d1.step();
}

int main(){
  srand(0); alloc();
  FC g1(Z,HG),g2(HG,PIX), d1(PIX,HD),d2(HD,1);

  std::vector<float> hz(BATCH*Z), hreal(BATCH*PIX);
  printf("Running DCGAN (FC) | LR %.1e\n",LR);

  // CUDA events for timing
  cudaEvent_t e_h2d,e_ker,e_d2h,e_tot;
  cudaEventCreate(&e_h2d); cudaEventCreate(&e_ker);
  cudaEventCreate(&e_d2h); cudaEventCreate(&e_tot);

  for(int e=0;e<EPOCHS;++e){
    /* ------ prepare host mini-batch ------ */
    for(auto&v:hreal) v=(rand()%256)/127.5f-1.f;
    for(auto&v:hz)    v=(rand()/float(RAND_MAX))*2.f-1.f;

    float d_loss_h=0,g_loss_h=0;

    cudaEventRecord(e_tot);

    // Host to device copy
    cudaEventRecord(e_h2d);
    CK(cudaMemcpy(d_z,hz.data(),hz.size()*sizeof(float),cudaMemcpyHostToDevice));
    CK(cudaMemcpy(realX,hreal.data(),hreal.size()*sizeof(float),cudaMemcpyHostToDevice));

    // Kernel
    cudaEventRecord(e_ker);

    cudaMemset(d_loss_dev,0,sizeof(float));
    cudaMemset(g_loss_dev,0,sizeof(float));

    // Train discriminator
    disc_update(d1,d2,realX,1);
    gen_fwd(d_z,g1,g2);  disc_update(d1,d2,fakeX,0);

    // Train generator
    gen_fwd(d_z,g1,g2); disc_fwd(fakeX,d1,d2);
    fill<<<vgrid(BATCH),256>>>(target,BATCH,1.f);
    bce_grad<<<vgrid(BATCH),256>>>(logit,target,BATCH,target,g_loss_dev);

    // Back-prop through discriminator
    dX<<<dim3((HD+15)/16,(BATCH+15)/16),{16,16}>>>(target,d2.W,BATCH,HD,1,d_h_grad);
    relu_g<<<vgrid(BATCH*HD),256>>>(BATCH*HD,d_h,d_h_grad,d_h_grad);
    dX<<<dim3((PIX+15)/16,(BATCH+15)/16),{16,16}>>>(d_h_grad,d1.W,BATCH,PIX,HD,g_h_grad);

    // through generator
    tanh_g<<<vgrid(BATCH*PIX),256>>>(BATCH*PIX,fakeX,g_h_grad,g_h_grad);
    dim3 gg2((PIX+15)/16,(HG+15)/16);
    dW_db<<<gg2,{16,16}>>>(g_h,g_h_grad,BATCH,HG,PIX,g2.dW,g2.db);
    dX<<<dim3((HG+15)/16,(BATCH+15)/16),{16,16}>>>(g_h_grad,g2.W,BATCH,HG,PIX,g_h_grad);
    relu_g<<<vgrid(BATCH*HG),256>>>(BATCH*HG,g_h,g_h_grad,g_h_grad);
    dim3 gg1((HG+15)/16,(Z+15)/16);
    dW_db<<<gg1,{16,16}>>>(d_z,g_h_grad,BATCH,Z,HG,g1.dW,g1.db);
    g2.step(); g1.step();

    // Device to host timing & loss copy
    cudaEventRecord(e_d2h);
    CK(cudaMemcpy(&d_loss_h,d_loss_dev,sizeof(float),cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(&g_loss_h,g_loss_dev,sizeof(float),cudaMemcpyDeviceToHost));

    cudaEventRecord(e_tot); cudaEventSynchronize(e_tot);

    float t_h2d,t_ker,t_d2h,t_tot;
    cudaEventElapsedTime(&t_h2d,e_h2d,e_ker);
    cudaEventElapsedTime(&t_ker,e_ker,e_d2h);
    cudaEventElapsedTime(&t_d2h,e_d2h,e_tot);
    cudaEventElapsedTime(&t_tot,e_h2d,e_tot);

    printf("Epoch %3d | H2D %.3f ms | Kern %.3f ms | D2H %.3f ms | Tot %.3f ms | D %.4f | G %.4f\n",
           e,t_h2d,t_ker,t_d2h,t_tot,d_loss_h/BATCH,g_loss_h/BATCH);
  }

  CK(cudaDeviceSynchronize());
  return 0;
}
