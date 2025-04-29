#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <curand.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>

#ifdef _MSC_VER
#pragma comment(lib,"curand.lib")
#endif

#define CK(x) do{cudaError_t e=(x); if(e){                                 \
    printf("CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));     \
    exit(1);} }while(0)

// Hyper parameters
constexpr int   BATCH = 16;
constexpr int   Z_DIM = 512;
constexpr int   W_DIM = 512;
constexpr int   F_CH  = 128;          // max feature maps
constexpr int   STEPS = 20;           // demo iterations
constexpr float LR_G  = 2.5e-3f, LR_D = 2.5e-3f;
constexpr int   RES   = 16;           // final resolution

// Helpers
inline dim3 grid1d(int N){ return dim3((N+255)/256); }
__global__ void fill_const(float* p,int n,float v){int i=blockIdx.x*256+threadIdx.x; if(i<n) p[i]=v;}
__global__ void lrelu_inplace(float* x,int n){int i=blockIdx.x*256+threadIdx.x; if(i<n){float y=x[i]; x[i]=y>0?y:0.2f*y;}}
__global__ void bias_act(float* fmap,const float* b,int C,int H,int W){
  int c = blockIdx.x, hw = threadIdx.x;
  if(c>=C || hw>=H*W) return;
  fmap[c*H*W+hw] = tanhf(fmap[c*H*W+hw] + b[c]);
}

// Naive 3×3 mod-conv kernel
__global__ void mod_conv3x3(const float* x,const float* w,const float* style,
                            float* y,int B,int Cin,int Cout,int H,int W)
{
  int n=blockIdx.x, co=blockIdx.y, hw=threadIdx.x;
  if(hw>=H*W) return;
  float sum=0; int h=hw/W, ww=hw%W;
  for(int ci=0;ci<Cin;++ci){
    float s=style[n*Cin+ci];
    for(int ky=-1;ky<=1;++ky)
      for(int kx=-1;kx<=1;++kx){
        int ih=h+ky, iw=ww+kx;
        if(ih<0||ih>=H||iw<0||iw>=W) continue;
        float xv=x[((n*Cin+ci)*H+ih)*W+iw];
        float wv=w[(((co*Cin)+ci)*3+(ky+1))*3+(kx+1)];
        sum+=xv*(s*wv);
      }
  }
  y[((n*Cout+co)*H+h)*W+ww]=sum;
}

// Mapping FC kernel
__global__ void fc512(const float* in,const float* W,const float* b,float* out,int N){
  int n=blockIdx.x, o=threadIdx.x; if(o>=512) return;
  float acc=0; for(int i=0;i<512;++i) acc+=in[n*512+i]*W[i*512+o];
  out[n*512+o]=acc+b[o];
}

// Lightweight tensor helper
struct Tensor{
  float* d=nullptr; size_t sz=0;
  Tensor()=default;
  explicit Tensor(size_t n){alloc(n);}
  void alloc(size_t n){sz=n; CK(cudaMalloc(&d,sz*sizeof(float))); }
  void zero(){ CK(cudaMemset(d,0,sz*sizeof(float))); }
};

// Modulated-conv block
struct ModBlock{
  Tensor w,b;
  int Cin,Cout,H,W;
  ModBlock(int cin,int cout,int h,int width)
       : Cin(cin),Cout(cout),H(h),W(width)
  {
    w.alloc(Cout*Cin*3*3);  b.alloc(Cout);
    std::vector<float> tmp(w.sz + b.sz);
    for(auto&v:tmp) v = (rand()/float(RAND_MAX))*0.02f-0.01f;
    CK(cudaMemcpy(w.d,tmp.data(),             w.sz*sizeof(float),cudaMemcpyHostToDevice));
    CK(cudaMemcpy(b.d,tmp.data()+ w.sz,       b.sz*sizeof(float),cudaMemcpyHostToDevice));
  }
};

// Mapping network
struct Mapping{
  Tensor w[8],b[8];
  Mapping(){
    for(int l=0;l<8;++l){
      w[l].alloc(512*512); b[l].alloc(512);
      std::vector<float> tmp(512*512+512);
      for(auto&v:tmp) v=(rand()/float(RAND_MAX))*0.02f-0.01f;
      CK(cudaMemcpy(w[l].d,tmp.data(),       512*512*sizeof(float),cudaMemcpyHostToDevice));
      CK(cudaMemcpy(b[l].d,tmp.data()+512*512,512*sizeof(float),cudaMemcpyHostToDevice));
    }
  }
};

// Global feature tensors
Tensor z_lat(BATCH*Z_DIM), w_lat(BATCH*W_DIM);
Tensor g4(F_CH*4*4*BATCH), g8(F_CH/2*8*8*BATCH), g16(3*16*16*BATCH);

// Generator forward
void generator_forward(const float* z,Mapping& map,
                       ModBlock& m4,ModBlock& m8,Tensor& out,
                       cudaStream_t stream=0)
{
  CK(cudaMemcpyAsync(w_lat.d,z,BATCH*Z_DIM*sizeof(float),
                     cudaMemcpyDeviceToDevice,stream));
  for(int l=0;l<8;++l){
    fc512<<<BATCH,512,0,stream>>>(w_lat.d,map.w[l].d,map.b[l].d,w_lat.d,BATCH);
    lrelu_inplace<<<grid1d(BATCH*512),256,0,stream>>>(w_lat.d,BATCH*512);
  }
  CK(cudaMemsetAsync(g4.d,0,g4.sz*sizeof(float),stream));
  mod_conv3x3<<<dim3(BATCH,m4.Cout),4*4,0,stream>>>(
      g4.d,m4.w.d,w_lat.d,g8.d,BATCH,m4.Cin,m4.Cout,4,4);
  bias_act<<<m4.Cout,4*4,0,stream>>>(g8.d,m4.b.d,m4.Cout,4,4);

  mod_conv3x3<<<dim3(BATCH,m8.Cout),8*8,0,stream>>>(
      g8.d,m8.w.d,w_lat.d,g16.d,BATCH,m8.Cin,m8.Cout,8,8);
  bias_act<<<m8.Cout,8*8,0,stream>>>(g16.d,m8.b.d,m8.Cout,8,8);

  out = g16;
}

int main(){
  printf("Micro-StyleGAN2 | %dx%d | B=%d | steps=%d | LR_G %.3e LR_D %.3e\n",
         RES,RES,BATCH,STEPS,LR_G,LR_D);

  Mapping mapping;
  ModBlock m4(F_CH,F_CH/2,4,4);
  ModBlock m8(F_CH/2,F_CH/4,8,8);

  std::vector<float> h_z(BATCH*Z_DIM);
  curandGenerator_t gen; curandCreateGeneratorHost(&gen,CURAND_RNG_PSEUDO_DEFAULT);
  curandSetPseudoRandomGeneratorSeed(gen,2025);

  cudaEvent_t e_h2d,e_ker,e_d2h,e_tot;
  cudaEventCreate(&e_h2d); cudaEventCreate(&e_ker);
  cudaEventCreate(&e_d2h); cudaEventCreate(&e_tot);

  for(int it=0; it<STEPS; ++it){
    curandGenerateNormal(gen,h_z.data(),h_z.size(),0,1);

    cudaEventRecord(e_tot);
    cudaEventRecord(e_h2d);
    CK(cudaMemcpy(z_lat.d,h_z.data(),h_z.size()*sizeof(float),cudaMemcpyHostToDevice));

    cudaEventRecord(e_ker);
    generator_forward(z_lat.d,mapping,m4,m8,g16);

    float dummy; cudaMemcpy(&dummy,g16.d,sizeof(float),cudaMemcpyDeviceToHost);
    cudaEventRecord(e_d2h);

    cudaEventRecord(e_tot); cudaEventSynchronize(e_tot);

    float th2d,tker,td2h,ttot;
    cudaEventElapsedTime(&th2d,e_h2d,e_ker);
    cudaEventElapsedTime(&tker,e_ker,e_d2h);
    cudaEventElapsedTime(&td2h,e_d2h,e_tot);
    cudaEventElapsedTime(&ttot,e_h2d,e_tot);

    printf("Iter %3d | H2D %.3f ms | Kern %.3f ms | D2H %.3f ms | Tot %.3f ms\n",
           it,th2d,tker,td2h,ttot);
  }
  return 0;
}
