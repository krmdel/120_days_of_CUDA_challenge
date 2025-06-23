#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <chrono>
#include <fstream>
#include <iostream>
#include <cstring>           // <-- for std::memcpy

#define CUDA_CHECK(x)  do{ cudaError_t e=(x); if(e!=cudaSuccess){                \
  std::cerr<<"CUDA error: "<<cudaGetErrorString(e)<<" ("<<__FILE__<<':'          \
           <<__LINE__<<")\n"; std::exit(1);} }while(0)
template<class TP> inline double ms(TP a,TP b){
  return std::chrono::duration<double,std::milli>(b-a).count(); }

// PGM loader
bool readPGM(const char* fn,std::vector<uint8_t>& img,int& W,int& H){
  std::ifstream f(fn,std::ios::binary); if(!f) return false;
  std::string m; int maxv; f>>m>>W>>H>>maxv; f.get();
  if(m!="P5"||maxv!=255) return false;
  img.resize(size_t(W)*H); f.read(reinterpret_cast<char*>(img.data()),img.size());
  return bool(f);
}

// CPU reference pipeline
void sobelCPU(const std::vector<uint8_t>& g,int W,int H,
              std::vector<float>& mag,std::vector<float>& ang){
  mag.assign(g.size(),0); ang.assign(g.size(),0);
  for(int y=1;y<H-1;++y)
    for(int x=1;x<W-1;++x){
      int i=y*W+x;
      int gx=-g[(y-1)*W+x-1]+g[(y-1)*W+x+1]-2*g[y*W+x-1]+2*g[y*W+x+1]
            -g[(y+1)*W+x-1]+g[(y+1)*W+x+1];
      int gy= g[(y-1)*W+x-1]+2*g[(y-1)*W+x]+g[(y-1)*W+x+1]
            -g[(y+1)*W+x-1]-2*g[(y+1)*W+x]-g[(y+1)*W+x+1];
      float fx=float(gx), fy=float(gy);
      mag[i]=std::sqrt(fx*fx+fy*fy);
      float a=std::atan2(fy,fx); if(a<0)a+=3.14159265f;
      ang[i]=a;
    }
}
void cellCPU(const std::vector<float>& mag,const std::vector<float>& ang,
             int W,int H,std::vector<float>& cell){
  int Cx=W/8,Cy=H/8; cell.assign(size_t(Cx)*Cy*9,0);
  for(int cy=0;cy<Cy;++cy)
    for(int cx=0;cx<Cx;++cx){
      float* h=&cell[(cy*Cx+cx)*9];
      for(int ly=0;ly<8;++ly)
        for(int lx=0;lx<8;++lx){
          int gx=cx*8+lx, gy=cy*8+ly;
          float m=mag[gy*W+gx], a=ang[gy*W+gx];
          int b=int((a*57.2957795f)/20.f); if(b>8)b=8;
          h[b]+=m;
        }
    }
}
void blockCPU(const std::vector<float>& cell,int Cx,int Cy,
              std::vector<float>& desc){
  int Bx=Cx-1, By=Cy-1; desc.resize(size_t(Bx)*By*36);
  const float eps=1e-6f, clip=0.2f;
  for(int by=0;by<By;++by)
    for(int bx=0;bx<Bx;++bx){
      float v[36];
      for(int cy=0;cy<2;++cy)
        for(int cx=0;cx<2;++cx){
          const float* h=&cell[((by+cy)*Cx+(bx+cx))*9];
          for(int b=0;b<9;++b) v[(cy*2+cx)*9+b]=h[b];
        }
      float n=eps; for(float s:v) n+=s*s; n=std::sqrt(n);
      for(float& s:v){ s/=n; if(s>clip)s=clip; }
      n=eps; for(float s:v) n+=s*s; n=std::sqrt(n);
      for(int i=0;i<36;++i) desc[(by*Bx+bx)*36+i]=v[i]/n;
    }
}

// CUDA kernels
__device__ __forceinline__ float wSum(float v){
  for(int o=16;o;o>>=1) v+=__shfl_down_sync(0xffffffff,v,o);
  return v;
}

// Sobel gradient kernel
__global__ void sobelKer(const uint8_t* g,float* mag,float* ang,int W,int H){
  extern __shared__ uint8_t sh[];
  const int B=16; int lx=threadIdx.x, ly=threadIdx.y;
  int gx=blockIdx.x*B+lx, gy=blockIdx.y*B+ly;
  int shW=B+2, sx=lx+1, sy=ly+1;
  if(gx<W&&gy<H) sh[sy*shW+sx]=g[gy*W+gx];
  if(lx==0&&gx>0)            sh[sy*shW]      =g[gy*W+gx-1];
  if(lx==B-1&&gx+1<W)        sh[sy*shW+sx+1] =g[gy*W+gx+1];
  if(ly==0&&gy>0)            sh[sx]          =g[(gy-1)*W+gx];
  if(ly==B-1&&gy+1<H)        sh[(sy+1)*shW+sx]=g[(gy+1)*W+gx];
  if(lx==0&&ly==0&&gx>0&&gy>0)               sh[0]=g[(gy-1)*W+gx-1];
  if(lx==B-1&&ly==0&&gx+1<W&&gy>0)           sh[sx+1]=g[(gy-1)*W+gx+1];
  if(lx==0&&ly==B-1&&gx>0&&gy+1<H)           sh[(sy+1)*shW]=g[(gy+1)*W+gx-1];
  if(lx==B-1&&ly==B-1&&gx+1<W&&gy+1<H)       sh[(sy+1)*shW+sx+1]=g[(gy+1)*W+gx+1];
  __syncthreads();
  if(gx>0&&gx<W-1&&gy>0&&gy<H-1){
    int gxv=-sh[(sy-1)*shW+sx-1]+sh[(sy-1)*shW+sx+1]
            -2*sh[sy*shW+sx-1]  +2*sh[sy*shW+sx+1]
            -sh[(sy+1)*shW+sx-1]+sh[(sy+1)*shW+sx+1];
    int gyv= sh[(sy-1)*shW+sx-1]+2*sh[(sy-1)*shW+sx]+sh[(sy-1)*shW+sx+1]
            -sh[(sy+1)*shW+sx-1]-2*sh[(sy+1)*shW+sx]-sh[(sy+1)*shW+sx+1];
    float fx=float(gxv), fy=float(gyv);
    int idx=gy*W+gx;
    mag[idx]=sqrtf(fx*fx+fy*fy);
    float a=atan2f(fy,fx); if(a<0)a+=3.14159265f;
    ang[idx]=a;
  }
}

// Cell histogram kernel
__global__ void cellKer(const float* mag,const float* ang,float* hist,int W,int H,int Cx){
  int cx=blockIdx.x, cy=blockIdx.y, lane=threadIdx.x;
  int gx0=cx*8, gy0=cy*8;
  float bins[9]={0};
  for(int k=0;k<2;++k){
    int p=lane+k*32; if(p<64){
      int lx=p&7, ly=p>>3;
      float m=mag[(gy0+ly)*W+gx0+lx];
      float a=ang[(gy0+ly)*W+gx0+lx];
      int b=int((a*57.2957795f)/20.f); if(b>8)b=8;
      bins[b]+=m;
    }
  }
  for(int b=0;b<9;++b){
    float v=wSum(bins[b]);
    if(lane==0) hist[(cy*Cx+cx)*9+b]=v;
  }
}

// Block kernel
__global__ void blockKer(const float* cell,float* desc,int Cx,int Bx,int By){
  int tid=blockIdx.x*blockDim.x+threadIdx.x; if(tid>=Bx*By) return;
  int by=tid/Bx, bx=tid-by*Bx;
  float v[36];
#pragma unroll
  for(int cy=0;cy<2;++cy)
    for(int cx=0;cx<2;++cx){
      const float* h=&cell[((by+cy)*Cx+(bx+cx))*9];
#pragma unroll
      for(int b=0;b<9;++b) v[(cy*2+cx)*9+b]=h[b];
    }
  const float eps=1e-6f,clip=0.2f;
  float n=eps;
#pragma unroll
  for(int i=0;i<36;++i) n+=v[i]*v[i];
  n=sqrtf(n);
#pragma unroll
  for(int i=0;i<36;++i){ v[i]/=n; if(v[i]>clip) v[i]=clip; }
  n=eps;
#pragma unroll
  for(int i=0;i<36;++i) n+=v[i]*v[i];
  n=sqrtf(n);
#pragma unroll
  for(int i=0;i<36;++i) desc[tid*36+i]=v[i]/n;
}

// RMSE
double rmse(const std::vector<float>& a,const std::vector<float>& b){
  double s=0; for(size_t i=0;i<a.size();++i){ double d=a[i]-b[i]; s+=d*d; }
  return std::sqrt(s/a.size());
}

int main(int argc,char** argv){
  int W=4096,H=4096;
  std::vector<uint8_t> img;
  if(argc==2){
    if(!readPGM(argv[1],img,W,H)){ std::cerr<<"PGM fail\n"; return 1; }
  }else{
    img.resize(size_t(W)*H); for(auto& p:img) p=rand()&0xFF;
  }
  size_t N=size_t(W)*H;
  int Cx=W/8,Cy=H/8,Bx=Cx-1,By=Cy-1;

  // CPU reference
  std::vector<float> magC,angC,cellC,descC;
  auto c0=std::chrono::high_resolution_clock::now();
  sobelCPU(img,W,H,magC,angC);
  cellCPU(magC,angC,W,H,cellC);
  blockCPU(cellC,Cx,Cy,descC);
  auto c1=std::chrono::high_resolution_clock::now();
  double cpu_ms=ms(c0,c1);

  // Host pinned
  uint8_t* hIn;   CUDA_CHECK(cudaHostAlloc(&hIn,N,cudaHostAllocDefault));
  float  * hDesc; CUDA_CHECK(cudaHostAlloc(&hDesc,Bx*By*36*4,cudaHostAllocDefault));
  std::memcpy(hIn,img.data(),N);

  uint8_t *d_img; CUDA_CHECK(cudaMalloc(&d_img,N));
  float *d_mag,*d_ang; CUDA_CHECK(cudaMalloc(&d_mag,N*4)); CUDA_CHECK(cudaMalloc(&d_ang,N*4));
  float *d_cell; CUDA_CHECK(cudaMalloc(&d_cell,size_t(Cx)*Cy*9*4));
  float *d_desc; CUDA_CHECK(cudaMalloc(&d_desc,Bx*By*36*4));

  // Streams & events
  cudaStream_t s0,s1; CUDA_CHECK(cudaStreamCreate(&s0)); CUDA_CHECK(cudaStreamCreate(&s1));
  cudaEvent_t eH,eG,eC,eB,eD; for(auto ev:{&eH,&eG,&eC,&eB,&eD}) CUDA_CHECK(cudaEventCreate(ev));

  // H2D
  CUDA_CHECK(cudaMemcpyAsync(d_img,hIn,N,cudaMemcpyHostToDevice,s0));
  CUDA_CHECK(cudaEventRecord(eH,s0));

  // Grad
  dim3 blkG(16,16), grdG((W+15)/16,(H+15)/16); size_t smem=(16+2)*(16+2);
  sobelKer<<<grdG,blkG,smem,s0>>>(d_img,d_mag,d_ang,W,H);
  CUDA_CHECK(cudaEventRecord(eG,s0));

  // Cell
  CUDA_CHECK(cudaStreamWaitEvent(s1,eG,0));
  dim3 grdC(Cx,Cy);
  cellKer<<<grdC,32,0,s1>>>(d_mag,d_ang,d_cell,W,H,Cx);
  CUDA_CHECK(cudaEventRecord(eC,s1));

  // Block
  CUDA_CHECK(cudaStreamWaitEvent(s0,eC,0));
  int threads=256, grid=(Bx*By+threads-1)/threads;
  blockKer<<<grid,threads,0,s0>>>(d_cell,d_desc,Cx,Bx,By);
  CUDA_CHECK(cudaEventRecord(eB,s0));

  // D2H
  CUDA_CHECK(cudaStreamWaitEvent(s1,eB,0));
  CUDA_CHECK(cudaMemcpyAsync(hDesc,d_desc,Bx*By*36*4,cudaMemcpyDeviceToHost,s1));
  CUDA_CHECK(cudaEventRecord(eD,s1));
  CUDA_CHECK(cudaEventSynchronize(eD));

  // Timings
  auto evms=[&](cudaEvent_t a,cudaEvent_t b){ float t; CUDA_CHECK(cudaEventElapsedTime(&t,a,b)); return double(t); };
  double tH2D=evms(eH,eG), tK=evms(eG,eB), tD2H=evms(eB,eD), tTot=evms(eH,eD);

  std::vector<float> descG(Bx*By*36); std::memcpy(descG.data(),hDesc,descG.size()*4);
  double err=rmse(descC,descG);

  std::cout<<"CPU total               : "<<cpu_ms<<" ms\n";
  std::cout<<"GPU H2D copy            : "<<tH2D <<" ms\n";
  std::cout<<"GPU kernels             : "<<tK   <<" ms\n";
  std::cout<<"GPU D2H copy            : "<<tD2H <<" ms\n";
  std::cout<<"GPU total               : "<<tTot <<" ms\n";
  std::cout<<"Speed-up (CPU/GPU)      : "<<cpu_ms/tTot<<"×\n";
  std::cout<<"RMSE descriptor         : "<<err  <<"\n";

  // Cleanup
  CUDA_CHECK(cudaFree(d_img)); CUDA_CHECK(cudaFree(d_mag)); CUDA_CHECK(cudaFree(d_ang));
  CUDA_CHECK(cudaFree(d_cell)); CUDA_CHECK(cudaFree(d_desc));
  CUDA_CHECK(cudaFreeHost(hIn)); CUDA_CHECK(cudaFreeHost(hDesc));
  CUDA_CHECK(cudaStreamDestroy(s0)); CUDA_CHECK(cudaStreamDestroy(s1));
  return 0;
}
