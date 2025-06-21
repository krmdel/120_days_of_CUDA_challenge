#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <chrono>
#include <fstream>
#include <iostream>

#define CUDA_CHECK(x)  do{ cudaError_t e=(x); if(e!=cudaSuccess){                \
  std::cerr<<"CUDA error: "<<cudaGetErrorString(e)<<" ("<<__FILE__<<':'          \
           <<__LINE__<<")\n"; std::exit(1);} }while(0)
template<class TP> double ms(TP a,TP b){
  return std::chrono::duration<double,std::milli>(b-a).count(); }

// PGM loader
bool readPGM(const char*fn,std::vector<uint8_t>& img,int& W,int& H){
  std::ifstream f(fn,std::ios::binary); if(!f) return false;
  std::string m;int maxv;f>>m>>W>>H>>maxv;f.get();
  if(m!="P5"||maxv!=255) return false;
  img.resize(size_t(W)*H);
  f.read(reinterpret_cast<char*>(img.data()),img.size());
  return bool(f);
}

// CUDA warp-histogram kernel
__device__ __forceinline__ float warpReduceSum(float v){
  #pragma unroll
  for(int off=16;off;off>>=1)
    v += __shfl_down_sync(0xffffffff,v,off);
  return v;
}

__global__ void cellHistKernel(const float* __restrict__ mag,
                               const float* __restrict__ ang,
                               float* __restrict__ hist,
                               int W,int H,int cellsX)
{
  const int cellX = blockIdx.x;
  const int cellY = blockIdx.y;
  const int lane  = threadIdx.x;        // 0-31
  const int pix0  = lane;               // first pixel idx (0-63)
  const int baseX = cellX*8, baseY = cellY*8;
  float h[9]={0};

  // Process up to two pixels per thread
  #pragma unroll
  for(int k=0;k<2;++k){
    int p = pix0 + k*32;                // 0..63
    if(p<64){
      int lx = p & 7;
      int ly = p >> 3;                  // /8
      int gx = baseX + lx;
      int gy = baseY + ly;
      float m = mag[gy*W + gx];
      float a = ang[gy*W + gx];         // radians (0..π)
      int bin = int((a*57.295779513f) / 20.f); // 0..8  (π≈180°)
      bin = (bin>8)?8:bin;
      h[bin] += m;
    }
  }

  // Warp reduce 9 scalars
  for(int b=0;b<9;++b){
    float v = warpReduceSum(h[b]);
    if(lane==0) hist[(cellY*cellsX + cellX)*9 + b] = v;
  }
}

// CPU reference cell histogram
void cpuCellHist(const std::vector<float>& mag,const std::vector<float>& ang,
                 int W,int H,std::vector<float>& hist)
{
  int cellsX=W/8,cellsY=H/8;
  hist.assign(size_t(cellsX)*cellsY*9,0.f);
  for(int cy=0;cy<cellsY;++cy)
    for(int cx=0;cx<cellsX;++cx){
      float* h=&hist[(cy*cellsX+cx)*9];
      for(int ly=0;ly<8;++ly)
        for(int lx=0;lx<8;++lx){
          int gx=cx*8+lx, gy=cy*8+ly;
          float m=mag[gy*W+gx], a=ang[gy*W+gx];
          int bin=int((a*57.295779513f)/20.f); bin=(bin>8)?8:bin;
          h[bin]+=m;
        }
    }
}

// RMSE
double rmse(const std::vector<float>& a,const std::vector<float>& b){
  double e=0; for(size_t i=0;i<a.size();++i){ double d=a[i]-b[i]; e+=d*d; }
  return std::sqrt(e/a.size());
}

int main(int argc,char**argv)
{
  int W=4096,H=4096;
  std::vector<uint8_t> img;
  if(argc==2){
    if(!readPGM(argv[1],img,W,H)){ std::cerr<<"PGM load error\n"; return 1; }
    std::cout<<"Loaded "<<W<<"×"<<H<<" PGM\n";
  }else{
    img.resize(size_t(W)*H);
    for(auto& p:img) p = rand() & 0xFF;
    std::cout<<"Random "<<W<<"×"<<H<<" image\n";
  }
  const size_t N = size_t(W)*H;

  // Synth gradient: magnitude = img, angle uniform random [0,π)
  std::vector<float> mag(N), ang(N);
  for(size_t i=0;i<N;++i){
    mag[i]=float(img[i]);
    ang[i]=float(rand())/RAND_MAX*3.14159265f;
  }

  // CPU reference
  std::vector<float> cpuHist;
  auto tc0=std::chrono::high_resolution_clock::now();
  cpuCellHist(mag,ang,W,H,cpuHist);
  auto tc1=std::chrono::high_resolution_clock::now();
  double cpu_ms = ms(tc0,tc1);

  // GPU buffers
  float *d_mag,*d_ang,*d_hist;
  CUDA_CHECK(cudaMalloc(&d_mag ,N*sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_ang ,N*sizeof(float)));
  int cellsX=W/8,cellsY=H/8;
  CUDA_CHECK(cudaMalloc(&d_hist,size_t(cellsX)*cellsY*9*sizeof(float)));

  cudaEvent_t h0,h1,k0,k1,d0,d1,beg,end;
  for(auto ev:{&h0,&h1,&k0,&k1,&d0,&d1,&beg,&end}) CUDA_CHECK(cudaEventCreate(ev));

  CUDA_CHECK(cudaEventRecord(beg));

  CUDA_CHECK(cudaEventRecord(h0));
  CUDA_CHECK(cudaMemcpy(d_mag,mag.data(),N*4,cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_ang,ang.data(),N*4,cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaEventRecord(h1));

  dim3 grid(cellsX,cellsY);
  CUDA_CHECK(cudaEventRecord(k0));
  cellHistKernel<<<grid,32,0>>>(d_mag,d_ang,d_hist,W,H,cellsX);
  CUDA_CHECK(cudaEventRecord(k1));

  std::vector<float> gpuHist(cellsX*cellsY*9);
  CUDA_CHECK(cudaEventRecord(d0));
  CUDA_CHECK(cudaMemcpy(gpuHist.data(),d_hist,gpuHist.size()*4,cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaEventRecord(d1));

  CUDA_CHECK(cudaEventRecord(end));
  CUDA_CHECK(cudaEventSynchronize(end));

  auto evms=[&](cudaEvent_t a,cudaEvent_t b){ float m; CUDA_CHECK(cudaEventElapsedTime(&m,a,b)); return double(m); };
  double h2d=evms(h0,h1), ker=evms(k0,k1), d2h=evms(d0,d1), tot=evms(beg,end);

  std::cout<<"CPU total               : "<<cpu_ms<<" ms\n";
  std::cout<<"GPU H2D copy            : "<<h2d    <<" ms\n";
  std::cout<<"GPU kernel              : "<<ker    <<" ms\n";
  std::cout<<"GPU D2H copy            : "<<d2h    <<" ms\n";
  std::cout<<"GPU total               : "<<tot    <<" ms\n";
  std::cout<<"RMSE  histograms        : "<<rmse(cpuHist,gpuHist)<<"\n";

  CUDA_CHECK(cudaFree(d_mag)); CUDA_CHECK(cudaFree(d_ang)); CUDA_CHECK(cudaFree(d_hist));
  return 0;
}
