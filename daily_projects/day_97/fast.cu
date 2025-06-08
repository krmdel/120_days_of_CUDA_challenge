#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <chrono>
#include <fstream>
#include <iostream>
#include <vector>
#include <string>

#define CUDA_CHECK(x)                                                             \
  do { cudaError_t e=(x); if(e!=cudaSuccess){                                     \
         std::cerr<<"CUDA error: "<<cudaGetErrorString(e)<<" ("<<__FILE__<<':'    \
                  <<__LINE__<<")\n"; std::exit(1);} } while(0)

template<typename TP>
inline double msec(TP a, TP b){
  return std::chrono::duration<double,std::milli>(b-a).count(); }

// Minimal PGM loader
bool readPGM(const char* fn,std::vector<uint8_t>& img,int& W,int& H){
  std::ifstream f(fn,std::ios::binary); if(!f) return false;
  std::string m; f>>m; if(m!="P5") return false;
  int maxv; f>>W>>H>>maxv; f.get(); if(maxv!=255) return false;
  img.resize(size_t(W)*H);
  f.read(reinterpret_cast<char*>(img.data()),img.size());
  return bool(f);
}

// Constants for circle offsets & 9-bit masks
__constant__ int8_t d_dx[16];
__constant__ int8_t d_dy[16];
__constant__ uint16_t d_mask9[16];

// CPU reference FAST-9
inline bool isFASTCornerCPU(const uint8_t* img,int x,int y,int W,int t){
  int c = img[y*W+x];
  uint16_t bright=0,dark=0;
  constexpr int8_t dx[16]={0,1,2,2,2,1,0,-1,-2,-2,-2,-1,0,1,2,1};
  constexpr int8_t dy[16]={-3,-3,-2,-1,0,1,2,2,2,1,0,-1,-2,-3,-2,-1};
  for(int i=0;i<16;++i){
    int p = img[(y+dy[i])*W + x+dx[i]];
    bright |= (p>=c+t)<<i;
    dark   |= (p<=c-t)<<i;
  }
  for(int s=0;s<16;++s){
    uint16_t m = ((1u<<9)-1u);
    uint16_t rot = (m<<s)|(m>>(16-s));
    if((bright & rot) == rot) return true;
    if((dark   & rot) == rot) return true;
  }
  return false;
}
static void fastCPU(const std::vector<uint8_t>& img,
                    std::vector<uint8_t>& corner,
                    int W,int H,int t){
  corner.assign(size_t(W)*H,0);
  for(int y=3;y<H-3;++y)
    for(int x=3;x<W-3;++x)
      corner[y*W+x] = isFASTCornerCPU(img.data(),x,y,W,t);
}

// GPU kernel FAST-9 + warp ballot compaction
__global__ void fastKernel(const uint8_t* __restrict__ img,
                           uint8_t* __restrict__ out,
                           int W,int H,int t)
{
  int x  = blockIdx.x*32 + threadIdx.x;        // 32-wide warp block
  int y  = blockIdx.y;                         // one row per block
  int idx= y*W + x;
  bool isCorner=false;

  if(x>=3 && x<W-3 && y>=3 && y<H-3){
    int c = img[idx];
    uint16_t bright=0,dark=0;
    #pragma unroll
    for(int i=0;i<16;++i){
      int p = img[(y+d_dy[i])*W + x+d_dx[i]];
      bright |= (p>=c+t)<<i;
      dark   |= (p<=c-t)<<i;
    }
    #pragma unroll
    for(int s=0;s<16;++s){
      uint16_t mask=d_mask9[s];
      if((bright&mask)==mask || (dark&mask)==mask){ isCorner=true; break; }
    }
  }

  // Warp ballot: pack 32 flags, store by lane 0
  unsigned warpMask = __ballot_sync(0xffffffff,isCorner);
  if((threadIdx.x&31)==0){
    int base   = idx - threadIdx.x;  // x back to warp start
    for(int bit=0;bit<32 && (base+bit)<W*H;++bit)
      out[base+bit] = (warpMask>>bit)&1;
  }
}

int main(int argc,char**argv)
{
  int W=4096,H=4096,th=20;
  if(argc>=3) th=std::stoi(argv[2]);
  std::vector<uint8_t> img;
  if(argc>=2 && readPGM(argv[1],img,W,H))
    std::cout<<"Loaded "<<W<<"×"<<H<<" PGM,  threshold="<<th<<"\n";
  else{
    img.resize(size_t(W)*H); std::srand(1234);
    for(auto& p:img) p=std::rand()&0xFF;
    std::cout<<"Random "<<W<<"×"<<H<<" image, threshold="<<th<<"\n";
  }
  size_t N=size_t(W)*H;

  // Pre-compute masks & copy to constant memory
  int8_t hx[16]={0,1,2,2,2,1,0,-1,-2,-2,-2,-1,0,1,2,1};
  int8_t hy[16]={-3,-3,-2,-1,0,1,2,2,2,1,0,-1,-2,-3,-2,-1};
  uint16_t mask9[16];
  for(int s=0;s<16;++s){
    uint16_t m=0;
    for(int k=0;k<9;++k) m |= 1u<<((s+k)&15);
    mask9[s]=m;
  }
  CUDA_CHECK(cudaMemcpyToSymbol(d_dx,hx,sizeof(hx)));
  CUDA_CHECK(cudaMemcpyToSymbol(d_dy,hy,sizeof(hy)));
  CUDA_CHECK(cudaMemcpyToSymbol(d_mask9,mask9,sizeof(mask9)));

  // CPU reference
  std::vector<uint8_t> cornerCPU;
  auto c0=std::chrono::high_resolution_clock::now();
  fastCPU(img,cornerCPU,W,H,th);
  auto c1=std::chrono::high_resolution_clock::now();
  double cpu_ms=msec(c0,c1);

  // GPU buffers
  uint8_t *d_img,*d_out;
  CUDA_CHECK(cudaMalloc(&d_img,N));
  CUDA_CHECK(cudaMalloc(&d_out,N));

  // Events
  cudaEvent_t beg,end,h2d0,h2d1,k0,k1,d2h0,d2h1;
  CUDA_CHECK(cudaEventCreate(&beg));CUDA_CHECK(cudaEventCreate(&end));
  CUDA_CHECK(cudaEventCreate(&h2d0));CUDA_CHECK(cudaEventCreate(&h2d1));
  CUDA_CHECK(cudaEventCreate(&k0));CUDA_CHECK(cudaEventCreate(&k1));
  CUDA_CHECK(cudaEventCreate(&d2h0));CUDA_CHECK(cudaEventCreate(&d2h1));

  CUDA_CHECK(cudaEventRecord(beg));

  // H2D
  CUDA_CHECK(cudaEventRecord(h2d0));
  CUDA_CHECK(cudaMemcpy(d_img,img.data(),N,cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaEventRecord(h2d1));

  // Kernel launch: 32 threads × (H rows) blocks
  dim3 blk(32,1), grd((W+31)/32,H);
  CUDA_CHECK(cudaEventRecord(k0));
  fastKernel<<<grd,blk>>>(d_img,d_out,W,H,th);
  CUDA_CHECK(cudaEventRecord(k1));

  // D2H
  std::vector<uint8_t> cornerGPU(N);
  CUDA_CHECK(cudaEventRecord(d2h0));
  CUDA_CHECK(cudaMemcpy(cornerGPU.data(),d_out,N,cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaEventRecord(d2h1));

  CUDA_CHECK(cudaEventRecord(end));
  CUDA_CHECK(cudaEventSynchronize(end));

  // Timings
  auto evms=[&](cudaEvent_t a,cudaEvent_t b){ float m;CUDA_CHECK(cudaEventElapsedTime(&m,a,b)); return double(m); };
  double h2d=evms(h2d0,h2d1), ker=evms(k0,k1), d2h=evms(d2h0,d2h1), tot=evms(beg,end);

  // Accuracy
  size_t diff=0; for(size_t i=0;i<N;++i) if(cornerGPU[i]!=cornerCPU[i]) ++diff;
  double err=100.0*diff/N;

  // Timing
  std::cout<<"CPU total               : "<<cpu_ms<<" ms\n";
  std::cout<<"GPU H2D copy            : "<<h2d    <<" ms\n";
  std::cout<<"GPU kernel              : "<<ker    <<" ms\n";
  std::cout<<"GPU D2H copy            : "<<d2h    <<" ms\n";
  std::cout<<"GPU total               : "<<tot    <<" ms\n";
  std::cout<<"Pixel disagreement      : "<<err    <<" %\n";

  CUDA_CHECK(cudaFree(d_img)); CUDA_CHECK(cudaFree(d_out));
  return 0;
}
