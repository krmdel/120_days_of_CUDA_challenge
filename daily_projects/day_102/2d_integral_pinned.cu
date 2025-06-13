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
#include <cstring>           // ::memcpy

#define CUDA_CHECK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){                 \
  std::cerr<<"CUDA error: "<<cudaGetErrorString(e)<<" ("                         \
           <<__FILE__<<':'<<__LINE__<<")\n"; std::exit(1);} }while(0)

template<class TP> inline double ms(TP a,TP b){
  return std::chrono::duration<double,std::milli>(b-a).count(); }

// PGM loader
bool readPGM(const char* fn,std::vector<uint8_t>& img,int& W,int& H){
  std::ifstream f(fn,std::ios::binary); if(!f) return false;
  std::string m; f>>m; int maxv; if(m!="P5") return false;
  f>>W>>H>>maxv; f.get(); if(maxv!=255) return false;
  img.resize(size_t(W)*H); f.read(reinterpret_cast<char*>(img.data()),img.size());
  return bool(f);
}

// CPU reference
void integralCPU(const std::vector<uint8_t>& src,std::vector<uint32_t>& dst,
                 int W,int H){
  dst.resize(size_t(W)*H);
  for(int y=0;y<H;++y){
    uint32_t run=0;
    for(int x=0;x<W;++x){
      run+=src[y*W+x];
      dst[y*W+x]=run+(y?dst[(y-1)*W+x]:0);
    }
  }
}

// Transpose (32×32)
__global__ void transpose32u(const uint32_t* __restrict__ in,
                             uint32_t*       __restrict__ out,
                             int W,int H){
  __shared__ uint32_t tile[32][33];
  int x=blockIdx.x*32+threadIdx.x, y=blockIdx.y*32+threadIdx.y;
  if(x<W&&y<H) tile[threadIdx.y][threadIdx.x]=in[y*W+x];
  __syncthreads();
  x=blockIdx.y*32+threadIdx.x; y=blockIdx.x*32+threadIdx.y;
  if(x<H&&y<W) out[y*H+x]=tile[threadIdx.x][threadIdx.y];
}

// Row-tile scans
constexpr int TILE=1024;

// uint8 → uint32
__global__ void scan_u8(const uint8_t* __restrict__ in,uint32_t* __restrict__ out,
                        uint32_t* __restrict__ sums,int W,int tilesPer){
  __shared__ uint32_t s[TILE];
  int row=blockIdx.y, tile=blockIdx.x;
  int gx=tile*TILE+threadIdx.x, idx=row*W+gx;
  s[threadIdx.x]=(gx<W)?uint32_t(in[idx]):0u; __syncthreads();
#pragma unroll
  for(int o=1;o<TILE;o<<=1){
    uint32_t v=(threadIdx.x>=o)?s[threadIdx.x-o]:0u;
    __syncthreads(); s[threadIdx.x]+=v; __syncthreads();
  }
  if(gx<W) out[idx]=s[threadIdx.x];
  if(threadIdx.x==TILE-1||gx==W-1) sums[row*tilesPer+tile]=s[threadIdx.x];
}

// uint32 → uint32
__global__ void scan_u32(uint32_t* buf,uint32_t* sums,int W,int tilesPer){
  __shared__ uint32_t s[TILE];
  int row=blockIdx.y, tile=blockIdx.x;
  int gx=tile*TILE+threadIdx.x, idx=row*W+gx;
  s[threadIdx.x]=(gx<W)?buf[idx]:0u; __syncthreads();
#pragma unroll
  for(int o=1;o<TILE;o<<=1){
    uint32_t v=(threadIdx.x>=o)?s[threadIdx.x-o]:0u;
    __syncthreads(); s[threadIdx.x]+=v; __syncthreads();
  }
  if(gx<W) buf[idx]=s[threadIdx.x];
  if(threadIdx.x==TILE-1||gx==W-1) sums[row*tilesPer+tile]=s[threadIdx.x];
}

// Prefix over tile endings
__global__ void scanSums(uint32_t* sums,int tilesPer,int rows){
  int row=blockIdx.x; if(row>=rows) return;
  uint32_t acc=0;
  for(int t=0;t<tilesPer;++t){
    uint32_t v=sums[row*tilesPer+t];
    sums[row*tilesPer+t]=acc; acc+=v;
  }
}

// Add offsets
__global__ void addOff(uint32_t* img,const uint32_t* sums,int W,int tilesPer){
  int row=blockIdx.y, tile=blockIdx.x;
  uint32_t off=sums[row*tilesPer+tile];
  int gx=tile*TILE+threadIdx.x; if(gx>=W) return;
  img[row*W+gx]+=off;
}

int main(int argc,char**argv)
{
  int W=4096,H=4096;
  std::vector<uint8_t> src;
  if(argc==2){ if(!readPGM(argv[1],src,W,H)){std::cerr<<"PGM fail\n";return 1;}
    std::cout<<"Loaded "<<W<<"×"<<H<<"\n";}
  else { src.resize(size_t(W)*H); std::srand(1234);
    for(auto& p:src) p=std::rand()&0xFF; std::cout<<"Random "<<W<<"×"<<H<<"\n"; }

  // Pinned host buffers
  uint8_t  *h_in ; CUDA_CHECK(cudaMallocHost(&h_in ,src.size()));
  uint32_t *h_out; CUDA_CHECK(cudaMallocHost(&h_out,src.size()*4));
  std::memcpy(h_in, src.data(), src.size());

  // CPU baseline
  std::vector<uint32_t> ref;
  auto c0=std::chrono::high_resolution_clock::now();
  integralCPU(src,ref,W,H);
  auto c1=std::chrono::high_resolution_clock::now();
  double cpu_ms = ms(c0,c1);

  // Device buffers
  uint8_t  *d_src; CUDA_CHECK(cudaMalloc(&d_src,src.size()));
  uint32_t *d_tmp; CUDA_CHECK(cudaMalloc(&d_tmp,src.size()*4));
  uint32_t *d_tr ; CUDA_CHECK(cudaMalloc(&d_tr ,src.size()*4));
  uint32_t *d_int; CUDA_CHECK(cudaMalloc(&d_int,src.size()*4));

  int tilesR=(W+TILE-1)/TILE;
  uint32_t* d_sums; CUDA_CHECK(cudaMalloc(&d_sums,size_t(tilesR)*H*sizeof(uint32_t)));

  // Stream + events
  cudaStream_t s; CUDA_CHECK(cudaStreamCreate(&s));
  cudaEvent_t eb,ee,h0,h1,k0,k1,d0,d1;
  CUDA_CHECK(cudaEventCreate(&eb)); CUDA_CHECK(cudaEventCreate(&ee));
  CUDA_CHECK(cudaEventCreate(&h0)); CUDA_CHECK(cudaEventCreate(&h1));
  CUDA_CHECK(cudaEventCreate(&k0)); CUDA_CHECK(cudaEventCreate(&k1));
  CUDA_CHECK(cudaEventCreate(&d0)); CUDA_CHECK(cudaEventCreate(&d1));

  CUDA_CHECK(cudaEventRecord(eb));

  // H2D async
  CUDA_CHECK(cudaEventRecord(h0,s));
  CUDA_CHECK(cudaMemcpyAsync(d_src,h_in,src.size(),cudaMemcpyHostToDevice,s));
  CUDA_CHECK(cudaEventRecord(h1,s));
  CUDA_CHECK(cudaStreamSynchronize(s));                      // wait for data

  // Kernels
  CUDA_CHECK(cudaEventRecord(k0,s));

  dim3 blk(TILE,1), grd(tilesR,H);
  scan_u8<<<grd,blk,0,s>>>(d_src,d_tmp,d_sums,W,tilesR);
  scanSums<<<H,1,0,s>>>(d_sums,tilesR,H);
  addOff  <<<grd,blk,0,s>>>(d_tmp,d_sums,W,tilesR);

  dim3 blkT(32,32), grdT((W+31)/32,(H+31)/32);
  transpose32u<<<grdT,blkT,0,s>>>(d_tmp,d_tr,W,H);

  int tilesR2=(H+TILE-1)/TILE;
  CUDA_CHECK(cudaFree(d_sums));
  CUDA_CHECK(cudaMalloc(&d_sums,size_t(tilesR2)*W*sizeof(uint32_t)));

  dim3 grd2(tilesR2,W);
  scan_u32<<<grd2,blk,0,s>>>(d_tr,d_sums,H,tilesR2);
  scanSums <<<W,1,0,s>>>(d_sums,tilesR2,W);
  addOff   <<<grd2,blk,0,s>>>(d_tr,d_sums,H,tilesR2);

  dim3 grdT2((H+31)/32,(W+31)/32);
  transpose32u<<<grdT2,blkT,0,s>>>(d_tr,d_int,H,W);

  CUDA_CHECK(cudaEventRecord(k1,s));

  // D2H async
  CUDA_CHECK(cudaEventRecord(d0,s));
  CUDA_CHECK(cudaMemcpyAsync(h_out,d_int,src.size()*4,cudaMemcpyDeviceToHost,s));
  CUDA_CHECK(cudaEventRecord(d1,s));

  CUDA_CHECK(cudaEventRecord(ee,s));
  CUDA_CHECK(cudaStreamSynchronize(s));

  // Timings
  auto dt=[&](cudaEvent_t a,cudaEvent_t b){ float t;CUDA_CHECK(cudaEventElapsedTime(&t,a,b)); return double(t); };
  double tH2D=dt(h0,h1), tKer=dt(k0,k1), tD2H=dt(d0,d1), tTot=dt(eb,ee);

  // RMSE
  double mse=0; for(size_t i=0;i<src.size();++i){ double diff=double(h_out[i])-double(ref[i]); mse+=diff*diff; }
  double rmse=std::sqrt(mse/src.size());

  std::cout<<"CPU total               : "<<cpu_ms<<" ms\n";
  std::cout<<"GPU H2D copy            : "<<tH2D  <<" ms\n";
  std::cout<<"GPU kernels             : "<<tKer  <<" ms\n";
  std::cout<<"GPU D2H copy            : "<<tD2H  <<" ms\n";
  std::cout<<"GPU total               : "<<tTot  <<" ms\n";
  std::cout<<"RMSE                    : "<<rmse  <<"\n";

  // Cleanup
  CUDA_CHECK(cudaFree(d_src)); CUDA_CHECK(cudaFree(d_tmp));
  CUDA_CHECK(cudaFree(d_tr )); CUDA_CHECK(cudaFree(d_int));
  CUDA_CHECK(cudaFree(d_sums));
  CUDA_CHECK(cudaFreeHost(h_in)); CUDA_CHECK(cudaFreeHost(h_out));
  CUDA_CHECK(cudaStreamDestroy(s));
  return 0;
}