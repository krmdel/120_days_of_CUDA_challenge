#include <cuda_runtime.h>
#include <vector>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <chrono>
#include <random>
#include <cstring>
#include <algorithm>

#define CUDA_CHECK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){               \
  std::cerr<<"CUDA error: "<<cudaGetErrorString(e)<<" ("<<__LINE__<<")\n";     \
  std::exit(1);} }while(0)
template<class T> inline double ms(T a,T b){
  return std::chrono::duration<double,std::milli>(b-a).count(); }

// PGM loader (if no OpenCV)
static bool readPGM(const char* fn,std::vector<uint8_t>& img,int& W,int& H){
  FILE* fp=fopen(fn,"rb"); if(!fp) return false;
  char m[3]; if(fscanf(fp,"%2s",m)!=1||strcmp(m,"P5")){fclose(fp);return false;}
  int maxv; if(fscanf(fp,"%d%d%d",&W,&H,&maxv)!=3||maxv!=255){fclose(fp);return false;}
  fgetc(fp);
  img.resize(size_t(W)*H);
  size_t rd=fread(img.data(),1,img.size(),fp);
  fclose(fp); return rd==img.size();
}

constexpr int WIN_R    = 3;     // 7×7 census
constexpr int MAX_DISP = 64;    // disparity search range
using u64               = unsigned long long;

// 1. Census transform kernel
__global__ void censusKer(const uint8_t* img,u64* cen,int W,int H)
{
  int x=blockIdx.x*blockDim.x+threadIdx.x;
  int y=blockIdx.y*blockDim.y+threadIdx.y;
  if(x<WIN_R||y<WIN_R||x>=W-WIN_R||y>=H-WIN_R){ cen[y*W+x]=0; return; }

  uint8_t c = img[y*W+x];
  u64 sig=0; int bit=0;
  for(int dy=-WIN_R;dy<=WIN_R;++dy)
    for(int dx=-WIN_R;dx<=WIN_R;++dx){
      if(dx==0&&dy==0) continue;
      sig |= (u64)(img[(y+dy)*W + (x+dx)] < c) << bit++;
    }
  cen[y*W+x]=sig;
}

// 2. Block-matching kernel
__global__ void matchKer(const u64* cenL,const u64* cenR,
                         uint8_t* disp,int W,int H,int maxD)
{
  int x=blockIdx.x*blockDim.x+threadIdx.x;
  int y=blockIdx.y*blockDim.y+threadIdx.y;
  if(x>=W||y>=H){ return; }

  u64 cL = cenL[y*W+x];
  int bestD = 0, bestCost = 64;

  #pragma unroll
  for(int d=0;d<=maxD;++d){
    int xr = x - d;
    if(xr<0) break;
    int cost = __popcll(cL ^ cenR[y*W + xr]);
    if(cost < bestCost){ bestCost = cost; bestD = d; }
  }
  disp[y*W+x] = uint8_t(bestD);
}

// CPU reference (scalar, census + match)
static inline u64 censusCPU(const std::vector<uint8_t>& im,int W,int H,int x,int y){
  uint8_t c = im[y*W+x]; u64 sig=0; int bit=0;
  for(int dy=-WIN_R;dy<=WIN_R;++dy)
    for(int dx=-WIN_R;dx<=WIN_R;++dx){
      if(dx==0&&dy==0) continue;
      sig |= (u64)(im[(y+dy)*W+(x+dx)] < c) << bit++;
    }
  return sig;
}
static void stereoCPU(const std::vector<uint8_t>& L,
                      const std::vector<uint8_t>& R,
                      int W,int H,int maxD,
                      std::vector<uint8_t>& D)
{
  D.resize(size_t(W)*H);
  std::vector<u64> cenL(W*H),cenR(W*H);
  for(int y=0;y<H;++y) for(int x=0;x<W;++x){
    cenL[y*W+x]=censusCPU(L,W,H,x,y);
    cenR[y*W+x]=censusCPU(R,W,H,x,y);
  }
  for(int y=0;y<H;++y) for(int x=0;x<W;++x){
    u64 cL=cenL[y*W+x]; int best=0,cost=64;
    for(int d=0;d<=maxD;++d){
      int xr=x-d; if(xr<0) break;
      int c=__builtin_popcountll(cL ^ cenR[y*W+xr]);
      if(c<cost){ cost=c; best=d; }
    }
    D[y*W+x]=uint8_t(best);
  }
}

int main(int argc,char** argv)
{
  int W,H; std::vector<uint8_t> left,right;
  if(argc>=3){
    if(!readPGM(argv[1],left,W,H)||!readPGM(argv[2],right,W,H)){
      std::cerr<<"PGM load error\n"; return 1;}
  }else{
    W=4096; H=4096;
    left.resize(size_t(W)*H); right.resize(left.size());
    std::mt19937 rng(0); std::uniform_int_distribution<int>d(0,255);
    for(auto& v:left) v=d(rng);
    for(auto& v:right) v=d(rng);
    std::cout<<"[Info] random stereo pair "<<W<<"×"<<H<<"\n";
  }
  const int N=W*H;

  // CPU reference
  auto tc0=std::chrono::high_resolution_clock::now();
  std::vector<uint8_t> dispCPU; stereoCPU(left,right,W,H,MAX_DISP,dispCPU);
  auto tc1=std::chrono::high_resolution_clock::now();
  double cpu_ms = ms(tc0,tc1);

  // GPU memory
  uint8_t *dL,*dR,*dDisp;   u64 *dCenL,*dCenR;
  CUDA_CHECK(cudaMalloc(&dL,N));
  CUDA_CHECK(cudaMalloc(&dR,N));
  CUDA_CHECK(cudaMalloc(&dDisp,N));
  CUDA_CHECK(cudaMalloc(&dCenL,N*sizeof(u64)));
  CUDA_CHECK(cudaMalloc(&dCenR,N*sizeof(u64)));

  cudaEvent_t h0,h1,c0,c1,m0,m1,t0,t1;
  for(auto p:{&h0,&h1,&c0,&c1,&m0,&m1,&t0,&t1}) CUDA_CHECK(cudaEventCreate(p));

  cudaEventRecord(t0);

  cudaEventRecord(h0);
  CUDA_CHECK(cudaMemcpy(dL,left.data(),N,cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dR,right.data(),N,cudaMemcpyHostToDevice));
  cudaEventRecord(h1);

  dim3 blk(16,16), grd((W+15)/16,(H+15)/16);

  cudaEventRecord(c0);
  censusKer<<<grd,blk>>>(dL,dCenL,W,H);
  censusKer<<<grd,blk>>>(dR,dCenR,W,H);
  cudaEventRecord(c1);

  cudaEventRecord(m0);
  matchKer<<<grd,blk>>>(dCenL,dCenR,dDisp,W,H,MAX_DISP);
  cudaEventRecord(m1);

  std::vector<uint8_t> dispGPU(N);
  CUDA_CHECK(cudaEventSynchronize(m1));
  cudaEventRecord(t1);

  CUDA_CHECK(cudaMemcpy(dispGPU.data(),dDisp,N,cudaMemcpyDeviceToHost));

  // Timings
  float tH2D,tCen,tMatch,gpuTot;
  cudaEventElapsedTime(&tH2D ,h0,h1);
  cudaEventElapsedTime(&tCen ,c0,c1);
  cudaEventElapsedTime(&tMatch,m0,m1);
  cudaEventElapsedTime(&gpuTot,t0,t1);

  std::cout<<"\nImage size              : "<<W<<" × "<<H<<"\n";
  std::cout<<"Search range            : 0-"<<MAX_DISP<<" px\n\n";
  std::cout<<"GPU H2D copy            : "<<tH2D  <<" ms\n";
  std::cout<<"GPU census kernels      : "<<tCen  <<" ms\n";
  std::cout<<"GPU match kernel        : "<<tMatch<<" ms\n";
  std::cout<<"GPU total               : "<<gpuTot<<" ms\n";
  std::cout<<"CPU total               : "<<cpu_ms <<" ms\n";

  // Cleanup
  CUDA_CHECK(cudaFree(dL)); CUDA_CHECK(cudaFree(dR));
  CUDA_CHECK(cudaFree(dDisp)); CUDA_CHECK(cudaFree(dCenL)); CUDA_CHECK(cudaFree(dCenR));
  return 0;
}
