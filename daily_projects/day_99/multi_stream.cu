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
#include <algorithm>            // std::max_element

// CUDA error macro
#define CUDA_CHECK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){                 \
  std::cerr<<"CUDA error: "<<cudaGetErrorString(e)<<" ("                         \
           <<__FILE__<<':'<<__LINE__<<")\n"; std::exit(1);} }while(0)

// Chrono helper
template<class TP>
inline double elapsed_ms(TP a, TP b) {
  return std::chrono::duration<double,std::milli>(b - a).count();
}

// Clamp helper
__host__ __device__ inline int clamp(int v,int lo,int hi){
  return v<lo?lo:(v>hi?hi:v);
}

// Raw float32 reader
bool readRawFloat(const char* fn,std::vector<float>& buf,size_t count){
  std::ifstream f(fn,std::ios::binary); if(!f) return false;
  buf.resize(count);
  f.read(reinterpret_cast<char*>(buf.data()),count*sizeof(float));
  return bool(f);
}

//  CPU stripe pipeline
static void cpuStripe(const float* Ix,const float* Iy,float* R,
                      int W,int H,int y0,int y1,float k,int rad=3)
{
  int Hs=y1-y0;
  std::vector<float> Ix2(W*Hs),Iy2(W*Hs),Ixy(W*Hs),
                     tmp(W*Hs), Ix2b(W*Hs),Iy2b(W*Hs),Ixyb(W*Hs);

  // Products
  for(int y=0;y<Hs;++y)
    for(int x=0;x<W;++x){
      float ix=Ix[(y0+y)*W+x], iy=Iy[(y0+y)*W+x];
      Ix2[y*W+x]=ix*ix; Iy2[y*W+x]=iy*iy; Ixy[y*W+x]=ix*iy;
    }

  auto blur7=[&](const std::vector<float>& in,std::vector<float>& out){
    // Horizontal
    for(int y=0;y<Hs;++y)
      for(int x=0;x<W;++x){
        float s=0.f;
        for(int k=-rad;k<=rad;++k)
          s+=in[y*W+clamp(x+k,0,W-1)];
        tmp[y*W+x]=s;
      }
    // Vertical
    for(int y=0;y<Hs;++y)
      for(int x=0;x<W;++x){
        float s=0.f;
        for(int k=-rad;k<=rad;++k)
          s+=tmp[clamp(y+k,0,Hs-1)*W+x];
        out[y*W+x]=s;
      }
  };
  blur7(Ix2,Ix2b); blur7(Iy2,Iy2b); blur7(Ixy,Ixyb);

  // Harris
  for(int y=0;y<Hs;++y)
    for(int x=0;x<W;++x){
      size_t i=y*W+x;
      float a=Ix2b[i], c=Iy2b[i], b=Ixyb[i];
      float det=a*c-b*b, tr=a+c;
      R[(y0+y)*W+x]=det - k*tr*tr;
    }
}

// GPU kernels
__constant__ float d_k;

__global__ void prodKernel(const float* Ix,const float* Iy,
                           float* Ix2,float* Iy2,float* Ixy,
                           int W)
{
  int x=blockIdx.x*blockDim.x+threadIdx.x;
  int y=blockIdx.y*blockDim.y+threadIdx.y;
  if(x>=W) return;
  int idx=y*W+x;
  float ix=Ix[idx], iy=Iy[idx];
  Ix2[idx]=ix*ix; Iy2[idx]=iy*iy; Ixy[idx]=ix*iy;
}

template<int R>
__global__ void blurH(const float* in,float* out,int W)
{
  int x=blockIdx.x*blockDim.x+threadIdx.x;
  int y=blockIdx.y;
  if(x>=W) return;
  float s=0.f;
#pragma unroll
  for(int k=-R;k<=R;++k)
    s+=in[y*W+clamp(x+k,0,W-1)];
  out[y*W+x]=s;
}

template<int R>
__global__ void blurV(const float* in,float* out,int W,int Hs)
{
  int x=blockIdx.x;
  int y=blockIdx.y*blockDim.y+threadIdx.y;
  if(y>=Hs||x>=W) return;
  float s=0.f;
#pragma unroll
  for(int k=-R;k<=R;++k)
    s+=in[clamp(y+k,0,Hs-1)*W+x];
  out[y*W+x]=s;
}

__global__ void harrisKernel(const float* Ix2b,const float* Iy2b,const float* Ixyb,
                             float* R,int N)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<N){
    float a=Ix2b[i], c=Iy2b[i], b=Ixyb[i];
    float det=a*c-b*b, tr=a+c;
    R[i]=det - d_k*tr*tr;
  }
}

int main(int argc,char**argv)
{
  const float kH=0.04f; const int CHUNK=256;   // stripe height
  int W=4096,H=4096;
  std::vector<float> Ix,Iy;

  if(argc==5){ W=std::stoi(argv[3]); H=std::stoi(argv[4]);
    if(!readRawFloat(argv[1],Ix,size_t(W)*H)||
       !readRawFloat(argv[2],Iy,size_t(W)*H)){
      std::cerr<<"Failed to read gradients\n"; return 1;}
    std::cout<<"Loaded "<<W<<"×"<<H<<" gradients\n";
  }else{
    Ix.resize(size_t(W)*H); Iy.resize(size_t(W)*H);
    std::srand(1234);
    for(size_t i=0;i<Ix.size();++i){
      Ix[i]=float((std::rand()&511)-255);
      Iy[i]=float((std::rand()&511)-255);
    }
    std::cout<<"Generated random gradients "<<W<<"×"<<H<<"\n";
  }
  const size_t N=size_t(W)*H;

  // CPU baseline
  std::vector<float> Rcpu(N);
  auto t0=std::chrono::high_resolution_clock::now();
  for(int y=0;y<H;y+=CHUNK)
    cpuStripe(Ix.data(),Iy.data(),Rcpu.data(),W,H,y,std::min(y+CHUNK,H),kH);
  auto t1=std::chrono::high_resolution_clock::now();
  double cpu_ms=elapsed_ms(t0,t1);

  // GPU setup
  CUDA_CHECK(cudaMemcpyToSymbol(d_k,&kH,sizeof(float)));

  cudaStream_t streams[2];
  for(int i=0;i<2;++i) CUDA_CHECK(cudaStreamCreate(&streams[i]));

  // Allocate per-stream chunk buffers
  float *d_Ix[2],*d_Iy[2],*d_Ix2[2],*d_Iy2[2],*d_Ixy[2];
  float *d_tmp[2],*d_Ix2b[2],*d_Iy2b[2],*d_Ixyb[2],*d_R[2];
  size_t chunkBytes = CHUNK*W*sizeof(float);
  for(int s=0;s<2;++s){
    CUDA_CHECK(cudaMalloc(&d_Ix [s],chunkBytes));
    CUDA_CHECK(cudaMalloc(&d_Iy [s],chunkBytes));
    CUDA_CHECK(cudaMalloc(&d_Ix2[s],chunkBytes));
    CUDA_CHECK(cudaMalloc(&d_Iy2[s],chunkBytes));
    CUDA_CHECK(cudaMalloc(&d_Ixy[s],chunkBytes));
    CUDA_CHECK(cudaMalloc(&d_tmp [s],chunkBytes));
    CUDA_CHECK(cudaMalloc(&d_Ix2b[s],chunkBytes));
    CUDA_CHECK(cudaMalloc(&d_Iy2b[s],chunkBytes));
    CUDA_CHECK(cudaMalloc(&d_Ixyb[s],chunkBytes));
    CUDA_CHECK(cudaMalloc(&d_R   [s],chunkBytes));
  }

  std::vector<float> Rgpu(N);

  cudaEvent_t eBeg,eEnd; CUDA_CHECK(cudaEventCreate(&eBeg)); CUDA_CHECK(cudaEventCreate(&eEnd));
  CUDA_CHECK(cudaEventRecord(eBeg));

  for(int stripe=0,y=0;y<H;y+=CHUNK,++stripe){
    int Hs=std::min(CHUNK,H-y);
    size_t bytes=Hs*W*sizeof(float);
    int s=stripe&1;              // ping-pong stream index

    // Async copies gradients
    CUDA_CHECK(cudaMemcpyAsync(d_Ix[s],Ix.data()+y*W,bytes,cudaMemcpyHostToDevice,streams[s]));
    CUDA_CHECK(cudaMemcpyAsync(d_Iy[s],Iy.data()+y*W,bytes,cudaMemcpyHostToDevice,streams[s]));

    // Kernels
    dim3 prodBlk(16,16), prodGrd((W+15)/16,(Hs+15)/16);
    prodKernel<<<prodGrd,prodBlk,0,streams[s]>>>(d_Ix[s],d_Iy[s],
                                                 d_Ix2[s],d_Iy2[s],d_Ixy[s],W);

    dim3 blkH(256), grdH((W+255)/256,Hs);
    blurH<3><<<grdH,blkH,0,streams[s]>>>(d_Ix2[s],d_tmp[s],W);
    blurV<3><<<dim3(W,(Hs+255)/256),blkH,0,streams[s]>>>(d_tmp[s],d_Ix2b[s],W,Hs);

    blurH<3><<<grdH,blkH,0,streams[s]>>>(d_Iy2[s],d_tmp[s],W);
    blurV<3><<<dim3(W,(Hs+255)/256),blkH,0,streams[s]>>>(d_tmp[s],d_Iy2b[s],W,Hs);

    blurH<3><<<grdH,blkH,0,streams[s]>>>(d_Ixy[s],d_tmp[s],W);
    blurV<3><<<dim3(W,(Hs+255)/256),blkH,0,streams[s]>>>(d_tmp[s],d_Ixyb[s],W,Hs);

    int threads=256, blocks=(Hs*W+threads-1)/threads;
    harrisKernel<<<blocks,threads,0,streams[s]>>>(d_Ix2b[s],d_Iy2b[s],d_Ixyb[s],d_R[s],Hs*W);

    // Async copy back
    CUDA_CHECK(cudaMemcpyAsync(Rgpu.data()+y*W,d_R[s],bytes,cudaMemcpyDeviceToHost,streams[s]));
  }

  // Wait for all work to finish
  CUDA_CHECK(cudaEventRecord(eEnd));
  CUDA_CHECK(cudaEventSynchronize(eEnd));

  float gpu_ms; CUDA_CHECK(cudaEventElapsedTime(&gpu_ms,eBeg,eEnd));

  // RMSE
  double mse=0; for(size_t i=0;i<N;++i){ double d=Rgpu[i]-Rcpu[i]; mse+=d*d; }
  double rmse=std::sqrt(mse/N);

  std::cout<<"CPU total               : "<<cpu_ms<<" ms\n";
  std::cout<<"GPU total (overlapped)  : "<<gpu_ms<<" ms\n";
  std::cout<<"RMSE (CPU vs GPU)       : "<<rmse<<"\n";

  for(int s=0;s<2;++s){
    CUDA_CHECK(cudaFree(d_Ix [s])); CUDA_CHECK(cudaFree(d_Iy [s]));
    CUDA_CHECK(cudaFree(d_Ix2[s])); CUDA_CHECK(cudaFree(d_Iy2[s])); CUDA_CHECK(cudaFree(d_Ixy[s]));
    CUDA_CHECK(cudaFree(d_tmp [s]));
    CUDA_CHECK(cudaFree(d_Ix2b[s])); CUDA_CHECK(cudaFree(d_Iy2b[s])); CUDA_CHECK(cudaFree(d_Ixyb[s]));
    CUDA_CHECK(cudaFree(d_R   [s]));
    CUDA_CHECK(cudaStreamDestroy(streams[s]));
  }
  return 0;
}
