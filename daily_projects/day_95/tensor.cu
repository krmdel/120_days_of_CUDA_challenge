
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

#define CUDA_CHECK(call)                                                          \
  do {                                                                            \
    cudaError_t err__ = (call);                                                   \
    if (err__ != cudaSuccess) {                                                   \
      std::cerr << "CUDA error: " << cudaGetErrorString(err__)                    \
                << " (" << __FILE__ << ':' << __LINE__ << ")\n"; std::exit(1);    \
    }                                                                             \
  } while (0)

template<typename TP>
inline double msec(TP a, TP b){
  return std::chrono::duration<double,std::milli>(b-a).count(); }

// Helpers: clamp + raw reader
inline int clamp(int v,int lo,int hi){ return v<lo?lo:(v>hi?hi:v); }

bool readRawFloat(const char* fn,std::vector<float>& buf,size_t count){
  std::ifstream f(fn,std::ios::binary); if(!f) return false;
  buf.resize(count);
  f.read(reinterpret_cast<char*>(buf.data()),count*sizeof(float));
  return bool(f);
}

// CPU reference (products + 7×7 box)
static void cpuTensor(const std::vector<float>& Ix,
                      const std::vector<float>& Iy,
                      std::vector<float>& Ix2b,
                      std::vector<float>& Iy2b,
                      std::vector<float>& Ixyb,
                      int W,int H,int R)
{
  const size_t N=size_t(W)*H;
  std::vector<float> Ix2(N),Iy2(N),Ixy(N);
  for(size_t i=0;i<N;++i){
    float ix=Ix[i], iy=Iy[i];
    Ix2[i]=ix*ix; Iy2[i]=iy*iy; Ixy[i]=ix*iy;
  }
  auto boxBlur=[&](const std::vector<float>& in,std::vector<float>& out){
    std::vector<float> tmp(N);
    // horizontal pass
    for(int y=0;y<H;++y)
      for(int x=0;x<W;++x){
        float s=0.f;
        for(int k=-R;k<=R;++k){
          int xx=clamp(x+k,0,W-1);
          s+=in[y*W+xx];
        }
        tmp[y*W+x]=s;
      }
    // vertical pass
    out.resize(N);
    for(int y=0;y<H;++y)
      for(int x=0;x<W;++x){
        float s=0.f;
        for(int k=-R;k<=R;++k){
          int yy=clamp(y+k,0,H-1);
          s+=tmp[yy*W+x];
        }
        out[y*W+x]=s;
      }
  };
  boxBlur(Ix2,Ix2b);
  boxBlur(Iy2,Iy2b);
  boxBlur(Ixy,Ixyb);
}

// GPU kernels
// Point-wise products
__global__ void productKernel(const float* Ix,const float* Iy,
                              float* Ix2,float* Iy2,float* Ixy,int N)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i>=N) return;
  float ix=Ix[i], iy=Iy[i];
  Ix2[i]=ix*ix; Iy2[i]=iy*iy; Ixy[i]=ix*iy;
}

// 2a. Horizontal 7-tap box (radius=3)
template<int R>
__global__ void boxBlurH(const float* in,float* out,int W,int H)
{
  int x=blockIdx.x*blockDim.x+threadIdx.x;
  int y=blockIdx.y;
  if(x>=W||y>=H) return;
  float s=0.f;
  for(int k=-R;k<=R;++k){
    int xx=max(0,min(W-1,x+k));
    s+=in[y*W+xx];
  }
  out[y*W+x]=s;
}

// Vertical 7-tap box (radius=3)
template<int R>
__global__ void boxBlurV(const float* in,float* out,int W,int H)
{
  int x=blockIdx.x;
  int y=blockIdx.y*blockDim.y+threadIdx.y;
  if(x>=W||y>=H) return;
  float s=0.f;
  for(int k=-R;k<=R;++k){
    int yy=max(0,min(H-1,y+k));
    s+=in[yy*W+x];
  }
  out[y*W+x]=s;
}

int main(int argc,char**argv)
{
  constexpr int R=3;                    // radius => 7×7 window
  int W=4096,H=4096;
  std::vector<float> Ix,Iy;
  if(argc==5){                          // Ix.bin Iy.bin W H
    W=std::stoi(argv[3]); H=std::stoi(argv[4]);
    if(!readRawFloat(argv[1],Ix,size_t(W)*H) ||
       !readRawFloat(argv[2],Iy,size_t(W)*H)){
      std::cerr<<"Failed to load gradients\n"; return 1;
    }
    std::cout<<"Loaded gradients "<<W<<"×"<<H<<"\n";
  }else{
    Ix.resize(size_t(W)*H);
    Iy.resize(size_t(W)*H);
    std::srand(1234);
    for(size_t i=0;i<Ix.size();++i){
      Ix[i]=float((std::rand()&511)-255);
      Iy[i]=float((std::rand()&511)-255);
    }
    std::cout<<"Generated random gradients "<<W<<"×"<<H<<"\n";
  }
  const size_t N=size_t(W)*H;

  // CPU reference
  std::vector<float> Ix2bCPU,Iy2bCPU,IxybCPU;
  auto c0=std::chrono::high_resolution_clock::now();
  cpuTensor(Ix,Iy,Ix2bCPU,Iy2bCPU,IxybCPU,W,H,R);
  auto c1=std::chrono::high_resolution_clock::now();
  double cpu_ms=msec(c0,c1);

  // GPU path
  float *d_Ix,*d_Iy,*d_Ix2,*d_Iy2,*d_Ixy;
  float *d_tmp,*d_Ix2b,*d_Iy2b,*d_Ixyb;
  CUDA_CHECK(cudaMalloc(&d_Ix ,N*sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_Iy ,N*sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_Ix2 ,N*sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_Iy2 ,N*sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_Ixy ,N*sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_tmp ,N*sizeof(float)));   // for intermediate blur
  CUDA_CHECK(cudaMalloc(&d_Ix2b,N*sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_Iy2b,N*sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_Ixyb,N*sizeof(float)));

  // Events
  cudaEvent_t tBeg,tEnd,eH2D0,eH2D1,eK0,eK1,eD2H0,eD2H1;
  CUDA_CHECK(cudaEventCreate(&tBeg));
  CUDA_CHECK(cudaEventCreate(&tEnd));
  CUDA_CHECK(cudaEventCreate(&eH2D0));
  CUDA_CHECK(cudaEventCreate(&eH2D1));
  CUDA_CHECK(cudaEventCreate(&eK0));
  CUDA_CHECK(cudaEventCreate(&eK1));
  CUDA_CHECK(cudaEventCreate(&eD2H0));
  CUDA_CHECK(cudaEventCreate(&eD2H1));

  CUDA_CHECK(cudaEventRecord(tBeg));

  // H2D
  CUDA_CHECK(cudaEventRecord(eH2D0));
  CUDA_CHECK(cudaMemcpy(d_Ix,Ix.data(),N*sizeof(float),cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_Iy,Iy.data(),N*sizeof(float),cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaEventRecord(eH2D1));

  // Kernels
  int threads=256, blocks=(N+threads-1)/threads;
  dim3 blkH(256); dim3 gridH((W+255)/256,H);     // for horizontal pass
  dim3 blkV(256); dim3 gridV(W,(H+255)/256);     // for vertical pass

  CUDA_CHECK(cudaEventRecord(eK0));

  // Products
  productKernel<<<blocks,threads>>>(d_Ix,d_Iy,d_Ix2,d_Iy2,d_Ixy,N);

  // Ix² blur
  boxBlurH<R><<<gridH,blkH>>>(d_Ix2,d_tmp,W,H);
  boxBlurV<R><<<gridV,blkV>>>(d_tmp,d_Ix2b,W,H);

  // Iy² blur
  boxBlurH<R><<<gridH,blkH>>>(d_Iy2,d_tmp,W,H);
  boxBlurV<R><<<gridV,blkV>>>(d_tmp,d_Iy2b,W,H);

  // IxIy blur
  boxBlurH<R><<<gridH,blkH>>>(d_Ixy,d_tmp,W,H);
  boxBlurV<R><<<gridV,blkV>>>(d_tmp,d_Ixyb,W,H);

  CUDA_CHECK(cudaEventRecord(eK1));

  // D2H
  std::vector<float> Ix2bGPU(N),Iy2bGPU(N),IxybGPU(N);
  CUDA_CHECK(cudaEventRecord(eD2H0));
  CUDA_CHECK(cudaMemcpy(Ix2bGPU.data(), d_Ix2b, N*sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(Iy2bGPU.data(), d_Iy2b, N*sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(IxybGPU.data(), d_Ixyb, N*sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaEventRecord(eD2H1));

  CUDA_CHECK(cudaEventRecord(tEnd));
  CUDA_CHECK(cudaEventSynchronize(tEnd));

  auto evms=[&](cudaEvent_t a,cudaEvent_t b){ float ms; CUDA_CHECK(cudaEventElapsedTime(&ms,a,b)); return double(ms); };
  double h2d = evms(eH2D0,eH2D1);
  double ker = evms(eK0 ,eK1 );
  double d2h = evms(eD2H0,eD2H1);
  double tot = evms(tBeg,tEnd);

  // RMSE check (Ix² blurred only)
  double mse=0; for(size_t i=0;i<N;++i){ double d=Ix2bGPU[i]-Ix2bCPU[i]; mse+=d*d; }
  double rmse=std::sqrt(mse/N);

  std::cout<<"CPU total                 : "<<cpu_ms<<" ms\n";
  std::cout<<"GPU H2D copy              : "<<h2d    <<" ms\n";
  std::cout<<"GPU kernels (all)         : "<<ker    <<" ms\n";
  std::cout<<"GPU D2H copy              : "<<d2h    <<" ms\n";
  std::cout<<"GPU total                 : "<<tot    <<" ms\n";
  std::cout<<"RMSE Ix² blurred (CPU/GPU): "<<rmse    <<"\n";

  CUDA_CHECK(cudaFree(d_Ix)); CUDA_CHECK(cudaFree(d_Iy));
  CUDA_CHECK(cudaFree(d_Ix2));CUDA_CHECK(cudaFree(d_Iy2));CUDA_CHECK(cudaFree(d_Ixy));
  CUDA_CHECK(cudaFree(d_tmp));
  CUDA_CHECK(cudaFree(d_Ix2b));CUDA_CHECK(cudaFree(d_Iy2b));CUDA_CHECK(cudaFree(d_Ixyb));
  return 0;
}
