#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cub/cub.cuh>

#include <thrust/device_ptr.h>
#include <thrust/transform.h>
#include <thrust/scan.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/functional.h>

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <chrono>
#include <fstream>
#include <iostream>

#define CUDA_CHECK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){                 \
  std::cerr<<"CUDA error: "<<cudaGetErrorString(e)<<" ("<<__FILE__<<':'          \
           <<__LINE__<<")\n"; std::exit(1);} }while(0)

template<class TP> inline double ms(TP a,TP b){
  return std::chrono::duration<double,std::milli>(b-a).count(); }

// PGM loader
bool readPGM(const char* fn,std::vector<uint8_t>& img,int& W,int& H){
  std::ifstream f(fn,std::ios::binary); if(!f) return false;
  std::string m; int maxv; f>>m>>W>>H>>maxv; f.get();
  if(m!="P5"||maxv!=255) return false;
  img.resize(size_t(W)*H);
  f.read(reinterpret_cast<char*>(img.data()),img.size());
  return bool(f);
}

// CPU uint32 integral
void integralCPU(const std::vector<uint8_t>& in,std::vector<uint32_t>& out,int W,int H){
  out.resize(in.size());
  for(int y=0;y<H;++y){
    uint32_t acc=0;
    for(int x=0;x<W;++x){
      acc += in[y*W+x];
      out[y*W+x] = acc + (y ? out[(y-1)*W+x] : 0);
    }
  }
}

// Per-row key iterator
struct RowKey{
  int W;
  __host__ __device__ int operator()(int idx) const { return idx / W; }
};

// Cast functor for Thrust
struct CastByte { __host__ __device__ uint32_t operator()(uint8_t v) const {
  return static_cast<uint32_t>(v); }};

// Saturating 8-bit add
struct SatAddU8{
  __device__ __forceinline__ uint8_t operator()(uint8_t a,uint8_t b) const{
    uint16_t t = uint16_t(a) + uint16_t(b);
    return t > 255 ? 255 : uint8_t(t);
  }
};

// FP16 helpers
__global__ void packLow12(const uint8_t* in,__half* lo,uint8_t* hi,int N){
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if(i<N){
    uint16_t v = in[i];
    hi[i] = v >> 12;
    lo[i] = __half(v & 0x0FFF);
  }
}
__global__ void addHigh(__half* lo,const uint8_t* hi,int N){
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if(i<N){
    uint16_t v = __half2ushort_rd(lo[i]) + (uint16_t(hi[i])<<12);
    lo[i] = __half(v);
  }
}
struct HalfAdd{
  __device__ __forceinline__ __half operator()(const __half& a,
                                               const __half& b) const{
    return __hadd(a,b);
  }
};

// PSNR
double psnr(const std::vector<uint32_t>& a,const std::vector<uint32_t>& b){
  double mse=0;
  for(size_t i=0;i<a.size();++i){ double d=double(a[i])-double(b[i]); mse+=d*d; }
  mse/=a.size();
  return 10.0*std::log10((255.0*255.0)/mse);
}

int main(int argc,char**argv)
{
  int W=8192,H=8192;
  std::vector<uint8_t> img;
  if(argc==2){
    if(!readPGM(argv[1],img,W,H)){ std::cerr<<"PGM load error\n"; return 1; }
  }else{
    img.resize(size_t(W)*H); for(auto& p:img) p = rand() & 0xFF;
  }
  const int N = W*H;

  // CPU reference
  std::vector<uint32_t> cpu32;
  auto tc0=std::chrono::high_resolution_clock::now();
  integralCPU(img,cpu32,W,H);
  auto tc1=std::chrono::high_resolution_clock::now();
  double cpu_ms = ms(tc0,tc1);

  // Device buffers
  uint8_t *d_img; CUDA_CHECK(cudaMalloc(&d_img,N));
  CUDA_CHECK(cudaMemcpy(d_img,img.data(),N,cudaMemcpyHostToDevice));

  auto keys = thrust::make_transform_iterator(
        thrust::counting_iterator<int>(0), RowKey{W});

  cudaEvent_t e0,e1; CUDA_CHECK(cudaEventCreate(&e0)); CUDA_CHECK(cudaEventCreate(&e1));

  // FP32
  uint32_t* d32; CUDA_CHECK(cudaMalloc(&d32,N*sizeof(uint32_t)));
  thrust::transform(thrust::device_pointer_cast(d_img),
                    thrust::device_pointer_cast(d_img)+N,
                    thrust::device_pointer_cast(d32),
                    CastByte());

  CUDA_CHECK(cudaEventRecord(e0));
  thrust::inclusive_scan_by_key(keys, keys+N,
                                thrust::device_pointer_cast(d32),
                                thrust::device_pointer_cast(d32));
  CUDA_CHECK(cudaEventRecord(e1)); CUDA_CHECK(cudaEventSynchronize(e1));
  float ker32_ms; CUDA_CHECK(cudaEventElapsedTime(&ker32_ms,e0,e1));

  std::vector<uint32_t> out32(N);
  CUDA_CHECK(cudaMemcpy(out32.data(),d32,N*4,cudaMemcpyDeviceToHost));

  // FP16
  __half* dLo; CUDA_CHECK(cudaMalloc(&dLo,N*sizeof(__half)));
  uint8_t* dHi; CUDA_CHECK(cudaMalloc(&dHi,N));
  packLow12<<<(N+255)/256,256>>>(d_img,dLo,dHi,N);

  CUDA_CHECK(cudaEventRecord(e0));
  thrust::inclusive_scan_by_key(keys, keys+N,
                                thrust::device_pointer_cast(dLo),
                                thrust::device_pointer_cast(dLo),
                                thrust::equal_to<int>(), HalfAdd());
  CUDA_CHECK(cudaEventRecord(e1)); CUDA_CHECK(cudaEventSynchronize(e1));
  float ker16_ms; CUDA_CHECK(cudaEventElapsedTime(&ker16_ms,e0,e1));

  addHigh<<<(N+255)/256,256>>>(dLo,dHi,N);

  std::vector<uint16_t> loHost(N); CUDA_CHECK(cudaMemcpy(loHost.data(),dLo,N*2,cudaMemcpyDeviceToHost));
  std::vector<uint8_t>  hiHost(N); CUDA_CHECK(cudaMemcpy(hiHost.data(),dHi,N  ,cudaMemcpyDeviceToHost));
  std::vector<uint32_t> out16(N);
  for(int i=0;i<N;++i) out16[i]=uint32_t(loHost[i]) + (uint32_t(hiHost[i])<<12);

  // INT8 (saturating)
  uint8_t* d8; CUDA_CHECK(cudaMalloc(&d8,N));
  CUDA_CHECK(cudaMemcpy(d8,d_img,N,cudaMemcpyDeviceToDevice));

  CUDA_CHECK(cudaEventRecord(e0));
  thrust::inclusive_scan_by_key(keys, keys+N,
        thrust::device_pointer_cast(d8),
        thrust::device_pointer_cast(d8),
        thrust::equal_to<int>(), SatAddU8());
  CUDA_CHECK(cudaEventRecord(e1)); CUDA_CHECK(cudaEventSynchronize(e1));
  float ker8_ms; CUDA_CHECK(cudaEventElapsedTime(&ker8_ms,e0,e1));

  std::vector<uint8_t> out8(N); CUDA_CHECK(cudaMemcpy(out8.data(),d8,N,cudaMemcpyDeviceToHost));
  std::vector<uint32_t> out8u32(N);          // promote to uint32 for PSNR
  for(int i=0;i<N;++i) out8u32[i]=out8[i];

  // Report
  std::cout<<"CPU total               : "<<cpu_ms   <<" ms\n\n";

  std::cout<<"GPU FP32 kernel         : "<<ker32_ms <<" ms\n";
  std::cout<<"PSNR  FP32 vs CPU       : "<<psnr(cpu32,out32)<<" dB\n\n";

  std::cout<<"GPU FP16 kernel         : "<<ker16_ms <<" ms\n";
  std::cout<<"PSNR  FP16 vs CPU       : "<<psnr(cpu32,out16)<<" dB\n\n";

  std::cout<<"GPU INT8 kernel         : "<<ker8_ms  <<" ms\n";
  std::cout<<"PSNR  INT8 vs CPU       : "<<psnr(cpu32,out8u32)<<" dB\n";

  // Cleanup
  CUDA_CHECK(cudaFree(d_img)); CUDA_CHECK(cudaFree(d32));
  CUDA_CHECK(cudaFree(dLo));  CUDA_CHECK(cudaFree(dHi));
  CUDA_CHECK(cudaFree(d8));
  return 0;
}