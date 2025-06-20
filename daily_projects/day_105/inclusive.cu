#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <cub/cub.cuh>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <chrono>
#include <cstring>

#define CUDA_CHECK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){                 \
  fprintf(stderr,"CUDA error: %s (%s:%d)\n",cudaGetErrorString(e),__FILE__,__LINE__); \
  std::exit(1);} }while(0)

template<class TP> inline double ms(TP a,TP b){
  return std::chrono::duration<double,std::milli>(b-a).count(); }

// CPU serial scan
void scanCPU(const std::vector<uint32_t>& in,std::vector<uint32_t>& out){
  out.resize(in.size());
  uint64_t acc=0;
  for(size_t i=0;i<in.size();++i){ acc+=in[i]; out[i]=uint32_t(acc); }
}

// GPU benchmark helper
double benchGPU(bool useThrust,
                uint32_t* d_buf, uint32_t* h_in, uint32_t* h_out,
                size_t N, const std::vector<uint32_t>& ref,
                cudaEvent_t beg,cudaEvent_t end,
                cudaEvent_t h0,cudaEvent_t h1,
                cudaEvent_t k0,cudaEvent_t k1,
                cudaEvent_t d0,cudaEvent_t d1)
{
  // H2D
  CUDA_CHECK(cudaEventRecord(beg));
  CUDA_CHECK(cudaEventRecord(h0));
  CUDA_CHECK(cudaMemcpyAsync(d_buf,h_in,N*4,cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaEventRecord(h1));

  // Kernel
  CUDA_CHECK(cudaEventRecord(k0));
  if(useThrust){
      thrust::device_ptr<uint32_t> dev(d_buf);
      thrust::inclusive_scan(dev,dev+N,dev);
  }else{
      // CUB DeviceScan
      void*  d_temp = nullptr;
      size_t bytes  = 0;
      cub::DeviceScan::InclusiveSum(d_temp,bytes,d_buf,d_buf,N);
      CUDA_CHECK(cudaMalloc(&d_temp,bytes));
      cub::DeviceScan::InclusiveSum(d_temp,bytes,d_buf,d_buf,N);
      CUDA_CHECK(cudaFree(d_temp));
  }
  CUDA_CHECK(cudaEventRecord(k1));

  // D2H
  CUDA_CHECK(cudaEventRecord(d0));
  CUDA_CHECK(cudaMemcpyAsync(h_out,d_buf,N*4,cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaEventRecord(d1));

  CUDA_CHECK(cudaEventRecord(end));
  CUDA_CHECK(cudaEventSynchronize(end));

  auto et=[&](cudaEvent_t a,cudaEvent_t b){ float t; CUDA_CHECK(cudaEventElapsedTime(&t,a,b)); return double(t); };
  double tH2D=et(h0,h1), tKer=et(k0,k1), tD2H=et(d0,d1), tTot=et(beg,end);
  double gbps=(2.0*N*4)/(tKer/1e3)/1e9;

  size_t mism=0; for(size_t i=0;i<N;++i) if(h_out[i]!=ref[i]){mism=1; break;}

  printf("\n[%s]\n",useThrust? "Thrust inclusive_scan":"CUB DeviceScan");
  printf("H2D    : %8.3f ms\n",tH2D);
  printf("Kernel : %8.3f ms   (%.1f GB/s)\n",tKer,gbps);
  printf("D2H    : %8.3f ms\n",tD2H);
  printf("Total  : %8.3f ms\n",tTot);
  printf("Mismatch: %zu\n",mism);
  return tTot;
}

int main(int argc,char**argv)
{
  size_t N=(argc>=2)? std::stoul(argv[1]) : (1ull<<28);     // 1 GiB default
  printf("Elements: %zu (%.1f GiB)\n",N,N*4.0/1073741824.0);

  // Host input
  std::vector<uint32_t> h(N);
  for(uint32_t& v:h) v=uint32_t(rand());

  // CPU reference
  auto c0=std::chrono::high_resolution_clock::now();
  std::vector<uint32_t> ref; scanCPU(h,ref);
  auto c1=std::chrono::high_resolution_clock::now();
  double cpu_ms=ms(c0,c1);
  printf("\nCPU serial scan: %.3f ms\n",cpu_ms);

  // Pinned host buffers
  uint32_t *h_in,*h_out;
  CUDA_CHECK(cudaMallocHost(&h_in ,N*4));
  CUDA_CHECK(cudaMallocHost(&h_out,N*4));
  std::memcpy(h_in,h.data(),N*4);

  // Device buffer
  uint32_t *d_buf; CUDA_CHECK(cudaMalloc(&d_buf,N*4));

  // Events
  cudaEvent_t beg,end,h0,h1,k0,k1,d0,d1;
  CUDA_CHECK(cudaEventCreate(&beg)); CUDA_CHECK(cudaEventCreate(&end));
  CUDA_CHECK(cudaEventCreate(&h0));  CUDA_CHECK(cudaEventCreate(&h1));
  CUDA_CHECK(cudaEventCreate(&k0));  CUDA_CHECK(cudaEventCreate(&k1));
  CUDA_CHECK(cudaEventCreate(&d0));  CUDA_CHECK(cudaEventCreate(&d1));

  // Run paths
  double tThrust = benchGPU(true ,d_buf,h_in,h_out,N,ref,beg,end,h0,h1,k0,k1,d0,d1);
  double tCUB    = benchGPU(false,d_buf,h_in,h_out,N,ref,beg,end,h0,h1,k0,k1,d0,d1);

  CUDA_CHECK(cudaFree(d_buf));
  CUDA_CHECK(cudaFreeHost(h_in));
  CUDA_CHECK(cudaFreeHost(h_out));
  return 0;
}