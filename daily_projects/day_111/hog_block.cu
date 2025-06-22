#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <chrono>
#include <iostream>

#define CUDA_CHECK(x)  do{ cudaError_t e=(x); if(e!=cudaSuccess){                \
  std::cerr<<"CUDA error: "<<cudaGetErrorString(e)<<" ("<<__FILE__<<':'          \
           <<__LINE__<<")\n"; std::exit(1);} }while(0)
template<class TP> double ms(TP a,TP b){
  return std::chrono::duration<double,std::milli>(b-a).count(); }

// CPU reference
void cpuBlock(const std::vector<float>& cell,int Cx,int Cy,
              std::vector<float>& desc)
{
  int Bx=Cx-1, By=Cy-1;
  desc.resize(size_t(Bx)*By*36);
  const float eps=1e-6f, clipVal=0.2f;
  for(int y=0;y<By;++y)
    for(int x=0;x<Bx;++x){
      float tmp[36];
      int outIdx=(y*Bx+x)*36;
      // gather 4 cells
      for(int cy=0;cy<2;++cy)
        for(int cx=0;cx<2;++cx){
          const float* h=&cell[((y+cy)*Cx+(x+cx))*9];
          for(int b=0;b<9;++b) tmp[(cy*2+cx)*9+b]=h[b];
        }
      // L2-norm
      float norm=eps;
      for(int i=0;i<36;++i) norm+=tmp[i]*tmp[i];
      norm=sqrtf(norm);
      for(int i=0;i<36;++i){ tmp[i]/=norm; if(tmp[i]>clipVal) tmp[i]=clipVal; }
      // renorm
      norm=eps;
      for(int i=0;i<36;++i) norm+=tmp[i]*tmp[i];
      norm=sqrtf(norm);
      for(int i=0;i<36;++i) desc[outIdx+i]=tmp[i]/norm;
    }
}

// GPU kernel
__global__ void blockKernel(const float* __restrict__ cell,float* __restrict__ desc,
                            int Cx,int Bx,int By)
{
  int bid = blockIdx.x*blockDim.x + threadIdx.x;          // 0 … Bx*By-1
  if(bid>=Bx*By) return;
  int by = bid / Bx;
  int bx = bid - by*Bx;
  float v[36];
  // gather 4×9 =36 bins
  #pragma unroll
  for(int cy=0;cy<2;++cy){
    int cyOff = (by+cy)*Cx;
    for(int cx=0;cx<2;++cx){
      const float* h = &cell[(cyOff + (bx+cx))*9];
      #pragma unroll
      for(int b=0;b<9;++b) v[(cy*2+cx)*9 + b] = h[b];
    }
  }
  // L2-Hys
  const float eps=1e-6f, clipVal=0.2f;
  float norm=eps;
  #pragma unroll
  for(int i=0;i<36;++i) norm+=v[i]*v[i];
  norm=sqrtf(norm);
  #pragma unroll
  for(int i=0;i<36;++i){
    v[i]/=norm;
    if(v[i]>clipVal) v[i]=clipVal;
  }
  norm=eps;
  #pragma unroll
  for(int i=0;i<36;++i) norm+=v[i]*v[i];
  norm=sqrtf(norm);
  #pragma unroll
  for(int i=0;i<36;++i) desc[bid*36+i]=v[i]/norm;
}

// RMSE
double rmse(const std::vector<float>& a,const std::vector<float>& b){
  double e=0; for(size_t i=0;i<a.size();++i){ double d=a[i]-b[i]; e+=d*d; }
  return std::sqrt(e/a.size());
}

int main(int argc,char**argv)
{
  // Cell grid size
  int Cx = (argc>=3)? std::atoi(argv[1]):512;
  int Cy = (argc>=3)? std::atoi(argv[2]):256;          // default 512×256 cells
  int Bx = Cx-1, By=Cy-1;
  size_t cells = size_t(Cx)*Cy;
  size_t blocks = size_t(Bx)*By;

  // Synthetic cell histogram (random)
  std::vector<float> cell(cells*9);
  for(auto& v:cell) v = float(rand())/RAND_MAX*0.5f;

  // CPU reference
  std::vector<float> cpuDesc;
  auto tc0=std::chrono::high_resolution_clock::now();
  cpuBlock(cell,Cx,Cy,cpuDesc);
  auto tc1=std::chrono::high_resolution_clock::now();
  double cpu_ms = ms(tc0,tc1);

  // GPU buffers
  float *d_cell,*d_desc;
  CUDA_CHECK(cudaMalloc(&d_cell,cell.size()*4));
  CUDA_CHECK(cudaMalloc(&d_desc,blocks*36*4));

  cudaEvent_t h0,h1,k0,k1,d0,d1,beg,end;
  for(auto ev:{&h0,&h1,&k0,&k1,&d0,&d1,&beg,&end}) CUDA_CHECK(cudaEventCreate(ev));

  CUDA_CHECK(cudaEventRecord(beg));

  CUDA_CHECK(cudaEventRecord(h0));
  CUDA_CHECK(cudaMemcpy(d_cell,cell.data(),cell.size()*4,cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaEventRecord(h1));

  int threads=256;
  int grid=(blocks+threads-1)/threads;
  CUDA_CHECK(cudaEventRecord(k0));
  blockKernel<<<grid,threads>>>(d_cell,d_desc,Cx,Bx,By);
  CUDA_CHECK(cudaEventRecord(k1));

  std::vector<float> gpuDesc(blocks*36);
  CUDA_CHECK(cudaEventRecord(d0));
  CUDA_CHECK(cudaMemcpy(gpuDesc.data(),d_desc,gpuDesc.size()*4,cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaEventRecord(d1));

  CUDA_CHECK(cudaEventRecord(end));
  CUDA_CHECK(cudaEventSynchronize(end));

  auto evms=[&](cudaEvent_t a,cudaEvent_t b){ float t; CUDA_CHECK(cudaEventElapsedTime(&t,a,b)); return double(t); };
  double h2d=evms(h0,h1), ker=evms(k0,k1), d2h=evms(d0,d1), tot=evms(beg,end);

  std::cout<<"Cells grid              : "<<Cx<<" × "<<Cy<<"  ("<<cells<<" cells)\n";
  std::cout<<"Blocks (out)            : "<<Bx<<" × "<<By<<"  ("<<blocks<<" blocks)\n\n";
  std::cout<<"CPU total               : "<<cpu_ms <<" ms\n";
  std::cout<<"GPU H2D copy            : "<<h2d     <<" ms\n";
  std::cout<<"GPU kernel              : "<<ker     <<" ms\n";
  std::cout<<"GPU D2H copy            : "<<d2h     <<" ms\n";
  std::cout<<"GPU total               : "<<tot     <<" ms\n";
  std::cout<<"RMSE  descriptors       : "<<rmse(cpuDesc,gpuDesc)<<"\n";

  CUDA_CHECK(cudaFree(d_cell)); CUDA_CHECK(cudaFree(d_desc));
  return 0;
}
