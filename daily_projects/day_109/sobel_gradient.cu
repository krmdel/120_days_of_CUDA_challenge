#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <vector>
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

// CPU Sobel reference
void sobelCPU(const std::vector<uint8_t>& g,
              int W,int H,
              std::vector<float>& mag,
              std::vector<float>& ang)
{
  mag.assign(size_t(W)*H,0.f);
  ang.assign(size_t(W)*H,0.f);
  for(int y=1;y<H-1;++y)
    for(int x=1;x<W-1;++x){
      int idx=y*W+x;
      int gx =
        -g[(y-1)*W + (x-1)] + g[(y-1)*W + (x+1)] +
        -2*g[y*W + (x-1)]   + 2*g[y*W + (x+1)] +
        -g[(y+1)*W + (x-1)] + g[(y+1)*W + (x+1)];
      int gy =
         g[(y-1)*W + (x-1)] + 2*g[(y-1)*W +  x ] + g[(y-1)*W + (x+1)] -
         g[(y+1)*W + (x-1)] - 2*g[(y+1)*W +  x ] - g[(y+1)*W + (x+1)];
      float fx=float(gx), fy=float(gy);
      mag[idx]=std::sqrt(fx*fx+fy*fy);
      ang[idx]=std::atan2(fy,fx);          // use std::atan2 (float overload)
    }
}

// CUDA kernel
__global__ void sobelKernel(const uint8_t* __restrict__ g,
                            float* __restrict__ mag,
                            float* __restrict__ ang,
                            int W,int H)
{
  extern __shared__ uint8_t sh[];
  constexpr int B=16;
  int lx=threadIdx.x, ly=threadIdx.y;
  int gx=blockIdx.x*B + lx;
  int gy=blockIdx.y*B + ly;

  int shW=B+2;
  int sx=lx+1, sy=ly+1;
  if(gx<W && gy<H) sh[sy*shW + sx] = g[gy*W + gx];
  if(lx==0 && gx>0)               sh[sy*shW]       = g[gy*W + (gx-1)];
  if(lx==B-1 && gx+1<W)           sh[sy*shW + sx+1]= g[gy*W + (gx+1)];
  if(ly==0 && gy>0)               sh[sx]           = g[(gy-1)*W + gx];
  if(ly==B-1 && gy+1<H)           sh[(sy+1)*shW+sx]= g[(gy+1)*W + gx];
  if(lx==0 && ly==0 && gx>0 && gy>0)
      sh[0]=g[(gy-1)*W + (gx-1)];
  if(lx==B-1&&ly==0&&gx+1<W&&gy>0)
      sh[sx+1]=g[(gy-1)*W + (gx+1)];
  if(lx==0&&ly==B-1&&gx>0&&gy+1<H)
      sh[(sy+1)*shW]=g[(gy+1)*W + (gx-1)];
  if(lx==B-1&&ly==B-1&&gx+1<W&&gy+1<H)
      sh[(sy+1)*shW + sx+1]=g[(gy+1)*W + (gx+1)];
  __syncthreads();

  if(gx>0 && gx<W-1 && gy>0 && gy<H-1){
    int gxv =
      -sh[(sy-1)*shW + (sx-1)] + sh[(sy-1)*shW + (sx+1)] +
      -2*sh[ sy   *shW + (sx-1)] + 2*sh[ sy   *shW + (sx+1)] +
      -sh[(sy+1)*shW + (sx-1)] + sh[(sy+1)*shW + (sx+1)];
    int gyv =
       sh[(sy-1)*shW + (sx-1)] + 2*sh[(sy-1)*shW +  sx ] + sh[(sy-1)*shW + (sx+1)] -
       sh[(sy+1)*shW + (sx-1)] - 2*sh[(sy+1)*shW +  sx ] - sh[(sy+1)*shW + (sx+1)];
    float fx=float(gxv), fy=float(gyv);
    int idx=gy*W+gx;
    mag[idx]=sqrtf(fx*fx+fy*fy);
    ang[idx]=atan2f(fy,fx);
  }
}

// RMSE helper
double rmse(const std::vector<float>& a,const std::vector<float>& b){
  double e=0; for(size_t i=0;i<a.size();++i){ double d=a[i]-b[i]; e+=d*d; }
  return std::sqrt(e/a.size());
}

int main(int argc,char**argv)
{
  int W=4096,H=4096;
  std::vector<uint8_t> img;
  if(argc==2){
    if(!readPGM(argv[1],img,W,H)){ std::cerr<<"PGM read failed\n"; return 1; }
    std::cout<<"Loaded "<<W<<"×"<<H<<" PGM\n";
  }else{
    img.resize(size_t(W)*H);
    for(auto& p:img) p=rand()&0xFF;
    std::cout<<"Generated random "<<W<<"×"<<H<<" image\n";
  }
  size_t N = size_t(W)*H;

  // CPU reference
  std::vector<float> magCPU, angCPU;
  auto c0=std::chrono::high_resolution_clock::now();
  sobelCPU(img,W,H,magCPU,angCPU);
  auto c1=std::chrono::high_resolution_clock::now();
  double cpu_ms = ms(c0,c1);

  // GPU buffers
  uint8_t *d_img;    CUDA_CHECK(cudaMalloc(&d_img,N));
  float   *d_mag,*d_ang; CUDA_CHECK(cudaMalloc(&d_mag,N*sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_ang,N*sizeof(float)));

  cudaEvent_t eBeg,eEnd,eH0,eH1,eK0,eK1,eD0,eD1;
  for(auto ev:{&eBeg,&eEnd,&eH0,&eH1,&eK0,&eK1,&eD0,&eD1})
      CUDA_CHECK(cudaEventCreate(ev));

  CUDA_CHECK(cudaEventRecord(eBeg));

  CUDA_CHECK(cudaEventRecord(eH0));
  CUDA_CHECK(cudaMemcpy(d_img,img.data(),N,cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaEventRecord(eH1));

  dim3 blk(16,16);
  dim3 grd((W+15)/16,(H+15)/16);
  size_t smem=(16+2)*(16+2)*sizeof(uint8_t);

  CUDA_CHECK(cudaEventRecord(eK0));
  sobelKernel<<<grd,blk,smem>>>(d_img,d_mag,d_ang,W,H);
  CUDA_CHECK(cudaEventRecord(eK1));

  std::vector<float> magGPU(N), angGPU(N);
  CUDA_CHECK(cudaEventRecord(eD0));
  CUDA_CHECK(cudaMemcpy(magGPU.data(),d_mag,N*sizeof(float),cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(angGPU.data(),d_ang,N*sizeof(float),cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaEventRecord(eD1));

  CUDA_CHECK(cudaEventRecord(eEnd));
  CUDA_CHECK(cudaEventSynchronize(eEnd));

  auto evms=[&](cudaEvent_t a,cudaEvent_t b){ float m; CUDA_CHECK(cudaEventElapsedTime(&m,a,b)); return double(m); };
  double h2d=evms(eH0,eH1), ker=evms(eK0,eK1), d2h=evms(eD0,eD1), tot=evms(eBeg,eEnd);

  std::cout<<"CPU total               : "<<cpu_ms<<" ms\n";
  std::cout<<"GPU H2D copy            : "<<h2d    <<" ms\n";
  std::cout<<"GPU kernel              : "<<ker    <<" ms\n";
  std::cout<<"GPU D2H copy            : "<<d2h    <<" ms\n";
  std::cout<<"GPU total               : "<<tot    <<" ms\n";
  std::cout<<"RMSE  magnitude         : "<<rmse(magCPU,magGPU)<<"\n";
  std::cout<<"RMSE  angle             : "<<rmse(angCPU,angGPU)<<"\n";

  CUDA_CHECK(cudaFree(d_img));
  CUDA_CHECK(cudaFree(d_mag));
  CUDA_CHECK(cudaFree(d_ang));
  return 0;
}
