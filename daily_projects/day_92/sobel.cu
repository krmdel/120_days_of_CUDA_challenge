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
  do { cudaError_t e = (x); if(e != cudaSuccess) {                                \
       std::cerr << "CUDA error: " << cudaGetErrorString(e)                       \
                 << " (" << __FILE__ << ':' << __LINE__ << ")\n"; std::exit(1);}  \
  } while (0)

constexpr float PI = 3.14159265358979323846f;

// PGM loader (binary, 8-bit)
bool readPGM(const char* fn, std::vector<uint8_t>& img, int& w, int& h)
{
  std::ifstream f(fn, std::ios::binary); if(!f) return false;
  std::string m; f >> m; if(m != "P5") return false;
  int maxv; f >> w >> h >> maxv; f.get(); if(maxv != 255) return false;
  img.resize(size_t(w)*h);
  f.read(reinterpret_cast<char*>(img.data()), img.size());
  return bool(f);
}

// CPU reference Sobel (for validation)
inline int sobelGx(const uint8_t* p, int s) {
  return (p[-s-1] + 2*p[-1] + p[s-1]) - (p[-s+1] + 2*p[+1] + p[s+1]);
}
inline int sobelGy(const uint8_t* p, int s) {
  return (p[-s-1] + 2*p[-s] + p[-s+1]) - (p[s-1] + 2*p[s] + p[s+1]);
}
static void sobelCPU(const std::vector<uint8_t>& in,
                     std::vector<float>& mag,
                     std::vector<uint8_t>& ang,
                     int W, int H)
{
  mag.assign(W*H, 0.0f); ang.assign(W*H, 0);
  for(int y = 1; y < H-1; ++y)
    for(int x = 1; x < W-1; ++x) {
      const uint8_t* p = &in[y*W + x];
      int gx = sobelGx(p, W);
      int gy = sobelGy(p, W);
      float m  = std::sqrt(float(gx*gx + gy*gy));
      float th = static_cast<float>(
                   std::atan2(static_cast<float>(gy),
                              static_cast<float>(gx)) * 180.0 / PI);
      if(th < 0.f) th += 180.f;
      uint8_t bin = (th < 22.5f) ? 0 :
                    (th < 67.5f) ? 1 :
                    (th <112.5f) ? 2 : 3;
      mag[y*W + x] = m;
      ang[y*W + x] = bin;
    }
}

// GPU Sobel (16×16 tile + halo)
__global__ void sobelKernel(const uint8_t* __restrict__ src,
                            float*       __restrict__ mag,
                            uint8_t*     __restrict__ ang,
                            int W, int H)
{
  __shared__ uint8_t sh[18][18];               // 16×16 tile + 1-pixel halo
  int tx = threadIdx.x, ty = threadIdx.y;
  int gx = blockIdx.x * 16 + tx;
  int gy = blockIdx.y * 16 + ty;
  int lx = tx + 1, ly = ty + 1;                // local coords in shared

  if(gx < W && gy < H) sh[ly][lx] = src[gy*W + gx];

  // Halo loads
  if(tx==0   && gx>0   )                sh[ly][0 ]  = src[gy*W + gx-1];
  if(tx==15  && gx+1<W )                sh[ly][17]  = src[gy*W + gx+1];
  if(ty==0   && gy>0   )                sh[0 ][lx]  = src[(gy-1)*W + gx];
  if(ty==15  && gy+1<H )                sh[17][lx]  = src[(gy+1)*W + gx];
  if(tx==0 && ty==0   && gx>0   && gy>0  ) sh[0 ][0 ] = src[(gy-1)*W+gx-1];
  if(tx==15&& ty==0   && gx+1<W && gy>0  ) sh[0 ][17] = src[(gy-1)*W+gx+1];
  if(tx==0 && ty==15  && gx>0   && gy+1<H) sh[17][0 ] = src[(gy+1)*W+gx-1];
  if(tx==15&& ty==15  && gx+1<W && gy+1<H) sh[17][17]= src[(gy+1)*W+gx+1];

  __syncthreads();

  if(gx==0 || gy==0 || gx==W-1 || gy==H-1) {
    if(gx < W && gy < H){ mag[gy*W+gx]=0.f; ang[gy*W+gx]=0; }
    return;
  }

  int Gx = (sh[ly-1][lx-1] + 2*sh[ly][lx-1] + sh[ly+1][lx-1]) -
           (sh[ly-1][lx+1] + 2*sh[ly][lx+1] + sh[ly+1][lx+1]);
  int Gy = (sh[ly-1][lx-1] + 2*sh[ly-1][lx] + sh[ly-1][lx+1]) -
           (sh[ly+1][lx-1] + 2*sh[ly+1][lx] + sh[ly+1][lx+1]);

  float m  = sqrtf(float(Gx*Gx + Gy*Gy));
  float th = atan2f(float(Gy), float(Gx))*180.f/PI; if(th<0.f) th+=180.f;
  uint8_t bin = (th<22.5f)?0:(th<67.5f)?1:(th<112.5f)?2:3;

  mag[gy*W+gx] = m;
  ang[gy*W+gx] = bin;
}

template<typename TP>
inline double ms(TP a, TP b){ return std::chrono::duration<double,std::milli>(b-a).count(); }

int main(int argc,char**argv)
{
  int W=4096,H=4096; std::vector<uint8_t> img8;
  if(argc>=2){ if(!readPGM(argv[1],img8,W,H)){std::cerr<<"PGM read fail\n";return 1;} }
  else { img8.resize(size_t(W)*H); std::srand(1234); for(auto& p:img8) p=rand()&0xFF; }

  // CPU reference
  std::vector<float> magCPU; std::vector<uint8_t> angCPU;
  auto c0=std::chrono::high_resolution_clock::now();
  sobelCPU(img8,magCPU,angCPU,W,H);
  auto c1=std::chrono::high_resolution_clock::now();
  double cpu_ms = ms(c0,c1);

  // GPU buffers
  uint8_t *d_img,*d_ang; float *d_mag;
  size_t imgB=size_t(W)*H*sizeof(uint8_t);
  size_t magB=size_t(W)*H*sizeof(float);
  CUDA_CHECK(cudaMalloc(&d_img,imgB));
  CUDA_CHECK(cudaMalloc(&d_mag,magB));
  CUDA_CHECK(cudaMalloc(&d_ang,imgB));

  cudaEvent_t e0,e1,h2d0,h2d1,k0,k1,d2h0,d2h1;
  for(cudaEvent_t* ev:{&e0,&e1,&h2d0,&h2d1,&k0,&k1,&d2h0,&d2h1}) CUDA_CHECK(cudaEventCreate(ev));

  CUDA_CHECK(cudaEventRecord(e0));

  CUDA_CHECK(cudaEventRecord(h2d0));
  CUDA_CHECK(cudaMemcpy(d_img,img8.data(),imgB,cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaEventRecord(h2d1));

  dim3 blk(16,16), grd((W+15)/16,(H+15)/16);
  CUDA_CHECK(cudaEventRecord(k0));
  sobelKernel<<<grd,blk>>>(d_img,d_mag,d_ang,W,H);
  CUDA_CHECK(cudaEventRecord(k1));

  std::vector<float> magGPU(W*H);
  CUDA_CHECK(cudaEventRecord(d2h0));
  CUDA_CHECK(cudaMemcpy(magGPU.data(),d_mag,magB,cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaEventRecord(d2h1));

  CUDA_CHECK(cudaEventRecord(e1));
  CUDA_CHECK(cudaEventSynchronize(e1));

  auto ems=[&](cudaEvent_t a,cudaEvent_t b){ float m;CUDA_CHECK(cudaEventElapsedTime(&m,a,b));return double(m);};
  double h2d=ems(h2d0,h2d1), ker=ems(k0,k1), d2h=ems(d2h0,d2h1), tot=ems(e0,e1);

  // RMSE check
  double mse=0; for(size_t i=0;i<magGPU.size();++i){ double d=magGPU[i]-magCPU[i]; mse+=d*d; }
  double rmse=std::sqrt(mse/magGPU.size());

  std::cout << "Image size: " << W << " × " << H << " \n";
  std::cout<<"CPU total              : "<<cpu_ms<<" ms\n";
  std::cout<<"GPU H2D                : "<<h2d<<" ms\n";
  std::cout<<"GPU kernel             : "<<ker<<" ms\n";
  std::cout<<"GPU D2H                : "<<d2h<<" ms\n";
  std::cout<<"GPU total              : "<<tot<<" ms\n";
  std::cout<<"RMSE (mag CPU vs GPU)  : "<<rmse<<"\n";

  CUDA_CHECK(cudaFree(d_img)); CUDA_CHECK(cudaFree(d_mag)); CUDA_CHECK(cudaFree(d_ang));
  return 0;
}
