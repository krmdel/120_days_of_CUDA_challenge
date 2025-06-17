#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <fstream>
#include <iostream>
#include <chrono>
#include <cstring>

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
  img.resize(size_t(W)*H);
  f.read(reinterpret_cast<char*>(img.data()),img.size());
  return bool(f);
}

// CPU 15×15 mean for diff
void boxCPU(const std::vector<uint8_t>& in,std::vector<uint8_t>& out,int W,int H){
  const int R=7, D=15, denom=D*D;
  out.assign(in.size(),0);
  for(int y=R;y<H-R;++y)
    for(int x=R;x<W-R;++x){
      int s=0;
      for(int yy=-R;yy<=R;++yy)
        for(int xx=-R;xx<=R;++xx)
          s+=in[(y+yy)*W + x+xx];
      out[y*W+x]=uint8_t(s/denom);
    }
}

// GPU 15×15 mean (32×8 tile)
constexpr int R = 7, TILE_X=32, TILE_Y=8;
/* shared mem  8+2*7+8 (extra)=30 rows, 64 cols */
__global__ void box15(const uint8_t* src,size_t pitchSrc,
                      uint8_t* dst ,size_t pitchDst,
                      int W,int H,int yBase,int copyY)
{
  __shared__ uint8_t sm[30][64];

  int tx=threadIdx.x;                 // 0..31
  int ty=threadIdx.y;                 // 0..7
  int gx=blockIdx.x*TILE_X + tx;
  int gy=yBase      + blockIdx.y*TILE_Y + ty;

  for(int dy=-R; dy<TILE_Y+R; dy+=TILE_Y){            // -7,1,9
    int sy=gy+dy;
    if(sy>=0 && sy<H){
      int syRel = sy - copyY;                         // row inside stripe buffer
      // left halo
      int sx = gx - R;
      if(sx>=0 && sx<W) sm[ty+dy+R][tx] = *(src + syRel*pitchSrc + sx);
      // Right halo
      sx = gx + (TILE_X - R);                         // dx = 25
      if(sx>=0 && sx<W) sm[ty+dy+R][tx+TILE_X] = *(src + syRel*pitchSrc + sx);
    }
  }
  __syncthreads();

  if(gx>=R && gx<W-R && gy>=R && gy<H-R){
    int sum=0;
    for(int yy=0; yy<15; ++yy)
      #pragma unroll
      for(int xx=0; xx<15; ++xx)
        sum += sm[ty+yy][tx+xx];
    int dyRel = gy - copyY;
    *(dst + dyRel*pitchDst + gx) = uint8_t(sum/225);
  }
}

int main(int argc,char**argv)
{
  int W=4096,H=4096;
  std::vector<uint8_t> img;
  if(argc==2){ if(!readPGM(argv[1],img,W,H)){std::cerr<<"PGM load fail\n"; return 1;} }
  else { img.resize(size_t(W)*H); std::srand(1234); for(auto& p:img) p=std::rand()&0xFF; }

  // CPU reference
  std::vector<uint8_t> ref;
  auto c0=std::chrono::high_resolution_clock::now();
  boxCPU(img,ref,W,H);
  auto c1=std::chrono::high_resolution_clock::now();
  double cpu_ms=ms(c0,c1);

  // Stripe parameters
  const int stripeRows = 256;
  const int stripes    = (H + stripeRows - 1) / stripeRows;

  // Pinned host output
  uint8_t* h_out; CUDA_CHECK(cudaMallocHost(&h_out,img.size()));

  // Two streams + pitched buffers
  cudaStream_t S[2]; for(int i=0;i<2;++i) CUDA_CHECK(cudaStreamCreate(&S[i]));
  uint8_t *d_src[2],*d_dst[2]; size_t pitchSrc[2], pitchDst[2];
  for(int i=0;i<2;++i){
      CUDA_CHECK(cudaMallocPitch(&d_src[i],&pitchSrc[i],W,stripeRows+2*R));
      CUDA_CHECK(cudaMallocPitch(&d_dst[i],&pitchDst[i],W,stripeRows+2*R));
  }

  double h2d_ms=0, ker_ms=0, d2h_ms=0;
  cudaEvent_t wall0,wall1; CUDA_CHECK(cudaEventCreate(&wall0)); CUDA_CHECK(cudaEventCreate(&wall1));
  CUDA_CHECK(cudaEventRecord(wall0));

  for(int sIdx=0; sIdx<stripes; ++sIdx){
      int buf = sIdx & 1;
      cudaStream_t st = S[buf];

      int y0    = sIdx*stripeRows;
      int copyY = (y0==0)? 0 : y0 - R;
      int rows  = stripeRows + ((y0==0)? R : 2*R);
      if(copyY + rows > H) rows = H - copyY;

      // Events per stage
      cudaEvent_t eH0,eH1,eK0,eK1,eD0,eD1;
      CUDA_CHECK(cudaEventCreate(&eH0)); CUDA_CHECK(cudaEventCreate(&eH1));
      CUDA_CHECK(cudaEventCreate(&eK0)); CUDA_CHECK(cudaEventCreate(&eK1));
      CUDA_CHECK(cudaEventCreate(&eD0)); CUDA_CHECK(cudaEventCreate(&eD1));

      // H2D stripe copy
      CUDA_CHECK(cudaEventRecord(eH0,st));
      CUDA_CHECK(cudaMemcpy2DAsync(d_src[buf],pitchSrc[buf],
                                   img.data()+copyY*W, W,
                                   W, rows, cudaMemcpyHostToDevice, st));
      CUDA_CHECK(cudaEventRecord(eH1,st));

      // Kernel
      dim3 blk(TILE_X,TILE_Y);
      dim3 grd((W+TILE_X-1)/TILE_X, stripeRows/TILE_Y);
      CUDA_CHECK(cudaEventRecord(eK0,st));
      box15<<<grd,blk,0,st>>>(d_src[buf],pitchSrc[buf],
                              d_dst[buf],pitchDst[buf],
                              W,H,y0,copyY);
      CUDA_CHECK(cudaEventRecord(eK1,st));

      // D2H
      int procRows = (y0+stripeRows > H-R) ? (H - R - y0) : stripeRows;
      CUDA_CHECK(cudaEventRecord(eD0,st));
      CUDA_CHECK(cudaMemcpy2DAsync(h_out + y0*W, W,
                                   d_dst[buf] + (y0 - copyY)*pitchDst[buf], pitchDst[buf],
                                   W, procRows, cudaMemcpyDeviceToHost, st));
      CUDA_CHECK(cudaEventRecord(eD1,st));

      CUDA_CHECK(cudaStreamSynchronize(st));

      float t;
      CUDA_CHECK(cudaEventElapsedTime(&t,eH0,eH1)); h2d_ms += t;
      CUDA_CHECK(cudaEventElapsedTime(&t,eK0,eK1)); ker_ms += t;
      CUDA_CHECK(cudaEventElapsedTime(&t,eD0,eD1)); d2h_ms += t;
      for(auto ev:{eH0,eH1,eK0,eK1,eD0,eD1}) CUDA_CHECK(cudaEventDestroy(ev));
  }

  CUDA_CHECK(cudaEventRecord(wall1));
  CUDA_CHECK(cudaEventSynchronize(wall1));
  float wall_ms; CUDA_CHECK(cudaEventElapsedTime(&wall_ms,wall0,wall1));

  // Compare
  size_t diff=0; for(size_t i=0;i<img.size();++i) if(h_out[i]!=ref[i]) ++diff;

  std::cout<<"CPU total               : "<<cpu_ms <<" ms\n";
  std::cout<<"GPU H2D (sum)           : "<<h2d_ms<<" ms\n";
  std::cout<<"GPU kernels (sum)       : "<<ker_ms<<" ms\n";
  std::cout<<"GPU D2H (sum)           : "<<d2h_ms<<" ms\n";
  std::cout<<"GPU total               : "<<wall_ms<<" ms\n";
  std::cout<<"Mismatch pixels         : "<<diff<<"\n";

  // Cleanup
  for(int i=0;i<2;++i){
      CUDA_CHECK(cudaFree(d_src[i]));
      CUDA_CHECK(cudaFree(d_dst[i]));
      CUDA_CHECK(cudaStreamDestroy(S[i]));
  }
  CUDA_CHECK(cudaFreeHost(h_out));
  return 0;
}