#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <chrono>
#include <cstring>   // memcpy

#define CUDA_CHECK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){                 \
  std::cerr<<"CUDA error: "<<cudaGetErrorString(e)<<" ("                         \
           <<__FILE__<<':'<<__LINE__<<")\n"; std::exit(1);} }while(0)

template<class TP> inline double ms(TP a,TP b){
  return std::chrono::duration<double,std::milli>(b-a).count(); }

// PGM loader
bool readPGM(const char* fn,std::vector<uint8_t>& img,int& W,int& H){
  std::ifstream f(fn,std::ios::binary); if(!f) return false;
  std::string m; f>>m; if(m!="P5") return false;
  int maxv; f>>W>>H>>maxv; f.get(); if(maxv!=255) return false;
  img.resize(size_t(W)*H); f.read(reinterpret_cast<char*>(img.data()),img.size());
  return bool(f);
}

// CPU helpers
void integralCPU(const std::vector<uint8_t>& src,std::vector<uint32_t>& ii,
                 int W,int H){
  ii.assign(size_t(W)*H,0);
  for(int y=0;y<H;++y){
    uint32_t run=0;
    for(int x=0;x<W;++x){
      run+=src[y*W+x];
      ii[y*W+x]=run + (y? ii[(y-1)*W+x] : 0);
    }
  }
}
inline uint32_t rectSumCPU(const std::vector<uint32_t>& ii,int W,
                           int x1,int y1,int x2,int y2){
  uint32_t A=(x1>0&&y1>0)? ii[(y1-1)*W+(x1-1)]:0;
  uint32_t B=(y1>0)      ? ii[(y1-1)*W+x2]    :0;
  uint32_t C=(x1>0)      ? ii[y2*W+(x1-1)]    :0;
  uint32_t D=ii[y2*W+x2];
  return D + A - B - C;
}

// Integral-scan kernels
constexpr int TILE = 1024;
__global__ void scanRow_u8(const uint8_t* in,uint32_t* out,uint32_t* sums,
                           int W,int tiles){
  __shared__ uint32_t s[TILE];
  int row=blockIdx.y, tile=blockIdx.x;
  int gx = tile*TILE+threadIdx.x, idx=row*W+gx;
  s[threadIdx.x]=(gx<W)? uint32_t(in[idx]):0u; __syncthreads();
#pragma unroll
  for(int off=1; off<TILE; off<<=1){
    uint32_t v=(threadIdx.x>=off)?s[threadIdx.x-off]:0u;
    __syncthreads(); s[threadIdx.x]+=v; __syncthreads();
  }
  if(gx<W) out[idx]=s[threadIdx.x];
  if(threadIdx.x==TILE-1||gx==W-1) sums[row*tiles+tile]=s[threadIdx.x];
}
__global__ void scanRow_u32(uint32_t* buf,uint32_t* sums,int W,int tiles){
  __shared__ uint32_t s[TILE];
  int row=blockIdx.y, tile=blockIdx.x;
  int gx=tile*TILE+threadIdx.x, idx=row*W+gx;
  s[threadIdx.x]=(gx<W)?buf[idx]:0u; __syncthreads();
#pragma unroll
  for(int off=1; off<TILE; off<<=1){
    uint32_t v=(threadIdx.x>=off)?s[threadIdx.x-off]:0u;
    __syncthreads(); s[threadIdx.x]+=v; __syncthreads();
  }
  if(gx<W) buf[idx]=s[threadIdx.x];
  if(threadIdx.x==TILE-1||gx==W-1) sums[row*tiles+tile]=s[threadIdx.x];
}
__global__ void scanSums(uint32_t* sums,int tiles,int rows){
  int r=blockIdx.x; if(r>=rows) return;
  uint32_t acc=0;
  for(int t=0;t<tiles;++t){ uint32_t v=sums[r*tiles+t]; sums[r*tiles+t]=acc; acc+=v; }
}
__global__ void addOff(uint32_t* buf,const uint32_t* sums,int W,int tiles){
  int row=blockIdx.y, tile=blockIdx.x;
  uint32_t off=sums[row*tiles+tile];
  int gx=tile*TILE+threadIdx.x; if(gx>=W) return;
  buf[row*W+gx]+=off;
}
// 32×32 transpose
__global__ void transpose32u(const uint32_t* in,uint32_t* out,int W,int H){
  __shared__ uint32_t tile[32][33];
  int x=blockIdx.x*32+threadIdx.x, y=blockIdx.y*32+threadIdx.y;
  if(x<W&&y<H) tile[threadIdx.y][threadIdx.x]=in[y*W+x];
  __syncthreads();
  x=blockIdx.y*32+threadIdx.x; y=blockIdx.x*32+threadIdx.y;
  if(x<H&&y<W) out[y*H+x]=tile[threadIdx.x][threadIdx.y];
}

// Device rect helper
__device__ __forceinline__
uint32_t rectSumDev(const uint32_t* ii,int W,int x1,int y1,int x2,int y2){
  uint32_t A=(x1>0&&y1>0)? ii[(y1-1)*W+(x1-1)]:0;
  uint32_t B=(y1>0)      ? ii[(y1-1)*W+x2]    :0;
  uint32_t C=(x1>0)      ? ii[y2*W+(x1-1)]    :0;
  uint32_t D=ii[y2*W+x2];
  return D + A - B - C;
}

// Viola–Jones detector (one Haar stage)
constexpr int WIN    = 24;
constexpr int HALF   = WIN/2;
constexpr int STRIDE = 4;     // slide step

__global__ void vjDetector(const uint32_t* ii,uint8_t* out,
                           int W,int H,int outW,float thresh)
{
  int gx = blockIdx.x*blockDim.x + threadIdx.x;
  int gy = blockIdx.y*blockDim.y + threadIdx.y;

  int winX = gx * STRIDE;
  int winY = gy * STRIDE;
  if(winX+WIN-1>=W || winY+WIN-1>=H) return;

  int x1=winX, y1=winY;
  int x2=winX+WIN-1, y2=winY+WIN-1;
  int mid=y1+HALF-1;

  uint32_t top = rectSumDev(ii,W,x1,y1,x2,mid);
  uint32_t bot = rectSumDev(ii,W,x1,mid+1,x2,y2);
  float score=float(bot)-float(top);

  out[gy*outW + gx] = (score>thresh);
}

int main(int argc,char**argv)
{
  int W=4096,H=4096;
  std::vector<uint8_t> img;
  if(argc==2){ if(!readPGM(argv[1],img,W,H)){std::cerr<<"PGM fail\n";return 1;} }
  else { img.resize(size_t(W)*H); std::srand(1234); for(auto& p:img)p=std::rand()&0xFF; std::cout<<"Random "<<W<<"×"<<H<<"\n";}

  // CPU reference
  auto c0=std::chrono::high_resolution_clock::now();
  std::vector<uint32_t> iiCPU; integralCPU(img,iiCPU,W,H);
  const int numWinX = (W-WIN)/STRIDE + 1;
  const int numWinY = (H-WIN)/STRIDE + 1;
  const int outW = numWinX, outH=numWinY;
  std::vector<uint8_t> detCPU(size_t(outW)*outH,0);
  for(int gy=0; gy<numWinY; ++gy){
    for(int gx=0; gx<numWinX; ++gx){
      int winX=gx*STRIDE, winY=gy*STRIDE;
      int mid=winY+HALF-1;
      uint32_t top=rectSumCPU(iiCPU,W,winX,winY,winX+WIN-1,mid);
      uint32_t bot=rectSumCPU(iiCPU,W,winX,mid+1,winX+WIN-1,winY+WIN-1);
      detCPU[gy*outW+gx]=( (float(bot)-float(top)) > 50.f );
    }
  }
  auto c1=std::chrono::high_resolution_clock::now();
  double cpu_ms=ms(c0,c1);

  const size_t N=size_t(W)*H;

  // Pinned host buffers
  uint8_t *h_img; CUDA_CHECK(cudaMallocHost(&h_img,N));
  uint8_t *h_det; CUDA_CHECK(cudaMallocHost(&h_det,size_t(outW)*outH));
  std::memcpy(h_img,img.data(),N);

  // Device buffers
  uint8_t  *d_src;  CUDA_CHECK(cudaMalloc(&d_src ,N));
  uint32_t *d_tmp;  CUDA_CHECK(cudaMalloc(&d_tmp ,N*4));
  uint32_t *d_tr ;  CUDA_CHECK(cudaMalloc(&d_tr  ,N*4));
  uint32_t *d_int;  CUDA_CHECK(cudaMalloc(&d_int ,N*4));
  uint8_t  *d_det;  CUDA_CHECK(cudaMalloc(&d_det ,size_t(outW)*outH));
  int tilesRow=(W+TILE-1)/TILE;
  uint32_t* d_sums; CUDA_CHECK(cudaMalloc(&d_sums,size_t(tilesRow)*H*4));

  // Stream & events
  cudaStream_t s; CUDA_CHECK(cudaStreamCreate(&s));
  cudaEvent_t eb,ee,h0,h1,k0,k1,d0,d1;
  CUDA_CHECK(cudaEventCreate(&eb)); CUDA_CHECK(cudaEventCreate(&ee));
  CUDA_CHECK(cudaEventCreate(&h0)); CUDA_CHECK(cudaEventCreate(&h1));
  CUDA_CHECK(cudaEventCreate(&k0)); CUDA_CHECK(cudaEventCreate(&k1));
  CUDA_CHECK(cudaEventCreate(&d0)); CUDA_CHECK(cudaEventCreate(&d1));

  CUDA_CHECK(cudaEventRecord(eb));

  // H2D
  CUDA_CHECK(cudaEventRecord(h0,s));
  CUDA_CHECK(cudaMemcpyAsync(d_src,h_img,N,cudaMemcpyHostToDevice,s));
  CUDA_CHECK(cudaEventRecord(h1,s));
  CUDA_CHECK(cudaStreamSynchronize(s));

  // Kernels
  CUDA_CHECK(cudaEventRecord(k0,s));
  dim3 blk(TILE,1), grd(tilesRow,H);
  scanRow_u8<<<grd,blk,0,s>>>(d_src,d_tmp,d_sums,W,tilesRow);
  scanSums   <<<H,1,0,s>>>(d_sums,tilesRow,H);
  addOff     <<<grd,blk,0,s>>>(d_tmp,d_sums,W,tilesRow);

  transpose32u<<<dim3((W+31)/32,(H+31)/32),dim3(32,32),0,s>>>(d_tmp,d_tr,W,H);

  int tilesRowT=(H+TILE-1)/TILE;
  CUDA_CHECK(cudaFree(d_sums));
  CUDA_CHECK(cudaMalloc(&d_sums,size_t(tilesRowT)*W*4));
  dim3 grd2(tilesRowT,W);
  scanRow_u32<<<grd2,blk,0,s>>>(d_tr,d_sums,H,tilesRowT);
  scanSums    <<<W,1,0,s>>>(d_sums,tilesRowT,W);
  addOff      <<<grd2,blk,0,s>>>(d_tr,d_sums,H,tilesRowT);
  transpose32u<<<dim3((H+31)/32,(W+31)/32),dim3(32,32),0,s>>>(d_tr,d_int,H,W);

  // Detector
  dim3 blkD(32,32);
  dim3 grdD( (numWinX+blkD.x-1)/blkD.x, (numWinY+blkD.y-1)/blkD.y );
  vjDetector<<<grdD,blkD,0,s>>>(d_int,d_det,W,H,outW,50.f);

  CUDA_CHECK(cudaEventRecord(k1,s));

  // D2H
  CUDA_CHECK(cudaEventRecord(d0,s));
  CUDA_CHECK(cudaMemcpyAsync(h_det,d_det,size_t(outW)*outH,cudaMemcpyDeviceToHost,s));
  CUDA_CHECK(cudaEventRecord(d1,s));

  CUDA_CHECK(cudaEventRecord(ee,s));
  CUDA_CHECK(cudaStreamSynchronize(s));

  // Timings
  auto dt=[&](cudaEvent_t a,cudaEvent_t b){ float t;CUDA_CHECK(cudaEventElapsedTime(&t,a,b)); return double(t); };
  double tH2D=dt(h0,h1), tKer=dt(k0,k1), tD2H=dt(d0,d1), tTot=dt(eb,ee);

  // Mismatch
  size_t diff=0; for(size_t i=0;i<size_t(outW)*outH;++i) if(h_det[i]!=detCPU[i]) ++diff;

  std::cout<<"CPU total               : "<<cpu_ms<<" ms\n";
  std::cout<<"GPU H2D copy            : "<<tH2D <<" ms\n";
  std::cout<<"GPU kernels             : "<<tKer <<" ms\n";
  std::cout<<"GPU D2H copy            : "<<tD2H <<" ms\n";
  std::cout<<"GPU total               : "<<tTot <<" ms\n";
  std::cout<<"Mismatching windows     : "<<diff<<"\n";

  // Cleanup
  CUDA_CHECK(cudaFree(d_src)); CUDA_CHECK(cudaFree(d_tmp)); CUDA_CHECK(cudaFree(d_tr));
  CUDA_CHECK(cudaFree(d_int)); CUDA_CHECK(cudaFree(d_det)); CUDA_CHECK(cudaFree(d_sums));
  CUDA_CHECK(cudaFreeHost(h_img)); CUDA_CHECK(cudaFreeHost(h_det));
  CUDA_CHECK(cudaStreamDestroy(s));
  return 0;
}