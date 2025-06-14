#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <chrono>
#include <cstring>    // memcpy

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

// CPU reference
void integralCPU(const std::vector<uint8_t>& src,std::vector<uint32_t>& ii,int W,int H){
  ii.assign(size_t(W)*H,0);
  for(int y=0;y<H;++y){
    uint32_t row=0;
    for(int x=0;x<W;++x){
      row+=src[y*W+x];
      ii[y*W+x]=row + (y? ii[(y-1)*W+x] : 0);
    }
  }
}
inline uint32_t rectSumCPU(const std::vector<uint32_t>& ii,int W,
                           int x1,int y1,int x2,int y2){   // inclusive coords
  uint32_t A=(x1>0&&y1>0)? ii[(y1-1)*W+(x1-1)]:0;
  uint32_t B=(y1>0)      ? ii[(y1-1)*W+ x2   ]:0;
  uint32_t C=(x1>0)      ? ii[ y2   *W+(x1-1)]:0;
  uint32_t D=ii[y2*W+x2];
  return D + A - B - C;
}

// Integral scan (GPU)
constexpr int TILE = 1024;
__global__ void scanRow_u8(const uint8_t* in,uint32_t* out,uint32_t* sums,int W,int tiles){
  __shared__ uint32_t s[TILE];
  int row=blockIdx.y, tile=blockIdx.x;
  int gx=tile*TILE+threadIdx.x, idx=row*W+gx;
  s[threadIdx.x]=(gx<W)? uint32_t(in[idx]):0u;  __syncthreads();
#pragma unroll
  for(int o=1;o<TILE;o<<=1){
    uint32_t v = (threadIdx.x>=o)? s[threadIdx.x-o]:0u;
    __syncthreads(); s[threadIdx.x]+=v; __syncthreads();
  }
  if(gx<W) out[idx]=s[threadIdx.x];
  if(threadIdx.x==TILE-1||gx==W-1) sums[row*tiles+tile]=s[threadIdx.x];
}
__global__ void scanRow_u32(uint32_t* buf,uint32_t* sums,int W,int tiles){
  __shared__ uint32_t s[TILE];
  int row=blockIdx.y, tile=blockIdx.x;
  int gx=tile*TILE+threadIdx.x, idx=row*W+gx;
  s[threadIdx.x]=(gx<W)? buf[idx]:0u; __syncthreads();
#pragma unroll
  for(int o=1;o<TILE;o<<=1){
    uint32_t v=(threadIdx.x>=o)? s[threadIdx.x-o]:0u;
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
__global__ void addOffsets(uint32_t* buf,const uint32_t* sums,int W,int tiles){
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

// Box (15×15) & Haar (24×24 horizontal)
constexpr int BOX = 15;
constexpr int HAAR= 24;
__device__ __forceinline__ uint32_t rectSumDev(const uint32_t* ii,int W,
                                               int x1,int y1,int x2,int y2){
  uint32_t A=(x1>0&&y1>0)? ii[(y1-1)*W+(x1-1)]:0;
  uint32_t B=(y1>0)      ? ii[(y1-1)*W+ x2   ]:0;
  uint32_t C=(x1>0)      ? ii[ y2   *W+(x1-1)]:0;
  uint32_t D=ii[y2*W+x2];
  return D + A - B - C;
}
__global__ void boxKernel(const uint32_t* ii,uint16_t* out,int W,int H){
  int x=blockIdx.x*32+threadIdx.x;
  int y=blockIdx.y*8 +threadIdx.y;
  if(x>=W||y>=H) return;
  int h=BOX/2;
  if(x<h||y<h||x+h>=W||y+h>=H){ out[y*W+x]=0; return; }
  int x1=x-h, y1=y-h, x2=x+h, y2=y+h;
  uint32_t sum=rectSumDev(ii,W,x1,y1,x2,y2);
  out[y*W+x]=uint16_t(sum/(BOX*BOX));            // average
}
__global__ void haarKernel(const uint32_t* ii,int32_t* resp,int W,int H){
  int x=blockIdx.x*32+threadIdx.x;
  int y=blockIdx.y*8 +threadIdx.y;
  if(x>=W||y>=H) return;
  int h=HAAR/2;
  if(x<h||y<h||x+h-1>=W||y+h-1>=H){ resp[y*W+x]=0; return; }
  int x1=x-h, y1=y-h, x2=x+h-1, y2=y+h-1;
  int mid = y1+h-1;
  uint32_t top = rectSumDev(ii,W,x1,y1,x2,mid);
  uint32_t bot = rectSumDev(ii,W,x1,mid+1,x2,y2);
  resp[y*W+x]=int32_t(bot)-int32_t(top);
}

int main(int argc,char**argv)
{
  int W=4096,H=4096;
  std::vector<uint8_t> img;
  if(argc==2){ if(!readPGM(argv[1],img,W,H)){std::cerr<<"PGM load fail\n";return 1;} }
  else { img.resize(size_t(W)*H); std::srand(1234); for(auto& p:img)p=std::rand()&0xFF; std::cout<<"Random "<<W<<"×"<<H<<"\n";}

  // CPU reference (integral + filters)
  auto tCpu0=std::chrono::high_resolution_clock::now();
  std::vector<uint32_t> iiCPU; integralCPU(img,iiCPU,W,H);
  std::vector<uint16_t> boxCPU(size_t(W)*H,0);
  std::vector<int32_t>  haarCPU(size_t(W)*H,0);
  for(int y=0;y<H;++y)for(int x=0;x<W;++x){
    if(x>=BOX/2&&y>=BOX/2&&x+BOX/2<W&&y+BOX/2<H){
      int x1=x-BOX/2, y1=y-BOX/2, x2=x+BOX/2, y2=y+BOX/2;
      boxCPU[y*W+x]=rectSumCPU(iiCPU,W,x1,y1,x2,y2)/(BOX*BOX);
    }
    if(x>=HAAR/2&&y>=HAAR/2&&x+HAAR/2-1<W&&y+HAAR/2-1<H){
      int x1=x-HAAR/2, y1=y-HAAR/2, x2=x+HAAR/2-1, y2=y+HAAR/2-1;
      int mid=y1+HAAR/2-1;
      uint32_t top=rectSumCPU(iiCPU,W,x1,y1,x2,mid);
      uint32_t bot=rectSumCPU(iiCPU,W,x1,mid+1,x2,y2);
      haarCPU[y*W+x]=int32_t(bot)-int32_t(top);
    }
  }
  auto tCpu1=std::chrono::high_resolution_clock::now();
  double cpu_ms=ms(tCpu0,tCpu1);

  const size_t N=size_t(W)*H;

  // Pinned host buffers
  uint8_t  *h_img ; CUDA_CHECK(cudaMallocHost(&h_img ,N));
  uint16_t *h_box ; CUDA_CHECK(cudaMallocHost(&h_box ,N*sizeof(uint16_t)));
  int32_t  *h_haar; CUDA_CHECK(cudaMallocHost(&h_haar,N*sizeof(int32_t)));
  std::memcpy(h_img,img.data(),N);

  // Device buffers
  uint8_t  *d_src;  CUDA_CHECK(cudaMalloc(&d_src ,N));
  uint32_t *d_tmp;  CUDA_CHECK(cudaMalloc(&d_tmp ,N*4));
  uint32_t *d_tr ;  CUDA_CHECK(cudaMalloc(&d_tr  ,N*4));
  uint32_t *d_int;  CUDA_CHECK(cudaMalloc(&d_int ,N*4));
  uint16_t *d_box;  CUDA_CHECK(cudaMalloc(&d_box ,N*2));
  int32_t  *d_haar; CUDA_CHECK(cudaMalloc(&d_haar,N*4));
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
  addOffsets <<<grd,blk,0,s>>>(d_tmp,d_sums,W,tilesRow);

  dim3 blkT(32,32), grdT((W+31)/32,(H+31)/32);
  transpose32u<<<grdT,blkT,0,s>>>(d_tmp,d_tr,W,H);

  int tilesRowT=(H+TILE-1)/TILE;
  CUDA_CHECK(cudaFree(d_sums));
  CUDA_CHECK(cudaMalloc(&d_sums,size_t(tilesRowT)*W*4));
  dim3 grd2(tilesRowT,W);
  scanRow_u32<<<grd2,blk,0,s>>>(d_tr,d_sums,H,tilesRowT);
  scanSums    <<<W,1,0,s>>>(d_sums,tilesRowT,W);
  addOffsets  <<<grd2,blk,0,s>>>(d_tr,d_sums,H,tilesRowT);
  transpose32u<<<dim3((H+31)/32,(W+31)/32),blkT,0,s>>>(d_tr,d_int,H,W);

  dim3 blkF(32,8), grdF((W+31)/32,(H+7)/8);
  boxKernel <<<grdF,blkF,0,s>>>(d_int,d_box ,W,H);
  haarKernel<<<grdF,blkF,0,s>>>(d_int,d_haar,W,H);

  CUDA_CHECK(cudaEventRecord(k1,s));

  // D2H
  CUDA_CHECK(cudaEventRecord(d0,s));
  CUDA_CHECK(cudaMemcpyAsync(h_box ,d_box ,N*2,cudaMemcpyDeviceToHost,s));
  CUDA_CHECK(cudaMemcpyAsync(h_haar,d_haar,N*4,cudaMemcpyDeviceToHost,s));
  CUDA_CHECK(cudaEventRecord(d1,s));

  CUDA_CHECK(cudaEventRecord(ee,s));
  CUDA_CHECK(cudaStreamSynchronize(s));

  // Timings
  auto dt=[&](cudaEvent_t a,cudaEvent_t b){ float t;CUDA_CHECK(cudaEventElapsedTime(&t,a,b)); return double(t); };
  double tH2D=dt(h0,h1), tKer=dt(k0,k1), tD2H=dt(d0,d1), tTot=dt(eb,ee);

  // Correctness
  size_t diffBox=0,diffHaar=0;
  for(size_t i=0;i<N;++i){ if(h_box[i]!=boxCPU[i]) ++diffBox;
                           if(h_haar[i]!=haarCPU[i]) ++diffHaar; }

  std::cout<<"CPU total (integral+filters) : "<<cpu_ms<<" ms\n";
  std::cout<<"GPU H2D copy                 : "<<tH2D <<" ms\n";
  std::cout<<"GPU kernels                  : "<<tKer <<" ms\n";
  std::cout<<"GPU D2H copy                 : "<<tD2H <<" ms\n";
  std::cout<<"GPU total                    : "<<tTot <<" ms\n";
  std::cout<<"Box mismatches               : "<<diffBox<<"\n";
  std::cout<<"Haar mismatches              : "<<diffHaar<<"\n";

  // Cleanup
  CUDA_CHECK(cudaFree(d_src)); CUDA_CHECK(cudaFree(d_tmp)); CUDA_CHECK(cudaFree(d_tr));
  CUDA_CHECK(cudaFree(d_int)); CUDA_CHECK(cudaFree(d_box)); CUDA_CHECK(cudaFree(d_haar));
  CUDA_CHECK(cudaFree(d_sums));
  CUDA_CHECK(cudaFreeHost(h_img)); CUDA_CHECK(cudaFreeHost(h_box)); CUDA_CHECK(cudaFreeHost(h_haar));
  CUDA_CHECK(cudaStreamDestroy(s));
  return 0;
}