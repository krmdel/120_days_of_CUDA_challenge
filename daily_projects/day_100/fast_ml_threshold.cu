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
#include <algorithm>

#define CUDA_CHECK(x)                                                             \
  do { cudaError_t _err = (x); if (_err != cudaSuccess) {                         \
       std::cerr << "CUDA error: " << cudaGetErrorString(_err)                    \
                 << " (" << __FILE__ << ':' << __LINE__ << ")\n"; std::exit(1);}  \
  } while (0)

template<class TP> inline double millis(TP a, TP b){
  return std::chrono::duration<double,std::milli>(b - a).count(); }

// PGM loader
bool readPGM(const char* fn,std::vector<uint8_t>& img,int& W,int& H){
  std::ifstream f(fn,std::ios::binary); if(!f) return false;
  std::string m; f>>m; if(m!="P5") return false;
  int maxv; f>>W>>H>>maxv; f.get(); if(maxv!=255) return false;
  img.resize(size_t(W)*H);
  f.read(reinterpret_cast<char*>(img.data()),img.size());
  return bool(f);
}

// CPU fallback FAST-9 (for frame-0 validation)
static void fastCPU(const uint8_t* img,
                    const uint8_t* thr,
                    uint8_t* flag,
                    int W,int H)
{
  const int8_t cx[16] ={0,1,2,2,2,1,0,-1,-2,-2,-2,-1,0,1,2,1};
  const int8_t cy[16] ={-3,-3,-2,-1,0,1,2,2,2,1,0,-1,-2,-3,-2,-1};
  uint16_t mask9[16];                   // contiguous 9-bit masks
  for(int s=0;s<16;++s){ uint16_t m=0; for(int k=0;k<9;++k) m |= 1u<<((s+k)&15); mask9[s]=m; }

  auto id=[&](int x,int y){return y*W+x;};
  for(int y=3;y<H-3;++y)
    for(int x=3;x<W-3;++x){
      int c = img[id(x,y)], t = thr[id(x,y)];
      uint16_t bright=0,dark=0;
      for(int i=0;i<16;++i){
        int p = img[id(x+cx[i],y+cy[i])];
        bright |= (p>=c+t)<<i;
        dark   |= (p<=c-t)<<i;
      }
      bool ok=false;
      for(int s=0;s<16 && !ok;++s){
        uint16_t m=mask9[s];
        ok = ((bright&m)==m)||((dark&m)==m);
      }
      flag[id(x,y)] = ok;
    }
}

// Constant device data
__constant__ float d_w0,d_w1;          // ML threshold
__constant__ int8_t   d_cx[16], d_cy[16];
__constant__ uint16_t d_mask9[16];

// Kernels
__global__ void gradThresh(const uint8_t* in,float* grad,uint8_t* thr,int W,int H){
  int x=blockIdx.x*16+threadIdx.x, y=blockIdx.y*16+threadIdx.y;
  if(x<1||x>=W-1||y<1||y>=H-1) return;
  int idx=y*W+x;
  int gx=int(in[idx+1])-int(in[idx-1]);
  int gy=int(in[idx+W])-int(in[idx-W]);
  float g=sqrtf(float(gx*gx+gy*gy));
  grad[idx]=g;
  thr[idx]=uint8_t(d_w0 + d_w1*g);
}

__global__ void fastKernel(const uint8_t* img,const uint8_t* thr,
                           uint8_t* flag,int W,int H){
  int x=blockIdx.x*32+threadIdx.x, y=blockIdx.y;
  if(x<3||x>=W-3||y<3||y>=H-3){ if(x<W&&y<H) flag[y*W+x]=0; return; }
  int c=img[y*W+x], t=thr[y*W+x];
  uint16_t bright=0,dark=0;
#pragma unroll
  for(int i=0;i<16;++i){
    int p=img[(y+d_cy[i])*W + x+d_cx[i]];
    bright|=(p>=c+t)<<i; dark|=(p<=c-t)<<i;
  }
  bool ok=false;
#pragma unroll
  for(int s=0;s<16 && !ok;++s){
    uint16_t m=d_mask9[s];
    ok = ((bright&m)==m)||((dark&m)==m);
  }
  flag[y*W+x]=ok;
}

__global__ void compactKernel(const uint8_t* flag,int* list,int* cnt,int N){
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<N && flag[i]) list[atomicAdd(cnt,1)]=i;
}

__device__ inline int clmp(int v,int lo,int hi){return v<lo?lo:(v>hi?hi:v);}

__global__ void orbOrient(const uint8_t* img,const int* kp,float* ang,int W,int H){
  extern __shared__ int sm[];
  int kid=blockIdx.x, tid=threadIdx.y*blockDim.x+threadIdx.x;
  int idx=kp[kid]; int cx=idx%W, cy=idx/W;
  int sumX=0,sumY=0;
  for(int y=threadIdx.y;y<31;y+=blockDim.y)
    for(int x=threadIdx.x;x<31;x+=blockDim.x){
      int ix=clmp(cx+x-15,0,W-1), iy=clmp(cy+y-15,0,H-1);
      int v=img[iy*W+ix];
      sumX+=v*(x-15); sumY+=v*(y-15);
    }
  sm[tid*2  ]=sumX; sm[tid*2+1]=sumY; __syncthreads();
  for(int s=(blockDim.x*blockDim.y)>>1;s;s>>=1){
    if(tid<s){ sm[tid*2]+=sm[(tid+s)*2]; sm[tid*2+1]+=sm[(tid+s)*2+1]; }
    __syncthreads();
  }
  if(tid==0) ang[kid]=atan2f(float(sm[1]),float(sm[0]));
}

int main(int argc,char**argv)
{
  // Load frames
  std::vector<std::vector<uint8_t>> imgs;
  int W=1920,H=1080;
  float w0=10.f,w1=0.04f;
  if(argc>1){
    for(int i=1;i<argc;++i){
      std::vector<uint8_t> im; int w,h;
      if(!readPGM(argv[i],im,w,h)){ std::cerr<<"PGM read fail\n"; return 1;}
      if(i==1){W=w;H=h;} else if(w!=W||h!=H){std::cerr<<"size mismatch\n";return 1;}
      imgs.emplace_back(std::move(im));
    }
  }else{
    for(int b=0;b<4;++b){
      std::vector<uint8_t> im(size_t(W)*H);
      std::srand(1234+b); for(auto& p:im) p=std::rand()&0xFF;
      imgs.emplace_back(std::move(im));
    }
  }
  const int B=imgs.size(); const size_t Npix=size_t(W)*H;
  std::cout<<"Batch "<<B<<" frames  ("<<W<<"×"<<H<<")\n";

  // Constants to device
  int8_t cx[16]={0,1,2,2,2,1,0,-1,-2,-2,-2,-1,0,1,2,1};
  int8_t cy[16]={-3,-3,-2,-1,0,1,2,2,2,1,0,-1,-2,-3,-2,-1};
  uint16_t m9[16]; for(int s=0;s<16;++s){uint16_t m=0;for(int k=0;k<9;++k)m|=1u<<((s+k)&15);m9[s]=m;}
  CUDA_CHECK(cudaMemcpyToSymbol(d_cx,cx,sizeof(cx)));
  CUDA_CHECK(cudaMemcpyToSymbol(d_cy,cy,sizeof(cy)));
  CUDA_CHECK(cudaMemcpyToSymbol(d_mask9,m9,sizeof(m9)));
  CUDA_CHECK(cudaMemcpyToSymbol(d_w0,&w0,sizeof(float)));
  CUDA_CHECK(cudaMemcpyToSymbol(d_w1,&w1,sizeof(float)));

  // GPU buffers
  uint8_t *d_img,*d_thr,*d_flag; float *d_grad,*d_ang;
  int *d_list,*d_cnt;
  CUDA_CHECK(cudaMalloc(&d_img ,Npix));
  CUDA_CHECK(cudaMalloc(&d_thr ,Npix));
  CUDA_CHECK(cudaMalloc(&d_flag,Npix));
  CUDA_CHECK(cudaMalloc(&d_grad,Npix*sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_ang ,Npix*sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_list,Npix*sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_cnt ,sizeof(int)));

  // Events
  cudaEvent_t eBeg,eEnd,h2d0,h2d1,k0,k1,d2h0,d2h1;
  for(cudaEvent_t* e:{&eBeg,&eEnd,&h2d0,&h2d1,&k0,&k1,&d2h0,&d2h1})
    CUDA_CHECK(cudaEventCreate(e));

  double sumH2D=0,sumKer=0,sumD2H=0,sumTot=0;
  int totalKp=0;

  for(int f=0;f<B;++f){
    CUDA_CHECK(cudaEventRecord(eBeg));

    // H2D
    CUDA_CHECK(cudaEventRecord(h2d0));
    CUDA_CHECK(cudaMemcpy(d_img,imgs[f].data(),Npix,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(h2d1));

    // Kernels
    CUDA_CHECK(cudaEventRecord(k0));
    dim3 blk(16,16), grd((W+15)/16,(H+15)/16);
    gradThresh<<<grd,blk>>>(d_img,d_grad,d_thr,W,H);
    dim3 blkF(32,1), grdF((W+31)/32,H);
    fastKernel <<<grdF,blkF>>>(d_img,d_thr,d_flag,W,H);

    CUDA_CHECK(cudaMemset(d_cnt,0,sizeof(int)));
    int threads=256, blocks=(Npix+threads-1)/threads;
    compactKernel<<<blocks,threads>>>(d_flag,d_list,d_cnt,Npix);

    int h_cnt; CUDA_CHECK(cudaMemcpy(&h_cnt,d_cnt,sizeof(int),cudaMemcpyDeviceToHost));
    if(h_cnt){
      orbOrient<<<h_cnt,dim3(32,8),32*8*2*sizeof(int)>>>(d_img,d_list,d_ang,W,H);
    }
    CUDA_CHECK(cudaEventRecord(k1));

    // D2H
    CUDA_CHECK(cudaEventRecord(d2h0));
    CUDA_CHECK(cudaMemcpy(&h_cnt,d_cnt,sizeof(int),cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(d2h1));

    CUDA_CHECK(cudaEventRecord(eEnd));
    CUDA_CHECK(cudaEventSynchronize(eEnd));

    auto ev=[&](cudaEvent_t a,cudaEvent_t b){ float m; CUDA_CHECK(cudaEventElapsedTime(&m,a,b)); return double(m); };
    sumH2D+=ev(h2d0,h2d1); sumKer+=ev(k0,k1); sumD2H+=ev(d2h0,d2h1); sumTot+=ev(eBeg,eEnd);
    totalKp+=h_cnt;

    // Validate first frame
    if(f==0){
      std::vector<uint8_t> thrCPU(Npix), flagCPU(Npix);
      // CPU threshold
      for(int y=1;y<H-1;++y)for(int x=1;x<W-1;++x){
        int idx=y*W+x;
        int gx= imgs[0][idx+1]-imgs[0][idx-1];
        int gy= imgs[0][idx+W]-imgs[0][idx-W];
        thrCPU[idx]=uint8_t(w0+w1*sqrtf(float(gx*gx+gy*gy)));
      }
      fastCPU(imgs[0].data(),thrCPU.data(),flagCPU.data(),W,H);
      std::vector<uint8_t> flagGPU(Npix);
      CUDA_CHECK(cudaMemcpy(flagGPU.data(),d_flag,Npix,cudaMemcpyDeviceToHost));
      size_t diff=0; for(size_t i=0;i<Npix;++i) if(flagCPU[i]!=flagGPU[i]) ++diff;
      std::cout<<"Frame0 flag diff : "<<100.0*diff/Npix<<" %\n";
    }
  }


  std::cout<<"GPU H2D copy         : "<<sumH2D<<" ms\n";
  std::cout<<"GPU kernels          : "<<sumKer<<" ms\n";
  std::cout<<"GPU D2H copy         : "<<sumD2H<<" ms\n";
  std::cout<<"GPU total            : "<<sumTot<<" ms ("
           <<1000.0*B/sumTot<<" fps)\n";
  std::cout<<"Total keypoints      : "<<totalKp<<"\n";

  CUDA_CHECK(cudaFree(d_img)); CUDA_CHECK(cudaFree(d_thr)); CUDA_CHECK(cudaFree(d_flag));
  CUDA_CHECK(cudaFree(d_grad));CUDA_CHECK(cudaFree(d_ang));
  CUDA_CHECK(cudaFree(d_list));CUDA_CHECK(cudaFree(d_cnt));
  return 0;
}
