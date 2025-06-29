#include <cuda_runtime.h>
#include <vector>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <chrono>
#include <random>
#include <cstring>
#include <cmath>
#include <algorithm>              // std::clamp

#if __has_include(<opencv2/imgcodecs.hpp>)
# define USE_OPENCV 1
# include <opencv2/imgcodecs.hpp>
#else
# define USE_OPENCV 0
#endif

#define CUDA_CHECK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){               \
  std::cerr<<"CUDA error: "<<cudaGetErrorString(e)<<" ("<<__LINE__<<")\n";     \
  std::exit(1);} }while(0)
template<class TP> inline double ms(TP a,TP b){
  return std::chrono::duration<double,std::milli>(b-a).count(); }

// PGM loader if no OpenCV
#if !USE_OPENCV
static bool readPGM(const char* fn,std::vector<uint8_t>& img,int& W,int& H){
  FILE* fp=fopen(fn,"rb"); if(!fp) return false;
  char m[3]; if(fscanf(fp,"%2s",m)!=1||strcmp(m,"P5")){fclose(fp);return false;}
  int maxv; if(fscanf(fp,"%d%d%d",&W,&H,&maxv)!=3||maxv!=255){fclose(fp);return false;}
  fgetc(fp); img.resize(size_t(W)*H); size_t rd=fread(img.data(),1,img.size(),fp);
  fclose(fp); return rd==img.size();
}
#endif

constexpr int MAX_PTS  = 8192;   // number of sample points
constexpr int LEVELS   = 3;      // Gaussian pyramid levels
constexpr int WIN_R    = 2;      // radius => 5×5 window
constexpr int BLOCKPTS = 32;     // threads per block (one pt per thread)


// Device helpers
__device__ __forceinline__ int clampI(int v,int lo,int hi){
  return v<lo?lo:(v>hi?hi:v);
}
// Bilinear read from flat image buffer
__device__ float bilerp_dev(const uint8_t* img,int W,int H,float x,float y){
  int x0=int(floorf(x)), y0=int(floorf(y));
  int x1=x0+1, y1=y0+1;
  float dx=x-x0, dy=y-y0;
  x0=clampI(x0,0,W-1); x1=clampI(x1,0,W-1);
  y0=clampI(y0,0,H-1); y1=clampI(y1,0,H-1);
  float a=img[y0*W+x0], b=img[y0*W+x1];
  float c=img[y1*W+x0], d=img[y1*W+x1];
  return (1-dx)*(1-dy)*a + dx*(1-dy)*b + (1-dx)*dy*c + dx*dy*d;
}

// Lucas–Kanade kernel (one pyramid level)
__global__ void lk_level(const float2*  prevPts,   // input positions (level 0 coords)
                         float2*        nextPts,   // output updated positions
                         int            nPts,
                         const uint8_t* I0,const uint8_t* I1,
                         int W,int H,float scale)
{
  int tid = blockIdx.x*blockDim.x + threadIdx.x;
  if(tid>=nPts) return;

  // Position scaled for this pyramid level
  float2 p = prevPts[tid];
  float x  = p.x*scale;
  float y  = p.y*scale;

  // Accumulate normal-equation components
  float A11=0.f,A12=0.f,A22=0.f,b1=0.f,b2=0.f;

  for(int dy=-WIN_R;dy<=WIN_R;++dy)
    for(int dx=-WIN_R;dx<=WIN_R;++dx){
      float   xx = x+dx, yy=y+dy;
      float Ix = 0.5f*(bilerp_dev(I0,W,H,xx+1,yy)-bilerp_dev(I0,W,H,xx-1,yy));
      float Iy = 0.5f*(bilerp_dev(I0,W,H,xx,yy+1)-bilerp_dev(I0,W,H,xx,yy-1));
      float It = bilerp_dev(I1,W,H,xx,yy)-bilerp_dev(I0,W,H,xx,yy);
      A11+=Ix*Ix; A12+=Ix*Iy; A22+=Iy*Iy;
      b1 +=Ix*It; b2 +=Iy*It;
    }

  float  det = A11*A22 - A12*A12 + 1e-3f;
  float2 d   = {-(A22*b1 - A12*b2)/det,
                -(-A12*b1 + A11*b2)/det};

  // Propagate back to level-0 coordinates
  nextPts[tid] = {p.x + d.x/scale,  p.y + d.y/scale};
}

// CPU reference helpers
static inline float bilerp_cpu(const std::vector<uint8_t>& im,int W,int H,
                               float x,float y)
{
  int x0=int(floorf(x)), y0=int(floorf(y));
  int x1=x0+1, y1=y0+1;
  float dx=x-x0, dy=y-y0;
  x0=std::clamp(x0,0,W-1); x1=std::clamp(x1,0,W-1);
  y0=std::clamp(y0,0,H-1); y1=std::clamp(y1,0,H-1);
  float a=im[y0*W+x0], b=im[y0*W+x1];
  float c=im[y1*W+x0], d=im[y1*W+x1];
  return (1-dx)*(1-dy)*a + dx*(1-dy)*b + (1-dx)*dy*c + dx*dy*d;
}
static void cpu_lk_level(const std::vector<uint8_t>& I0,
                         const std::vector<uint8_t>& I1,
                         int W,int H,
                         const std::vector<float2>& prev,
                         std::vector<float2>& next)
{
  next.resize(prev.size());
  for(size_t i=0;i<prev.size();++i){
    float x=prev[i].x, y=prev[i].y;
    float A11=0,A12=0,A22=0,b1=0,b2=0;
    for(int dy=-WIN_R;dy<=WIN_R;++dy)
      for(int dx=-WIN_R;dx<=WIN_R;++dx){
        float xx=x+dx, yy=y+dy;
        float Ix=0.5f*(bilerp_cpu(I0,W,H,xx+1,yy)-bilerp_cpu(I0,W,H,xx-1,yy));
        float Iy=0.5f*(bilerp_cpu(I0,W,H,xx,yy+1)-bilerp_cpu(I0,W,H,xx,yy-1));
        float It=bilerp_cpu(I1,W,H,xx,yy)-bilerp_cpu(I0,W,H,xx,yy);
        A11+=Ix*Ix; A12+=Ix*Iy; A22+=Iy*Iy;
        b1 +=Ix*It; b2 +=Iy*It;
      }
    float det=A11*A22-A12*A12+1e-3f;
    float2 d={-(A22*b1-A12*b2)/det, -(-A12*b1+A11*b2)/det};
    next[i]={x+d.x, y+d.y};
  }
}

// Gaussian downsample (very small, separable 5-tap box)
static void pyrDown(const std::vector<uint8_t>& src,int W,int H,
                    std::vector<uint8_t>& dst,int& Wd,int& Hd)
{
  Wd=W/2; Hd=H/2; dst.resize(size_t(Wd)*Hd);
  auto at=[&](int x,int y){ x=std::clamp(x,0,W-1); y=std::clamp(y,0,H-1);
                            return src[y*W+x]; };
  for(int y=0;y<Hd;++y)
    for(int x=0;x<Wd;++x){
      int sx=x*2, sy=y*2;
      int s=4*at(sx,sy)+2*(at(sx+1,sy)+at(sx-1,sy)+at(sx,sy+1)+at(sx,sy-1))
           +(at(sx+1,sy+1)+at(sx-1,sy+1)+at(sx+1,sy-1)+at(sx-1,sy-1));
      dst[y*Wd+x]=uint8_t(s/16);
    }
}

int main(int argc,char** argv)
{
  // Load or generate test frames
  int W,H; std::vector<uint8_t> img0,img1;
  if(argc>=3){
#if USE_OPENCV
    cv::Mat m0=cv::imread(argv[1],cv::IMREAD_GRAYSCALE);
    cv::Mat m1=cv::imread(argv[2],cv::IMREAD_GRAYSCALE);
    if(m0.empty()||m1.empty()){ std::cerr<<"load error\n"; return 1; }
    if(m0.size()!=m1.size()){ std::cerr<<"size mismatch\n"; return 1; }
    W=m0.cols; H=m0.rows;
    img0.assign(m0.datastart,m0.dataend);
    img1.assign(m1.datastart,m1.dataend);
#else
    if(!readPGM(argv[1],img0,W,H)||!readPGM(argv[2],img1,W,H)){
      std::cerr<<"PGM load error\n"; return 1; }
#endif
  }else{
    W=1920; H=1080;
    img0.resize(size_t(W)*H); img1.resize(img0.size());
    std::mt19937 rng(0); std::uniform_int_distribution<int>d(0,255);
    for(auto& v:img0) v=d(rng);
    for(auto& v:img1) v=d(rng);
    std::cout<<"[Info] random frames "<<W<<"×"<<H<<"\n";
  }

  // Build CPU pyramids (3 levels)
  std::vector<uint8_t> pyr0[LEVELS], pyr1[LEVELS];
  int Ws[LEVELS], Hs[LEVELS];
  pyr0[0]=img0; pyr1[0]=img1; Ws[0]=W; Hs[0]=H;
  for(int l=1;l<LEVELS;++l){
    pyrDown(pyr0[l-1],Ws[l-1],Hs[l-1],pyr0[l],Ws[l],Hs[l]);
    pyrDown(pyr1[l-1],Ws[l-1],Hs[l-1],pyr1[l],Ws[l],Hs[l]);
  }

  // Generate regular grid of points
  std::vector<float2> pts0; pts0.reserve(MAX_PTS);
  int step=std::max(8, int(std::sqrt(float(W*H)/MAX_PTS)));
  for(int y=WIN_R; y<H-WIN_R; y+=step)
    for(int x=WIN_R; x<W-WIN_R; x+=step)
      if(pts0.size()<MAX_PTS) pts0.push_back({float(x),float(y)});
  int nPts=pts0.size();

  // CPU reference flow
  auto tc0=std::chrono::high_resolution_clock::now();
  std::vector<float2> cpuPts=pts0,tmp;
  for(int l=LEVELS-1;l>=0;--l){
    cpu_lk_level(pyr0[l],pyr1[l],Ws[l],Hs[l],cpuPts,tmp);
    float sc = 1.0f/(1<<l);
    for(int i=0;i<nPts;++i){ cpuPts[i].x=tmp[i].x/sc; cpuPts[i].y=tmp[i].y/sc; }
  }
  auto tc1=std::chrono::high_resolution_clock::now();
  double cpu_ms = ms(tc0,tc1);

  // Upload pyramids to GPU
  uint8_t* dImg0[LEVELS]; uint8_t* dImg1[LEVELS];
  for(int l=0;l<LEVELS;++l){
    CUDA_CHECK(cudaMalloc(&dImg0[l], Ws[l]*Hs[l]));
    CUDA_CHECK(cudaMalloc(&dImg1[l], Ws[l]*Hs[l]));
    CUDA_CHECK(cudaMemcpy(dImg0[l],pyr0[l].data(),Ws[l]*Hs[l],cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dImg1[l],pyr1[l].data(),Ws[l]*Hs[l],cudaMemcpyHostToDevice));
  }

  float2 *dIn,*dOut;
  CUDA_CHECK(cudaMalloc(&dIn ,nPts*sizeof(float2)));
  CUDA_CHECK(cudaMalloc(&dOut,nPts*sizeof(float2)));
  CUDA_CHECK(cudaMemcpy(dIn,pts0.data(),nPts*sizeof(float2),cudaMemcpyHostToDevice));

  // Events for timing
  cudaEvent_t eK0,eK1,eD0,eD1,eTot0,eTot1;
  for(auto p:{&eK0,&eK1,&eD0,&eD1,&eTot0,&eTot1}) CUDA_CHECK(cudaEventCreate(p));

  cudaEventRecord(eTot0);
  cudaEventRecord(eK0);

  // GPU LK pyramid (coarse→fine)
  dim3 blk(BLOCKPTS);
  for(int l=LEVELS-1;l>=0;--l){
    dim3 grd((nPts+BLOCKPTS-1)/BLOCKPTS);
    float scale = 1.0f/(1<<l);
    lk_level<<<grd,blk>>>( (l==LEVELS-1)?dIn:dOut,
                           dOut,
                           nPts,dImg0[l],dImg1[l],Ws[l],Hs[l],scale);
    CUDA_CHECK(cudaGetLastError());
  }
  cudaEventRecord(eK1);

  // Download results
  std::vector<float2> gpuPts(nPts);
  cudaEventRecord(eD0);
  CUDA_CHECK(cudaMemcpy(gpuPts.data(),dOut,nPts*sizeof(float2),cudaMemcpyDeviceToHost));
  cudaEventRecord(eD1);
  cudaEventRecord(eTot1); CUDA_CHECK(cudaEventSynchronize(eTot1));

  // Timing breakdown
  float tKer,tD2H,gpuTot;
  cudaEventElapsedTime(&tKer ,eK0,eK1);
  cudaEventElapsedTime(&tD2H,eD0,eD1);
  cudaEventElapsedTime(&gpuTot,eTot0,eTot1);


  std::cout<<"\nFrame size              : "<<W<<" × "<<H<<"\n";
  std::cout<<"Tracked points          : "<<nPts<<"\n\n";
  std::cout<<"GPU kernels             : "<<tKer  <<" ms\n";
  std::cout<<"GPU D2H copy            : "<<tD2H  <<" ms\n";
  std::cout<<"GPU total               : "<<gpuTot<<" ms\n";
  std::cout<<"CPU total               : "<<cpu_ms <<" ms\n\n";
  std::cout<<"First 8 flow vectors (GPU):\n";
  for(int i=0;i<8&&i<nPts;++i){
    auto a=pts0[i], b=gpuPts[i];
    std::cout<<"("<<a.x<<','<<a.y<<")->("<<b.x<<','<<b.y<<")  "
             <<"dx="<<b.x-a.x<<" dy="<<b.y-a.y<<"\n";
  }

  // Cleanup
  for(int l=0;l<LEVELS;++l){ CUDA_CHECK(cudaFree(dImg0[l])); CUDA_CHECK(cudaFree(dImg1[l])); }
  CUDA_CHECK(cudaFree(dIn)); CUDA_CHECK(cudaFree(dOut));
  return 0;
}
