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

#define CUDA_CHECK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){                \
  std::cerr<<"CUDA error: "<<cudaGetErrorString(e)<<" ("<<__LINE__<<")\n";      \
  std::exit(1);} }while(0)
template<class T> inline double ms(T a,T b){
  return std::chrono::duration<double,std::milli>(b-a).count(); }

// PGM / PPM loaders
static bool readPGM16(const char* fn,std::vector<uint16_t>& img,int& W,int& H){
  FILE* fp=fopen(fn,"rb"); if(!fp){ return false; }
  char m[3]; if(fscanf(fp,"%2s",m)!=1||strcmp(m,"P5")){ fclose(fp); return false;}
  int maxv; if(fscanf(fp,"%d%d%d",&W,&H,&maxv)!=3||maxv>65535){fclose(fp); return false;}
  fgetc(fp);
  img.resize(size_t(W)*H);
  size_t rd=fread(img.data(),2,img.size(),fp);
  fclose(fp); return rd==img.size();
}
static bool readPPM(const char* fn,std::vector<uint8_t>& img,int& W,int& H){
  FILE* fp=fopen(fn,"rb"); if(!fp){ return false; }
  char m[3]; if(fscanf(fp,"%2s",m)!=1||strcmp(m,"P6")){ fclose(fp); return false;}
  int maxv; if(fscanf(fp,"%d%d%d",&W,&H,&maxv)!=3||maxv!=255){fclose(fp);return false;}
  fgetc(fp);
  img.resize(size_t(W)*H*3);
  size_t rd=fread(img.data(),1,img.size(),fp);
  fclose(fp); return rd==img.size();
}

// GPU kernel
__global__ void rgbd2pcKer(const uint16_t* __restrict__ depth,
                           const uchar3*  __restrict__ rgb,
                           float4*        xyz,
                           uchar4*        rgbOut,
                           int W,int H,
                           float fx,float fy,float cx,float cy,
                           float scale)
{
  int x=blockIdx.x*blockDim.x+threadIdx.x;
  int y=blockIdx.y*blockDim.y+threadIdx.y;
  if(x>=W||y>=H) return;

  int idx=y*W+x;
  uint16_t d = depth[idx];
  float z = d*scale;          // metres
  float4 p;
  if(d==0){                   // invalid depth → NaNs
    p = make_float4(NAN,NAN,NAN,1.f);
  }else{
    p.x = (x-cx)*z/fx;
    p.y = (y-cy)*z/fy;
    p.z =  z;
    p.w =  1.f;
  }
  xyz[idx]=p;
  uchar3 c = rgb[idx];
  rgbOut[idx] = make_uchar4(c.x,c.y,c.z,255);
}

// CPU reference
static void rgbd2pcCPU(const std::vector<uint16_t>& dep,
                       const std::vector<uint8_t>&  col,
                       int W,int H,
                       float fx,float fy,float cx,float cy,
                       float scale,
                       std::vector<float4>& xyz,
                       std::vector<uchar4>& rgb)
{
  xyz.resize(size_t(W)*H);
  rgb.resize(xyz.size());
  for(int y=0;y<H;++y)
    for(int x=0;x<W;++x){
      int idx=y*W+x;
      uint16_t d=dep[idx];
      float4 p;
      if(d==0){ p.x=p.y=p.z=NAN; p.w=1.f; }
      else{
        float z=d*scale;
        p.x=(x-cx)*z/fx;
        p.y=(y-cy)*z/fy;
        p.z=z; p.w=1.f;
      }
      xyz[idx]=p;
      rgb[idx]=make_uchar4(col[3*idx+0],col[3*idx+1],col[3*idx+2],255);
    }
}

int main(int argc,char** argv)
{
  // Input
  int W,H;
  std::vector<uint8_t>  color;
  std::vector<uint16_t> depth;

  float fx=525, fy=525, cx=319.5f, cy=239.5f;  // default intrinsics
  if(argc>=7){ fx=std::stof(argv[3]); fy=std::stof(argv[4]);
               cx=std::stof(argv[5]); cy=std::stof(argv[6]); }
  if(argc>=3){
    if(!readPPM(argv[1],color,W,H)||!readPGM16(argv[2],depth,W,H)){
      std::cerr<<"Failed to load RGB-D images\n"; return 1; }
  }else{
    /* random 4 K frame */
    W=3840; H=2160;
    color.resize(size_t(W)*H*3);
    depth.resize(size_t(W)*H);
    std::mt19937 rng(0); std::uniform_int_distribution<int> d8(0,255);
    std::uniform_int_distribution<int> d16(400,6000);  // 0.4-6 m
    for(auto& v:color) v=d8(rng);
    for(auto& v:depth) v=d16(rng);
    std::cout<<"[Info] random RGB-D "<<W<<"×"<<H<<"\n";
  }
  const int N=W*H;
  const float SCALE = 0.001f;     // mm → m

  // CPU
  auto tc0=std::chrono::high_resolution_clock::now();
  std::vector<float4> xyzCPU; std::vector<uchar4> rgbCPU;
  rgbd2pcCPU(depth,color,W,H,fx,fy,cx,cy,SCALE,xyzCPU,rgbCPU);
  auto tc1=std::chrono::high_resolution_clock::now();
  double cpu_ms = ms(tc0,tc1);

  // GPU buffers
  uint16_t* dDepth; uchar3* dRGB;
  float4* dXYZ; uchar4* dRGBA;
  CUDA_CHECK(cudaMalloc(&dDepth,N*sizeof(uint16_t)));
  CUDA_CHECK(cudaMalloc(&dRGB  ,N*sizeof(uchar3)));
  CUDA_CHECK(cudaMalloc(&dXYZ  ,N*sizeof(float4)));
  CUDA_CHECK(cudaMalloc(&dRGBA ,N*sizeof(uchar4)));

  cudaEvent_t eH0,eH1,eK0,eK1,eD0,eD1,eTot0,eTot1;
  for(auto p:{&eH0,&eH1,&eK0,&eK1,&eD0,&eD1,&eTot0,&eTot1}) CUDA_CHECK(cudaEventCreate(p));

  cudaEventRecord(eTot0);

  // H2D
  cudaEventRecord(eH0);
  CUDA_CHECK(cudaMemcpy(dDepth,depth.data(),N*2,cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dRGB  ,color.data(),N*3,cudaMemcpyHostToDevice));
  cudaEventRecord(eH1);

  // Kernel
  dim3 blk(16,16), grd((W+15)/16,(H+15)/16);
  cudaEventRecord(eK0);
  rgbd2pcKer<<<grd,blk>>>(dDepth,dRGB,dXYZ,dRGBA,W,H,fx,fy,cx,cy,SCALE);
  cudaEventRecord(eK1);

  // D2H
  std::vector<float4> xyzGPU(N); std::vector<uchar4> rgbGPU(N);
  cudaEventRecord(eD0);
  CUDA_CHECK(cudaMemcpy(xyzGPU.data(),dXYZ,N*sizeof(float4),cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(rgbGPU.data(),dRGBA,N*sizeof(uchar4),cudaMemcpyDeviceToHost));
  cudaEventRecord(eD1);

  cudaEventRecord(eTot1); CUDA_CHECK(cudaEventSynchronize(eTot1));

  // Timings
  float tH2D,tKer,tD2H,tTot;
  cudaEventElapsedTime(&tH2D,eH0,eH1);
  cudaEventElapsedTime(&tKer ,eK0,eK1);
  cudaEventElapsedTime(&tD2H,eD0,eD1);
  cudaEventElapsedTime(&tTot ,eTot0,eTot1);

  std::cout<<"\nImage size              : "<<W<<" × "<<H<<"  ("<<N<<" px)\n";
  std::cout<<"Intrinsics fx fy cx cy  : "<<fx<<' '<<fy<<' '<<cx<<' '<<cy<<"\n\n";
  std::cout<<"GPU H2D copy            : "<<tH2D <<" ms\n";
  std::cout<<"GPU kernel              : "<<tKer <<" ms\n";
  std::cout<<"GPU D2H copy            : "<<tD2H <<" ms\n";
  std::cout<<"GPU total               : "<<tTot <<" ms\n";
  std::cout<<"CPU total               : "<<cpu_ms<<" ms\n";

  // Cleanup
  CUDA_CHECK(cudaFree(dDepth)); CUDA_CHECK(cudaFree(dRGB));
  CUDA_CHECK(cudaFree(dXYZ));   CUDA_CHECK(cudaFree(dRGBA));
  return 0;
}
