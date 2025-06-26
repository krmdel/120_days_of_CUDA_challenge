#include <cuda_runtime.h>
#include <vector>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <chrono>
#include <random>
#include <cstring>

#if __has_include(<opencv2/imgcodecs.hpp>)
# define USE_OPENCV 1
# include <opencv2/imgcodecs.hpp>
#else
# define USE_OPENCV 0
#endif

// PGM reader
#if !USE_OPENCV
static bool readPGM(const char* fn,std::vector<uint8_t>& img,int& W,int& H){
    FILE* fp=fopen(fn,"rb"); if(!fp) return false;
    char m[3]; if(fscanf(fp,"%2s",m)!=1||strcmp(m,"P5")){fclose(fp);return false;}
    int maxv; if(fscanf(fp,"%d%d%d",&W,&H,&maxv)!=3||maxv!=255){fclose(fp);return false;}
    fgetc(fp);                                       // consume single whitespace
    img.resize(size_t(W)*H);
    size_t rd=fread(img.data(),1,img.size(),fp);
    fclose(fp);
    return rd==img.size();                           // full image read?
}
#endif

// Helpers
#define CUDA_CHECK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){                 \
    std::cerr<<"CUDA error: "<<cudaGetErrorString(e)<<" ("<<__LINE__<<")\n";     \
    std::exit(1);} }while(0)

template<class T> double ms(T a,T b){
    return std::chrono::duration<double,std::milli>(b-a).count(); }

// Constants
constexpr int MAX_KP = 80000;   // capacity returned to host

// Circle offsets (radius = 8-9 px)
__device__ __constant__ int8_t c_cx[16]={0, 3, 6, 8, 8, 6, 3, 0,-3,-6,-8,-8,-6,-3, 0, 0};
__device__ __constant__ int8_t c_cy[16]={8, 6, 3, 0,-3,-6,-8,-8,-8,-6,-3, 0, 3, 6, 8, 8};

struct KP { short x,y; };

// FAST-9 GPU kernel (global mem only)
__global__ void fast9Kernel(const uint8_t* img,int W,int H,int thr,
                            KP* out,int* d_cnt)
{
    int x=blockIdx.x*blockDim.x+threadIdx.x;
    int y=blockIdx.y*blockDim.y+threadIdx.y;
    if(x<16||y<16||x>=W-16||y>=H-16) return;

    int idx=y*W+x;
    uint8_t p=img[idx];

    int bright=0,dark=0;                 // early reject (4 compass pts @ r=9)
    if(img[idx+ 9]   >p+thr) ++bright; else if(img[idx+ 9]   <p-thr) ++dark;
    if(img[idx- 9]   >p+thr) ++bright; else if(img[idx- 9]   <p-thr) ++dark;
    if(img[idx+ 9*W] >p+thr) ++bright; else if(img[idx+ 9*W] <p-thr) ++dark;
    if(img[idx- 9*W] >p+thr) ++bright; else if(img[idx- 9*W] <p-thr) ++dark;
    if(bright<3 && dark<3) return;

    int consec=0,maxc=0;                 // full 16-pixel circle test
#pragma unroll
    for(int i=0;i<24;++i){
        int ii=i&15;
        uint8_t q=img[(y+c_cy[ii])*W + (x+c_cx[ii])];
        if(q>p+thr||q<p-thr) ++consec; else consec=0;
        if(consec>maxc) maxc=consec;
        if(maxc>=9) break;
    }
    if(maxc<9) return;

    int id=atomicAdd(d_cnt,1);           // write key-point
    if(id<MAX_KP) out[id]={short(x),short(y)};
}

// CPU FAST-9 (scalar) for timing
static void cpuFAST(const std::vector<uint8_t>& img,int W,int H,int thr,
                    std::vector<KP>& out)
{
    static const int cx[16]={0,3,6,8,8,6,3,0,-3,-6,-8,-8,-6,-3,0,0};
    static const int cy[16]={8,6,3,0,-3,-6,-8,-8,-8,-6,-3,0,3,6,8,8};
    auto at=[&](int x,int y){ return img[y*W+x]; };

    for(int y=16;y<H-16;++y)
        for(int x=16;x<W-16;++x){
            int p=at(x,y);
            int bright=0,dark=0;
            if(at(x,y+9)>p+thr) ++bright; else if(at(x,y+9)<p-thr) ++dark;
            if(at(x,y-9)>p+thr) ++bright; else if(at(x,y-9)<p-thr) ++dark;
            if(at(x+9,y)>p+thr) ++bright; else if(at(x+9,y)<p-thr) ++dark;
            if(at(x-9,y)>p+thr) ++bright; else if(at(x-9,y)<p-thr) ++dark;
            if(bright<3 && dark<3) continue;

            int consec=0,maxc=0;
            for(int i=0;i<24;++i){
                int q=at(x+cx[i&15],y+cy[i&15]);
                if(q>p+thr||q<p-thr) ++consec; else consec=0;
                if(consec>maxc) maxc=consec;
                if(maxc>=9) break;
            }
            if(maxc>=9) out.push_back({short(x),short(y)});
        }
}

int main(int argc,char** argv)
{
    int thr=(argc>=3)?std::stoi(argv[2]):20;

    int W,H; std::vector<uint8_t> img;
    if(argc>=2){
#if USE_OPENCV
        auto m=cv::imread(argv[1],cv::IMREAD_GRAYSCALE);
        if(m.empty()){ std::cerr<<"load error\n"; return 1; }
        W=m.cols; H=m.rows; img.assign(m.datastart,m.dataend);
#else
        if(!readPGM(argv[1],img,W,H)){ std::cerr<<"PGM load error\n"; return 1; }
#endif
    }else{
        W=1920; H=1080; img.resize(size_t(W)*H);
        std::mt19937 rng(0); std::uniform_int_distribution<int>d(0,255);
        for(auto& v:img) v=d(rng);
        std::cout<<"[Info] random frame "<<W<<"×"<<H<<"\n";
    }
    const int N=W*H;

    // CPU reference
    auto t0=std::chrono::high_resolution_clock::now();
    std::vector<KP> cpuKP; cpuFAST(img,W,H,thr,cpuKP);
    auto t1=std::chrono::high_resolution_clock::now();
    double cpu_ms=ms(t0,t1);

    // GPU buffers
    uint8_t* d_img; CUDA_CHECK(cudaMalloc(&d_img,N));
    KP*      d_kp;  CUDA_CHECK(cudaMalloc(&d_kp,MAX_KP*sizeof(KP)));
    int*     d_cnt; CUDA_CHECK(cudaMalloc(&d_cnt,4));

    cudaEvent_t h0,h1,k0,k1,d0,d1,tBeg,tEnd;
    for(auto p:{&h0,&h1,&k0,&k1,&d0,&d1,&tBeg,&tEnd}) CUDA_CHECK(cudaEventCreate(p));

    cudaEventRecord(tBeg);

    cudaEventRecord(h0);
    CUDA_CHECK(cudaMemcpy(d_img,img.data(),N,cudaMemcpyHostToDevice));
    cudaEventRecord(h1);

    CUDA_CHECK(cudaMemset(d_cnt,0,4));
    dim3 blk(16,16), grd((W+15)/16,(H+15)/16);

    cudaEventRecord(k0);
    fast9Kernel<<<grd,blk>>>(d_img,W,H,thr,d_kp,d_cnt);
    CUDA_CHECK(cudaGetLastError());
    cudaEventRecord(k1);

    int kcount;
    cudaEventRecord(d0);
    CUDA_CHECK(cudaMemcpy(&kcount,d_cnt,4,cudaMemcpyDeviceToHost));
    std::vector<KP> gpuKP(kcount>MAX_KP?MAX_KP:kcount);
    if(kcount){
        CUDA_CHECK(cudaMemcpy(gpuKP.data(),d_kp,gpuKP.size()*sizeof(KP),
                              cudaMemcpyDeviceToHost));
    }
    cudaEventRecord(d1);

    cudaEventRecord(tEnd); CUDA_CHECK(cudaEventSynchronize(tEnd));

    // Timings
    float h2d,ker,d2h,gpuTot;
    cudaEventElapsedTime(&h2d ,h0   ,h1  );
    cudaEventElapsedTime(&ker ,k0   ,k1  );
    cudaEventElapsedTime(&d2h ,d0   ,d1  );
    cudaEventElapsedTime(&gpuTot,tBeg,tEnd);

    std::cout<<"Image                   : "<<W<<" × "<<H<<"\n";
    std::cout<<"Threshold               : "<<thr<<"\n";
    std::cout<<"GPU keypoints           : "<<gpuKP.size()<<"\n";
    std::cout<<"CPU keypoints           : "<<cpuKP.size()<<"\n\n";
    std::cout<<"GPU H2D copy            : "<<h2d   <<" ms\n";
    std::cout<<"GPU kernel              : "<<ker   <<" ms\n";
    std::cout<<"GPU D2H copy            : "<<d2h   <<" ms\n";
    std::cout<<"GPU total               : "<<gpuTot<<" ms\n";
    std::cout<<"CPU total               : "<<cpu_ms<<" ms\n";

    // Cleanup
    CUDA_CHECK(cudaFree(d_img));
    CUDA_CHECK(cudaFree(d_kp));
    CUDA_CHECK(cudaFree(d_cnt));
    return 0;
}