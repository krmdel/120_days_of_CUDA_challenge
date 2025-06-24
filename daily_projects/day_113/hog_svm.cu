#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <iostream>
#include <chrono>
#include <cstring>
#include <algorithm>
#include <random>

// Optional OpenCV
#if __has_include(<opencv2/imgcodecs.hpp>)
#  define USE_OPENCV 1
#  include <opencv2/imgcodecs.hpp>
#  include <opencv2/imgproc.hpp>
#  include <opencv2/core.hpp>
#else
#  define USE_OPENCV 0
#endif

// Utilities
#define CUDA_CHECK(x)  do{ cudaError_t e=(x); if(e!=cudaSuccess){                \
    std::cerr<<"CUDA error: "<<cudaGetErrorString(e)<<" ("<<__LINE__<<")\n";     \
    std::exit(1);} }while(0)
template<class TP> inline double ms(TP a,TP b){
    return std::chrono::duration<double,std::milli>(b-a).count(); }

// Fallback PGM/PPM I/O (when no OpenCV)
#if !USE_OPENCV
static bool readPGM(const char* fn,std::vector<uint8_t>& img,int& W,int& H){
    FILE* fp=fopen(fn,"rb"); if(!fp) return false;
    char m[3]; if(fscanf(fp,"%2s",m)!=1||strcmp(m,"P5")){fclose(fp);return false;}
    if(fscanf(fp,"%d%d",&W,&H)!=2){fclose(fp);return false;}
    int maxv; if(fscanf(fp,"%d",&maxv)!=1||maxv!=255){fclose(fp);return false;}
    fgetc(fp);
    img.resize(size_t(W)*H);
    if(fread(img.data(),1,img.size(),fp)!=img.size()){fclose(fp);return false;}
    fclose(fp); return true;
}
static bool writePPM(const char* fn,const std::vector<uint8_t>& rgb,int W,int H){
    FILE* fp=fopen(fn,"wb"); if(!fp) return false;
    fprintf(fp,"P6\n%d %d\n255\n",W,H);
    bool ok=fwrite(rgb.data(),1,rgb.size(),fp)==rgb.size();
    fclose(fp); return ok;
}
#endif

// SVM constants
__constant__ float c_w[3780];
__constant__ float c_b;

// Device helpers
__device__ __forceinline__ float warpSum(float v){
    for(int o=16;o;o>>=1) v+=__shfl_down_sync(0xffffffff,v,o);
    return v;
}

// Kernels
// Sobel gradient (16×16 tile)
__global__ void sobelKer(const uint8_t* g,float* mag,float* ang,int W,int H){
    extern __shared__ uint8_t sh[];
    const int B=16;
    int lx=threadIdx.x, ly=threadIdx.y;
    int gx=blockIdx.x*B+lx, gy=blockIdx.y*B+ly;
    int shW=B+2, sx=lx+1, sy=ly+1;
    if(gx<W&&gy<H) sh[sy*shW+sx]=g[gy*W+gx];
    if(lx==0&&gx)           sh[sy*shW]      =g[gy*W+gx-1];
    if(lx==B-1&&gx+1<W)     sh[sy*shW+sx+1] =g[gy*W+gx+1];
    if(ly==0&&gy)           sh[sx]          =g[(gy-1)*W+gx];
    if(ly==B-1&&gy+1<H)     sh[(sy+1)*shW+sx]=g[(gy+1)*W+gx];
    if(lx==0&&ly==0&&gx&&gy)                   sh[0]=g[(gy-1)*W+gx-1];
    if(lx==B-1&&ly==0&&gx+1<W&&gy)             sh[sx+1]=g[(gy-1)*W+gx+1];
    if(lx==0&&ly==B-1&&gx&&gy+1<H)             sh[(sy+1)*shW]=g[(gy+1)*W+gx-1];
    if(lx==B-1&&ly==B-1&&gx+1<W&&gy+1<H)       sh[(sy+1)*shW+sx+1]=g[(gy+1)*W+gx+1];
    __syncthreads();
    if(gx>0&&gx<W-1&&gy>0&&gy<H-1){
        int gxv=-sh[(sy-1)*shW+sx-1]+sh[(sy-1)*shW+sx+1]
                -2*sh[sy*shW+sx-1]  +2*sh[sy*shW+sx+1]
                -sh[(sy+1)*shW+sx-1]+sh[(sy+1)*shW+sx+1];
        int gyv= sh[(sy-1)*shW+sx-1]+2*sh[(sy-1)*shW+sx]+sh[(sy-1)*shW+sx+1]
                -sh[(sy+1)*shW+sx-1]-2*sh[(sy+1)*shW+sx]-sh[(sy+1)*shW+sx+1];
        float fx=float(gxv), fy=float(gyv);
        int idx=gy*W+gx;
        mag[idx]=sqrtf(fx*fx+fy*fy);
        float a=atan2f(fy,fx); if(a<0)a+=3.14159265f;
        ang[idx]=a;
    }
}
// Cell histogram (8×8, one warp)
__global__ void cellKer(const float* mag,const float* ang,float* hist,
                        int W,int H,int Cx){
    int cx=blockIdx.x, cy=blockIdx.y, lane=threadIdx.x;
    int gx0=cx*8, gy0=cy*8;
    float bins[9]={0};
    for(int k=0;k<2;++k){
        int p=lane+k*32; if(p<64){
            int lx=p&7, ly=p>>3;
            float m=mag[(gy0+ly)*W+gx0+lx];
            float a=ang[(gy0+ly)*W+gx0+lx];
            int b=int((a*57.2957795f)/20.f); if(b>8)b=8;
            bins[b]+=m;
        }
    }
    for(int b=0;b<9;++b){
        float v=warpSum(bins[b]);
        if(lane==0) hist[(cy*Cx+cx)*9+b]=v;
    }
}
// Block normalisation (4 cells)
__global__ void blockKer(const float* cell,float* blk,int Cx,int Bx,int By){
    int id=blockIdx.x*blockDim.x+threadIdx.x; if(id>=Bx*By) return;
    int by=id/Bx, bx=id-by*Bx;
    float v[36];
#pragma unroll
    for(int cy=0;cy<2;++cy)
        for(int cx=0;cx<2;++cx){
            const float* h=&cell[((by+cy)*Cx+(bx+cx))*9];
#pragma unroll
            for(int b=0;b<9;++b) v[(cy*2+cx)*9+b]=h[b];
        }
    const float eps=1e-6f,clip=0.2f;
    float n=eps;
#pragma unroll
    for(int i=0;i<36;++i) n+=v[i]*v[i];
    n=sqrtf(n);
#pragma unroll
    for(int i=0;i<36;++i){ v[i]/=n; if(v[i]>clip)v[i]=clip; }
    n=eps;
#pragma unroll
    for(int i=0;i<36;++i) n+=v[i]*v[i];
    n=sqrtf(n);
#pragma unroll
    for(int i=0;i<36;++i) blk[id*36+i]=v[i]/n;
}
// Gather 7×15 window into descriptor
__global__ void gatherKer(const float* blk,float* desc,
                          int Bx,int By,int gridW,int Ndesc){
    int id=blockIdx.x*blockDim.x+threadIdx.x; if(id>=Ndesc) return;
    int wy=id/gridW, wx=id-wy*gridW;
    float* dst=desc+id*3780; int ptr=0;
#pragma unroll
    for(int y=0;y<15;++y){
        int by=wy+y;
        for(int x=0;x<7;++x){
            const float* src=&blk[((by*Bx)+(wx+x))*36];
#pragma unroll
            for(int k=0;k<36;++k) dst[ptr++]=src[k];
        }
    }
}

// Dot product
__global__ void dotKer(const float* desc,float* score,int N){
    int id=blockIdx.x*blockDim.x+threadIdx.x; if(id>=N) return;
    const float* v=desc+id*3780;
    float s=0.f;
#pragma unroll
    for(int i=0;i<3780;++i) s+=v[i]*c_w[i];
    score[id]=s+c_b;
}

// Threshold
struct Det{int x,y;float s;};
__global__ void threshKer(const float* score,Det* det,int* cnt,
                          float thr,int gridW,int N){
    int id=blockIdx.x*blockDim.x+threadIdx.x; if(id>=N) return;
    float s=score[id]; if(s<=thr) return;
    int idx=atomicAdd(cnt,1);
    int wy=id/gridW, wx=id-wy*gridW;
    det[idx]={wx*8,wy*8,s};
}

// CPU NMS
struct CDet{int x,y;float s;};
static std::vector<CDet> nms(const std::vector<CDet>& in,float iou=0.4f){
    auto v=in; std::sort(v.begin(),v.end(),[](auto&a,auto&b){return a.s>b.s;});
    std::vector<CDet> keep;
    while(!v.empty()){
        CDet d=v.front(); v.erase(v.begin()); keep.push_back(d);
        auto it=v.begin();
        while(it!=v.end()){
            int xx=std::max(d.x,it->x), yy=std::max(d.y,it->y);
            int xx2=std::min(d.x+64,it->x+64), yy2=std::min(d.y+128,it->y+128);
            int w=std::max(0,xx2-xx), h=std::max(0,yy2-yy);
            float inter=float(w)*h, uni=64*128*2 - inter;
            if(inter/uni > iou) it=v.erase(it); else ++it;
        }
    }
    return keep;
}

// Upload SVM
static void uploadSVM(){
    std::vector<float> w(3780,0.01f); float b=-0.4f;
    CUDA_CHECK(cudaMemcpyToSymbol(c_w,w.data(),3780*4));
    CUDA_CHECK(cudaMemcpyToSymbol(c_b,&b,4));
}


int main(int argc,char** argv)
{
    float thr = (argc>=3)? std::stof(argv[2]) : 0.5f;

    // Load or generate image
    int W,H; std::vector<uint8_t> img;
    if(argc>=2){
#if USE_OPENCV
        cv::Mat im=cv::imread(argv[1],cv::IMREAD_GRAYSCALE);
        if(im.empty()){ std::cerr<<"load error\n"; return 1; }
        W=im.cols; H=im.rows; img.assign(im.datastart,im.dataend);
#else
        if(!readPGM(argv[1],img,W,H)){ std::cerr<<"PGM read error\n"; return 1;}
#endif
    }else{
        std::cout<<"[Info] no image provided – generating random 1280×720 frame\n";
        W=1280; H=1280; img.resize(size_t(W)*H);
        std::mt19937 rng(42); std::uniform_int_distribution<int> d(0,255);
        for(auto& p:img) p=d(rng);
    }
    if(W%8||H%8){ std::cerr<<"Image dims must be multiples of 8\n"; return 1; }
    size_t Npix=size_t(W)*H;

    // Grid sizes
    int Cx=W/8, Cy=H/8, Bx=Cx-1, By=Cy-1;
    int gridW=Bx-6, gridH=By-14, Ndesc=gridW*gridH;

    // Host pinned
    uint8_t* hIn; CUDA_CHECK(cudaHostAlloc(&hIn,Npix,cudaHostAllocDefault));
    std::memcpy(hIn,img.data(),Npix);

    // Device alloc
    uint8_t *d_img;  CUDA_CHECK(cudaMalloc(&d_img,Npix));
    float *d_mag,*d_ang; CUDA_CHECK(cudaMalloc(&d_mag,Npix*4));
    CUDA_CHECK(cudaMalloc(&d_ang,Npix*4));
    float *d_cell;   CUDA_CHECK(cudaMalloc(&d_cell,size_t(Cx)*Cy*9*4));
    float *d_blk;    CUDA_CHECK(cudaMalloc(&d_blk ,Bx*By*36*4));
    float *d_desc;   CUDA_CHECK(cudaMalloc(&d_desc,Ndesc*3780*4));
    float *d_score;  CUDA_CHECK(cudaMalloc(&d_score,Ndesc*4));
    Det  *d_det;     CUDA_CHECK(cudaMalloc(&d_det,Ndesc*sizeof(Det)));
    int  *d_cnt;     CUDA_CHECK(cudaMalloc(&d_cnt,4));

    uploadSVM();

    cudaStream_t st; CUDA_CHECK(cudaStreamCreate(&st));
    cudaEvent_t eH,eK,eD; CUDA_CHECK(cudaEventCreate(&eH));
    CUDA_CHECK(cudaEventCreate(&eK)); CUDA_CHECK(cudaEventCreate(&eD));

    // Pipeline
    cudaEventRecord(eH,st);
    CUDA_CHECK(cudaMemcpyAsync(d_img,hIn,Npix,cudaMemcpyHostToDevice,st));

    dim3 blkG(16,16), grdG((W+15)/16,(H+15)/16);
    sobelKer<<<grdG,blkG,(16+2)*(16+2),st>>>(d_img,d_mag,d_ang,W,H);

    dim3 grdC(Cx,Cy);
    cellKer<<<grdC,32,0,st>>>(d_mag,d_ang,d_cell,W,H,Cx);

    int threads=256, blocks=(Bx*By+threads-1)/threads;
    blockKer<<<blocks,threads,0,st>>>(d_cell,d_blk,Cx,Bx,By);

    int t2=128, b2=(Ndesc+t2-1)/t2;
    gatherKer<<<b2,t2,0,st>>>(d_blk,d_desc,Bx,By,gridW,Ndesc);

    dotKer<<<b2,t2,0,st>>>(d_desc,d_score,Ndesc);
    CUDA_CHECK(cudaMemsetAsync(d_cnt,0,4,st));
    threshKer<<<b2,t2,0,st>>>(d_score,d_det,d_cnt,thr,gridW,Ndesc);
    cudaEventRecord(eK,st);

    // Copy count & detections
    int h_cnt; CUDA_CHECK(cudaMemcpyAsync(&h_cnt,d_cnt,4,cudaMemcpyDeviceToHost,st));
    cudaEventRecord(eD,st);
    CUDA_CHECK(cudaEventSynchronize(eD));

    std::vector<Det> detRaw(h_cnt);
    if(h_cnt) CUDA_CHECK(cudaMemcpy(detRaw.data(),d_det,h_cnt*sizeof(Det),cudaMemcpyDeviceToHost));

    // Timings
    float tH,tTot; cudaEventElapsedTime(&tH,eH,eK);
    cudaEventElapsedTime(&tTot,eH,eD);
    float tK=tTot-tH;

    // NMS
    std::vector<CDet> v(h_cnt);
    for(int i=0;i<h_cnt;++i) v[i]={detRaw[i].x,detRaw[i].y,detRaw[i].s};
    auto n0=std::chrono::high_resolution_clock::now();
    auto keep=nms(v);
    auto n1=std::chrono::high_resolution_clock::now();
    double nms_ms=ms(n0,n1);

    std::cout<<"Image "<<W<<"×"<<H<<"\n";
    std::cout<<"GPU H2D copy      : "<<tH  <<" ms\n";
    std::cout<<"GPU kernels       : "<<tK  <<" ms\n";
    std::cout<<"GPU total         : "<<tTot<<" ms\n";
    std::cout<<"CPU NMS           : "<<nms_ms<<" ms\n";
    std::cout<<"Detections raw/NMS: "<<h_cnt<<" / "<<keep.size()<<"\n";

    // Draw output
#if USE_OPENCV
    cv::Mat g(H,W,CV_8UC1,img.data());
    cv::Mat out; cv::cvtColor(g,out,cv::COLOR_GRAY2BGR);
    for(auto& d:keep) cv::rectangle(out,{d.x,d.y},{d.x+64,d.y+128},{0,255,0},2);
    cv::imwrite("hog_out.png",out);
    std::cout<<"Saved hog_out.png\n";
#else
    std::vector<uint8_t> rgb(W*H*3);
    for(size_t i=0;i<img.size();++i)
        rgb[i*3+0]=rgb[i*3+1]=rgb[i*3+2]=img[i];
    auto drawRect=[&](int x,int y){
        for(int dx=0;dx<64;++dx){
            if(x+dx>=W) break;
            if(y>=0&&y<H){ int p=(y*W+x+dx)*3; rgb[p]=0;rgb[p+1]=255;rgb[p+2]=0; }
            if(y+127<H){  int p=((y+127)*W+x+dx)*3; rgb[p]=0;rgb[p+1]=255;rgb[p+2]=0; }
        }
        for(int dy=0;dy<128;++dy){
            if(y+dy>=H) break;
            if(x>=0&&x<W){ int p=((y+dy)*W+x)*3; rgb[p]=0;rgb[p+1]=255;rgb[p+2]=0; }
            if(x+63<W){   int p=((y+dy)*W+x+63)*3; rgb[p]=0;rgb[p+1]=255;rgb[p+2]=0; }
        }
    };
    for(auto& d:keep) drawRect(d.x,d.y);
    writePPM("hog_out.ppm",rgb,W,H);
    std::cout<<"Saved hog_out.ppm\n";
#endif

    // Cleanup
    cudaFree(d_img); cudaFree(d_mag); cudaFree(d_ang);
    cudaFree(d_cell); cudaFree(d_blk); cudaFree(d_desc);
    cudaFree(d_score); cudaFree(d_det); cudaFree(d_cnt);
    cudaFreeHost(hIn);
    return 0;
}
