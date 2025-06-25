#include <cuda_runtime.h>

#include <vector>
#include <cmath>
#include <iostream>
#include <chrono>
#include <random>
#include <cstdio>
#include <cstring>          // std::memcpy its definition**

#if __has_include(<opencv2/imgcodecs.hpp>)
# define USE_OPENCV 1
# include <opencv2/imgcodecs.hpp>
# include <opencv2/imgproc.hpp>
#else
# define USE_OPENCV 0
#endif

// PGM loader
#if !USE_OPENCV
static bool readPGM(const char* fn,std::vector<uint8_t>& img,int& W,int& H){
    FILE* fp=fopen(fn,"rb"); if(!fp) return false;
    char m[3]; if(fscanf(fp,"%2s",m)!=1||strcmp(m,"P5")){fclose(fp);return false;}
    int maxv; if(fscanf(fp,"%d%d%d",&W,&H,&maxv)!=3||maxv!=255){fclose(fp);return false;}
    fgetc(fp);
    img.resize(size_t(W)*H);
    size_t rd=fread(img.data(),1,img.size(),fp); (void)rd;
    fclose(fp); return true;
}
#endif

#define CUDA_CHECK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){                 \
    std::cerr<<"CUDA error: "<<cudaGetErrorString(e)<<" ("<<__LINE__<<")\n";     \
    std::exit(1);} }while(0)
template<class T> double ms(const T&a,const T&b){
    return std::chrono::duration<double,std::milli>(b-a).count(); }

// Clamp usable on host+device
__host__ __device__ inline int iclamp(int v,int lo,int hi){
    return v<lo?lo:(v>hi?hi:v);
}

// 7-tap Gaussian coeffs in constant memory
__constant__ float c_g7[7];

// SIFT key-point structure
struct KP{ short x,y,oct,layer; unsigned char ori; };

// Kernel forward declarations
template<int R> __global__ void gaussH(const float*,float*,int,int);
template<int R> __global__ void gaussV(const float*,float*,int,int);
__global__ void u8ToF32 (const uint8_t*,float*,int);
__global__ void dogKer  (const float*,const float*,float*,int);
__global__ void sobelKer(const uint8_t*,float*,float*,int,int);
__global__ void extremaKer(const float*,const float*,const float*,
                           KP*,int*,int,int,float);
__global__ void oriKer  (KP*,int,const float*,const float*,int,int);
__global__ void descKer (const KP*,int,const float*,const float*,float*,int,int);

// Kernels
template<int R>
__global__ void gaussH(const float* in,float* out,int W,int H){
    int x=blockIdx.x*blockDim.x+threadIdx.x;
    int y=blockIdx.y*blockDim.y+threadIdx.y;
    if(x>=W||y>=H) return;
    float s=0;
#pragma unroll
    for(int k=-R;k<=R;++k){
        int xx=iclamp(x+k,0,W-1);
        s+=in[y*W+xx]*c_g7[k+R];
    }
    out[y*W+x]=s;
}
template<int R>
__global__ void gaussV(const float* in,float* out,int W,int H){
    int x=blockIdx.x*blockDim.x+threadIdx.x;
    int y=blockIdx.y*blockDim.y+threadIdx.y;
    if(x>=W||y>=H) return;
    float s=0;
#pragma unroll
    for(int k=-R;k<=R;++k){
        int yy=iclamp(y+k,0,H-1);
        s+=in[yy*W+x]*c_g7[k+R];
    }
    out[y*W+x]=s;
}
__global__ void u8ToF32(const uint8_t* src,float* dst,int N){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<N) dst[i]=float(src[i]);
}
__global__ void dogKer(const float* a,const float* b,float* d,int N){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<N) d[i]=b[i]-a[i];
}
__global__ void sobelKer(const uint8_t* g,float* mag,float* ang,int W,int H){
    int x=blockIdx.x*blockDim.x+threadIdx.x;
    int y=blockIdx.y*blockDim.y+threadIdx.y;
    if(x<=0||x>=W-1||y<=0||y>=H-1) return;
    int gx=-g[(y-1)*W+x-1]-2*g[y*W+x-1]-g[(y+1)*W+x-1]
            +g[(y-1)*W+x+1]+2*g[y*W+x+1]+g[(y+1)*W+x+1];
    int gy= g[(y-1)*W+x-1]+2*g[(y-1)*W+x]+g[(y-1)*W+x+1]
           -g[(y+1)*W+x-1]-2*g[(y+1)*W+x]-g[(y+1)*W+x+1];
    float fx=float(gx), fy=float(gy);
    float m=sqrtf(fx*fx+fy*fy); float a=atan2f(fy,fx); if(a<0)a+=3.14159265f;
    int idx=y*W+x; mag[idx]=m; ang[idx]=a;
}
__global__ void extremaKer(const float* d0,const float* d1,const float* d2,
                           KP* out,int* cnt,int W,int H,float thr){
    int x=blockIdx.x*blockDim.x+threadIdx.x;
    int y=blockIdx.y*blockDim.y+threadIdx.y;
    if(x<1||x>=W-1||y<1||y>=H-1) return;
    int idx=y*W+x; float v=d1[idx]; if(fabsf(v)<thr) return;
    bool mx=v>0;
    for(int dy=-1;dy<=1;++dy)
        for(int dx=-1;dx<=1;++dx){
            if(d1[idx+dy*W+dx]*(mx?1:-1) > v*(mx?1:-1)) return;
            if(d0[idx+dy*W+dx]*(mx?1:-1) > v*(mx?1:-1)) return;
            if(d2[idx+dy*W+dx]*(mx?1:-1) > v*(mx?1:-1)) return;
        }
    int id=atomicAdd(cnt,1);
    if(id<50000) out[id]={short(x),short(y),0,1,0};
}
__global__ void oriKer(KP* kp,int n,const float* mag,const float* ori,int W,int H){
    int id=blockIdx.x*blockDim.x+threadIdx.x; if(id>=n) return;
    KP& k=kp[id]; float hist[36]={0};
    for(int dy=-8;dy<8;++dy){
        int yy=iclamp(k.y+dy,0,H-1);
        for(int dx=-8;dx<8;++dx){
            int xx=iclamp(k.x+dx,0,W-1);
            float m=mag[yy*W+xx];
            float a=ori[yy*W+xx];
            int b=int(a*57.29578f/10.f); if(b>=36) b=35;
            hist[b]+=m;
        }
    }
    float best=0; int bi=0;
    for(int i=0;i<36;++i) if(hist[i]>best){best=hist[i];bi=i;}
    k.ori=uint8_t(bi);
}
__global__ void descKer(const KP* kp,int n,const float* mag,const float* ori,
                        float* desc,int W,int H){
    int id=blockIdx.x*blockDim.x+threadIdx.x; if(id>=n) return;
    const KP& k=kp[id];
    float hist[128]={0};
    for(int dy=-8;dy<8;++dy)
        for(int dx=-8;dx<8;++dx){
            int yy=iclamp(k.y+dy,0,H-1);
            int xx=iclamp(k.x+dx,0,W-1);
            float m=mag[yy*W+xx];
            float a=ori[yy*W+xx]-k.ori*10.f*0.017453292f;
            if(a<0)a+=6.2831853f;
            int bin=int(a*57.29578f/45.f); if(bin>7)bin=7;
            int cx=(dx+8)>>2, cy=(dy+8)>>2;
            hist[(cy*4+cx)*8+bin]+=m;
        }
    float nrm=1e-6f;
    for(int i=0;i<128;++i) nrm+=hist[i]*hist[i];
    nrm=sqrtf(nrm);
    for(int i=0;i<128;++i) desc[id*128+i]=hist[i]/nrm;
}

// Host gaussian helper
static void makeKernel(std::vector<float>& k,int R,float s){
    k.resize(R*2+1); float s2=2*s*s,sum=0;
    for(int i=-R;i<=R;++i){ float v=expf(-i*i/s2); k[i+R]=v; sum+=v;}
    for(float& v:k) v/=sum;
}

int main(int argc,char** argv)
{
    float thr=(argc>=3)?std::stof(argv[2]):0.03f;

    // Load / generate image
    int W,H; std::vector<uint8_t> img8;
    if(argc>=2){
#if USE_OPENCV
        cv::Mat m=cv::imread(argv[1],cv::IMREAD_GRAYSCALE);
        if(m.empty()){std::cerr<<"load err\n"; return 1;}
        W=m.cols; H=m.rows; img8.assign(m.datastart,m.dataend);
#else
        if(!readPGM(argv[1],img8,W,H)){std::cerr<<"PGM err\n"; return 1;}
#endif
    }else{
        W=3840; H=2160; img8.resize(size_t(W)*H);
        std::mt19937 g(42); std::uniform_int_distribution<int>d(0,255);
        for(auto& p:img8) p=d(g);
        std::cout<<"[Info] random 4K "<<W<<"×"<<H<<"\n";
    }
    const int N=W*H;

    // Upload Gaussian coeff
    std::vector<float> hG; makeKernel(hG,3,1.6f);
    CUDA_CHECK(cudaMemcpyToSymbol(c_g7,hG.data(),hG.size()*4));

    // Device buffers
    uint8_t* d_u8; CUDA_CHECK(cudaMalloc(&d_u8,N));
    float* d_f;    CUDA_CHECK(cudaMalloc(&d_f ,N*4));
    float *d_tmp,*d_b1,*d_b2; CUDA_CHECK(cudaMalloc(&d_tmp,N*4));
    CUDA_CHECK(cudaMalloc(&d_b1,N*4)); CUDA_CHECK(cudaMalloc(&d_b2,N*4));
    float *d_d0,*d_d1,*d_d2; CUDA_CHECK(cudaMalloc(&d_d0,N*4));
    CUDA_CHECK(cudaMalloc(&d_d1,N*4)); CUDA_CHECK(cudaMalloc(&d_d2,N*4));
    float *d_mag,*d_ori; CUDA_CHECK(cudaMalloc(&d_mag,N*4));
    CUDA_CHECK(cudaMalloc(&d_ori,N*4));
    KP* d_kp;   CUDA_CHECK(cudaMalloc(&d_kp,50000*sizeof(KP)));
    int* d_cnt; CUDA_CHECK(cudaMalloc(&d_cnt,4));
    float* d_desc; CUDA_CHECK(cudaMalloc(&d_desc,50000*128*4));

    // Pinned copy
    uint8_t* hPin; CUDA_CHECK(cudaHostAlloc(&hPin,N,0));
    std::memcpy(hPin,img8.data(),N);

    cudaStream_t st; CUDA_CHECK(cudaStreamCreate(&st));
    cudaEvent_t eH,eK,eD; CUDA_CHECK(cudaEventCreate(&eH));
    CUDA_CHECK(cudaEventCreate(&eK)); CUDA_CHECK(cudaEventCreate(&eD));

    // GPU
    cudaEventRecord(eH,st);
    CUDA_CHECK(cudaMemcpyAsync(d_u8,hPin,N,cudaMemcpyHostToDevice,st));

    // uint8 → float
    u8ToF32<<<(N+255)/256,256,0,st>>>(d_u8,d_f,N);

    dim3 blk(32,8), grd((W+31)/32,(H+7)/8);

    gaussH<3><<<grd,blk,0,st>>>(d_f ,d_tmp,W,H);   // σ1
    gaussV<3><<<grd,blk,0,st>>>(d_tmp,d_b1,W,H);
    gaussH<3><<<grd,blk,0,st>>>(d_b1,d_tmp,W,H);   // σ2
    gaussV<3><<<grd,blk,0,st>>>(d_tmp,d_b2,W,H);   // σ2
    dogKer<<<(N+255)/256,256,0,st>>>(d_f ,d_b1,d_d0,N);
    dogKer<<<(N+255)/256,256,0,st>>>(d_b1,d_b2,d_d1,N);

    gaussH<3><<<grd,blk,0,st>>>(d_b2,d_tmp,W,H);   // σ3
    gaussV<3><<<grd,blk,0,st>>>(d_tmp,d_b1,W,H);
    dogKer<<<(N+255)/256,256,0,st>>>(d_b2,d_b1,d_d2,N);

    // Gradient field for orientation / descriptor
    dim3 blkS(16,16), grdS((W+15)/16,(H+15)/16);
    sobelKer<<<grdS,blkS,0,st>>>(d_u8,d_mag,d_ori,W,H);

    // Extrema detection
    CUDA_CHECK(cudaMemsetAsync(d_cnt,0,4,st));
    dim3 blkE(16,16), grdE((W+15)/16,(H+15)/16);
    extremaKer<<<grdE,blkE,0,st>>>(d_d0,d_d1,d_d2,d_kp,d_cnt,W,H,thr);

    int h_cnt; CUDA_CHECK(cudaMemcpy(&h_cnt,d_cnt,4,cudaMemcpyDeviceToHost));
    if(h_cnt>50000) h_cnt=50000;

    if(h_cnt){
        oriKer <<< (h_cnt+255)/256 ,256 ,0,st>>>(d_kp,h_cnt,d_mag,d_ori,W,H);
        descKer<<< (h_cnt+255)/256 ,256 ,0,st>>>(d_kp,h_cnt,d_mag,d_ori,d_desc,W,H);
    }

    cudaEventRecord(eK,st);
    cudaEventRecord(eD,st); CUDA_CHECK(cudaEventSynchronize(eD));

    float h2d,tot; cudaEventElapsedTime(&h2d,eH,eK);
    cudaEventElapsedTime(&tot,eH,eD);
    float ker=tot-h2d;

    // CPU ref (one scale)
    auto c0=std::chrono::high_resolution_clock::now();
    std::vector<float> f(N); for(int i=0;i<N;++i) f[i]=float(img8[i]);
    std::vector<float> k; makeKernel(k,3,1.6f);
    auto blur=[&](const std::vector<float>& in,std::vector<float>& out){
        int R=3; out.resize(N); std::vector<float> tmp(N);
        for(int y=0;y<H;++y) for(int x=0;x<W;++x){
            float s=0; for(int r=-R;r<=R;++r)
                s+=in[y*W+iclamp(x+r,0,W-1)]*k[r+R];
            tmp[y*W+x]=s;
        }
        for(int y=0;y<H;++y) for(int x=0;x<W;++x){
            float s=0; for(int r=-R;r<=R;++r)
                s+=tmp[iclamp(y+r,0,H-1)*W+x]*k[r+R];
            out[y*W+x]=s;
        }
    };
    std::vector<float> b1,b2,dog(N);
    blur(f,b1); blur(b1,b2);
    for(int i=0;i<N;++i) dog[i]=b2[i]-b1[i];
    size_t cpu_kp=0;
    for(int y=1;y<H-1;++y) for(int x=1;x<W-1;++x){
        float v=dog[y*W+x]; if(fabsf(v)<thr) continue;
        bool mx=v>0,ok=true;
        for(int dy=-1;dy<=1&&ok;++dy)
            for(int dx=-1;dx<=1;++dx)
                if(dog[(y+dy)*W+x+dx]*(mx?1:-1) > v*(mx?1:-1)) ok=false;
        if(ok) ++cpu_kp;
    }
    auto c1=std::chrono::high_resolution_clock::now();
    double cpu_ms=ms(c0,c1);

    // Report
    std::cout<<"Image "<<W<<"×"<<H<<"\n";
    std::cout<<"GPU keypoints          : "<<h_cnt<<"\n";
    std::cout<<"CPU keypoints          : "<<cpu_kp<<"\n";
    std::cout<<"GPU H2D copy           : "<<h2d<<" ms\n";
    std::cout<<"GPU kernels            : "<<ker <<" ms\n";
    std::cout<<"GPU total              : "<<tot <<" ms\n";
    std::cout<<"CPU total              : "<<cpu_ms<<" ms\n";
    std::cout<<"Speed-up (CPU / GPU)   : "<<cpu_ms/tot<<"×\n";

    // Cleanup
    cudaFree(d_u8); cudaFree(d_f); cudaFree(d_tmp); cudaFree(d_b1); cudaFree(d_b2);
    cudaFree(d_d0); cudaFree(d_d1); cudaFree(d_d2);
    cudaFree(d_mag); cudaFree(d_ori); cudaFree(d_kp); cudaFree(d_cnt);
    cudaFree(d_desc); cudaFreeHost(hPin);
    return 0;
}
