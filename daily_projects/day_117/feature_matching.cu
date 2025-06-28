#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <vector>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <chrono>
#include <random>
#include <cstring>
#include <cmath>
#include <algorithm>

#if __has_include(<opencv2/imgcodecs.hpp>)
# define USE_OPENCV 1
# include <opencv2/imgcodecs.hpp>
# include <opencv2/imgproc.hpp>
#else
# define USE_OPENCV 0
#endif

// PGM when OpenCV absent
#if !USE_OPENCV
static bool readPGM(const char* fn,std::vector<uint8_t>& img,int& W,int& H){
    FILE* fp=fopen(fn,"rb"); if(!fp) return false;
    char m[3]; if(fscanf(fp,"%2s",m)!=1||strcmp(m,"P5")){fclose(fp);return false;}
    int maxv; if(fscanf(fp,"%d%d%d",&W,&H,&maxv)!=3||maxv!=255){fclose(fp);return false;}
    fgetc(fp);
    img.resize(size_t(W)*H);
    size_t rd=fread(img.data(),1,img.size(),fp);
    fclose(fp); return rd==img.size();
}
#endif

// Helpers
#define CUDA_CHECK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){                \
    std::cerr<<"CUDA error: "<<cudaGetErrorString(e)<<" ("<<__LINE__<<")\n";    \
    std::exit(1);} }while(0)
template<class T> double ms(T a,T b){
    return std::chrono::duration<double,std::milli>(b-a).count(); }

constexpr int MAX_KP  = 8192;
constexpr int PATCH_R = 16;

// FAST offsets & BRIEF pattern in constant memory
__device__ __constant__ int8_t c_cx[16]={0,3,6,8,8,6,3,0,-3,-6,-8,-8,-6,-3,0,0};
__device__ __constant__ int8_t c_cy[16]={8,6,3,0,-3,-6,-8,-8,-8,-6,-3,0,3,6,8,8};
__device__ __constant__ int8_t c_brief[1024];

struct KP{ short x,y; };
__host__ __device__ inline int iclamp(int v,int l,int h){
    return v<l?l:(v>h?h:v);
}

// 1. FAST-9 kernel
__global__ void fastKer(const uint8_t* img,int W,int H,int thr,
                        KP* out,int* d_cnt)
{
    int x=blockIdx.x*blockDim.x+threadIdx.x;
    int y=blockIdx.y*blockDim.y+threadIdx.y;
    if(x<16||y<16||x>=W-16||y>=H-16) return;
    int idx=y*W+x, p=img[idx];
    int b=0,d=0;
    if(img[idx+ 9]   >p+thr) ++b; else if(img[idx+ 9]   <p-thr) ++d;
    if(img[idx- 9]   >p+thr) ++b; else if(img[idx- 9]   <p-thr) ++d;
    if(img[idx+ 9*W] >p+thr) ++b; else if(img[idx+ 9*W] <p-thr) ++d;
    if(img[idx- 9*W] >p+thr) ++b; else if(img[idx- 9*W] <p-thr) ++d;
    if(b<3 && d<3) return;

    int consec=0,maxc=0;
#pragma unroll
    for(int i=0;i<24;++i){
        int ii=i&15;
        int q=img[(y+c_cy[ii])*W+(x+c_cx[ii])];
        if(q>p+thr||q<p-thr) ++consec; else consec=0;
        maxc=max(maxc,consec);
        if(maxc>=9) break;
    }
    if(maxc<9) return;
    int id=atomicAdd(d_cnt,1);
    if(id<MAX_KP) out[id]={short(x),short(y)};
}

// 2. BRIEF-256 kernel
__global__ void briefKer(const KP* kp,int n,const uint8_t* img,
                         uint32_t* desc,int W,int H)
{
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return;
    KP k=kp[i];

    // Orientation by intensity centroid
    int m01=0,m10=0;
    for(int dy=-PATCH_R;dy<=PATCH_R;++dy){
        int yy=iclamp(k.y+dy,0,H-1);
        for(int dx=-PATCH_R;dx<=PATCH_R;++dx){
            int xx=iclamp(k.x+dx,0,W-1);
            int v=img[yy*W+xx];
            m01+=v*dy; m10+=v*dx;
        }
    }
    float ang=atan2f(float(m01),float(m10));
    float ca=cosf(ang), sa=sinf(ang);

    uint32_t bits[8]={0};
#pragma unroll
    for(int b=0;b<256;++b){
        const int8_t* p=&c_brief[4*b];
        int x1=int(ca*p[0]-sa*p[1]+0.5f);
        int y1=int(sa*p[0]+ca*p[1]+0.5f);
        int x2=int(ca*p[2]-sa*p[3]+0.5f);
        int y2=int(sa*p[2]+ca*p[3]+0.5f);
        int v1=img[iclamp(k.y+y1,0,H-1)*W + iclamp(k.x+x1,0,W-1)];
        int v2=img[iclamp(k.y+y2,0,H-1)*W + iclamp(k.x+x2,0,W-1)];
        if(v1<v2) bits[b>>5] |= 1u<<(b&31);
    }
#pragma unroll
    for(int w=0;w<8;++w) desc[i*8+w]=bits[w];
}

// 3. Lowe-ratio matcher
__global__ void matchKer(const uint32_t* d1,const uint32_t* d2,
                         int n1,int n2,uint16_t* best,float ratio)
{
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n1) return;
    const uint32_t* A=&d1[i*8];
    int bestD=512,sec=512, bestJ=-1;
    for(int j=0;j<n2;++j){
        const uint32_t* B=&d2[j*8];
        int d=0;
#pragma unroll
        for(int w=0;w<8;++w) d+=__popc(A[w]^B[w]);
        if(d<bestD){ sec=bestD; bestD=d; bestJ=j; }
        else if(d<sec) sec=d;
    }
    best[i]=(bestD<ratio*sec)?uint16_t(bestJ):0xFFFF;
}

// 4.  GPU RANSAC (fixed shared-mem reduction)
__device__ float2 operator-(float2 a,float2 b){ return {a.x-b.x,a.y-b.y}; }
__device__ float  dot2(float2 a,float2 b){ return a.x*b.x+a.y*b.y; }

__global__ void ransacKer(const KP* kp1,const KP* kp2,
                          const uint16_t* match,int n,
                          int* bestInl,float tol2,int seed)
{
    extern __shared__ int gBestArr[];
    int &gBest = gBestArr[0];
    if(threadIdx.x==0) gBest=0;
    __syncthreads();

    curandState rng; curand_init(seed+blockIdx.x,threadIdx.x,0,&rng);

    int localBest = 0;

    for(int iter=0;iter<256;++iter){
        // 2 random correspondences ⇒ similarity (scale+shift)
        int i0,i1;
        do i0=curand(&rng)%n; while(match[i0]==0xFFFF);
        do i1=curand(&rng)%n; while(match[i1]==0xFFFF||i1==i0);

        float2 p0={float(kp1[i0].x),float(kp1[i0].y)};
        float2 p1={float(kp1[i1].x),float(kp1[i1].y)};
        float2 q0={float(kp2[match[i0]].x),float(kp2[match[i0]].y)};
        float2 q1={float(kp2[match[i1]].x),float(kp2[match[i1]].y)};

        float2 dp=p1-p0, dq=q1-q0;
        float s=sqrtf(dot2(dq,dq)/max(dot2(dp,dp),1e-6f));
        float2 t={q0.x - s*p0.x, q0.y - s*p0.y};

        int inl=0;
        for(int k=threadIdx.x;k<n;k+=blockDim.x){
            int j=match[k]; if(j==0xFFFF) continue;
            float2 pp={s*kp1[k].x + t.x, s*kp1[k].y + t.y};
            float2 qq={float(kp2[j].x),float(kp2[j].y)};
            float2 d=pp-qq;
            if(dot2(d,d)<tol2) ++inl;
        }
        localBest = max(localBest,inl);
    }
    // Block-wide best
    atomicMax(&gBest,localBest);
    __syncthreads();
    if(threadIdx.x==0) atomicMax(bestInl,gBest);
}

// CPU helpers for reference timing
static inline uint32_t pop32(uint32_t x){ return __builtin_popcount(x); }

static void cpuFAST(const std::vector<uint8_t>& img,int W,int H,int thr,
                    std::vector<KP>& kp);

static void cpuBRIEF(const std::vector<uint8_t>& img,int W,int H,
                     const std::vector<KP>& kp,std::vector<uint32_t>& desc,
                     const int8_t* pat);

static void cpuMatch(const std::vector<uint32_t>& d1,const std::vector<uint32_t>& d2,
                     float ratio,std::vector<uint16_t>& best);

// Pattern initialiser
static void initPattern(){
    std::mt19937 rng(123);
    std::uniform_int_distribution<int> d(-PATCH_R,PATCH_R);
    int8_t host[1024]; for(auto& v:host) v=int8_t(d(rng));
    CUDA_CHECK(cudaMemcpyToSymbol(c_brief,host,1024));
}

int main(int argc,char** argv)
{
    int thr   = (argc>=4)?std::stoi(argv[3]):25;
    float ratio = 0.75f;
    int W=0,H=0;
    std::vector<uint8_t> img1,img2;

    if(argc>=3){
#if USE_OPENCV
        cv::Mat m1=cv::imread(argv[1],cv::IMREAD_GRAYSCALE);
        cv::Mat m2=cv::imread(argv[2],cv::IMREAD_GRAYSCALE);
        if(m1.empty()||m2.empty()){ std::cerr<<"load error\n"; return 1; }
        if(m1.size()!=m2.size()){ std::cerr<<"size mismatch\n"; return 1; }
        W=m1.cols; H=m1.rows;
        img1.assign(m1.datastart,m1.dataend);
        img2.assign(m2.datastart,m2.dataend);
#else
        if(!readPGM(argv[1],img1,W,H) || !readPGM(argv[2],img2,W,H)){
            std::cerr<<"PGM load error\n"; return 1; }
#endif
    }else{
        W=1920; H=1080;
        img1.resize(size_t(W)*H); img2.resize(size_t(W)*H);
        std::mt19937 rng(0); std::uniform_int_distribution<int>d(0,255);
        for(auto& v:img1) v=d(rng);
        img2=img1;
        std::cout<<"[Info] generated two identical "<<W<<"×"<<H<<" frames\n";
    }
    const int N=W*H;
    initPattern();

    // CPU reference
    auto tc0=std::chrono::high_resolution_clock::now();
    std::vector<KP> cpuKP1,cpuKP2;
    cpuFAST(img1,W,H,thr,cpuKP1); cpuFAST(img2,W,H,thr,cpuKP2);
    if(cpuKP1.size()>MAX_KP) cpuKP1.resize(MAX_KP);
    if(cpuKP2.size()>MAX_KP) cpuKP2.resize(MAX_KP);

    int8_t hostPat[1024]; CUDA_CHECK(cudaMemcpyFromSymbol(hostPat,c_brief,1024));
    std::vector<uint32_t> cpuD1,cpuD2;
    cpuBRIEF(img1,W,H,cpuKP1,cpuD1,hostPat);
    cpuBRIEF(img2,W,H,cpuKP2,cpuD2,hostPat);

    std::vector<uint16_t> cpuMatchIdx; cpuMatch(cpuD1,cpuD2,ratio,cpuMatchIdx);
    auto tc1=std::chrono::high_resolution_clock::now();
    double cpu_ms=ms(tc0,tc1);

    // GPU pipeline with timing
    uint8_t *d_im1,*d_im2; CUDA_CHECK(cudaMalloc(&d_im1,N)); CUDA_CHECK(cudaMalloc(&d_im2,N));
    KP *d_kp1,*d_kp2; CUDA_CHECK(cudaMalloc(&d_kp1,MAX_KP*sizeof(KP))); CUDA_CHECK(cudaMalloc(&d_kp2,MAX_KP*sizeof(KP)));
    int *d_cnt1,*d_cnt2; CUDA_CHECK(cudaMalloc(&d_cnt1,4)); CUDA_CHECK(cudaMalloc(&d_cnt2,4));
    uint32_t *d_desc1,*d_desc2; CUDA_CHECK(cudaMalloc(&d_desc1,MAX_KP*8*4)); CUDA_CHECK(cudaMalloc(&d_desc2,MAX_KP*8*4));
    uint16_t *d_match; CUDA_CHECK(cudaMalloc(&d_match,MAX_KP*2));
    int* d_bestInl; CUDA_CHECK(cudaMalloc(&d_bestInl,4));

    cudaEvent_t ev[12]; for(int i=0;i<12;++i) CUDA_CHECK(cudaEventCreate(&ev[i]));

    cudaEventRecord(ev[0]);
    CUDA_CHECK(cudaMemcpyAsync(d_im1,img1.data(),N,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpyAsync(d_im2,img2.data(),N,cudaMemcpyHostToDevice));
    cudaEventRecord(ev[1]);

    CUDA_CHECK(cudaMemset(d_cnt1,0,4)); CUDA_CHECK(cudaMemset(d_cnt2,0,4));
    dim3 blk(16,16), grd((W+15)/16,(H+15)/16);

    cudaEventRecord(ev[2]);
    fastKer<<<grd,blk>>>(d_im1,W,H,thr,d_kp1,d_cnt1);
    fastKer<<<grd,blk>>>(d_im2,W,H,thr,d_kp2,d_cnt2);
    cudaEventRecord(ev[3]);

    int n1,n2;
    CUDA_CHECK(cudaMemcpy(&n1,d_cnt1,4,cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&n2,d_cnt2,4,cudaMemcpyDeviceToHost));
    n1=std::min(n1,MAX_KP); n2=std::min(n2,MAX_KP);

    cudaEventRecord(ev[4]);
    if(n1) briefKer<<<(n1+255)/256,256>>>(d_kp1,n1,d_im1,d_desc1,W,H);
    if(n2) briefKer<<<(n2+255)/256,256>>>(d_kp2,n2,d_im2,d_desc2,W,H);
    cudaEventRecord(ev[5]);

    cudaEventRecord(ev[6]);
    if(n1&&n2) matchKer<<<(n1+255)/256,256>>>(d_desc1,d_desc2,n1,n2,d_match,ratio);
    cudaEventRecord(ev[7]);

    cudaEventRecord(ev[8]);
    if(n1) ransacKer<<<1,128,4>>>(d_kp1,d_kp2,d_match,n1,d_bestInl,25.0f,1234);
    cudaEventRecord(ev[9]);

    int inlGPU=0;
    cudaEventRecord(ev[10]);
    CUDA_CHECK(cudaMemcpy(&inlGPU,d_bestInl,4,cudaMemcpyDeviceToHost));
    cudaEventRecord(ev[11]); CUDA_CHECK(cudaEventSynchronize(ev[11]));

    float tH2D,tFast,tBrief,tMatch,tRans,tD2H;
    cudaEventElapsedTime(&tH2D ,ev[0],ev[1]);
    cudaEventElapsedTime(&tFast,ev[2],ev[3]);
    cudaEventElapsedTime(&tBrief,ev[4],ev[5]);
    cudaEventElapsedTime(&tMatch,ev[6],ev[7]);
    cudaEventElapsedTime(&tRans ,ev[8],ev[9]);
    cudaEventElapsedTime(&tD2H ,ev[10],ev[11]);

    std::cout<<"Resolution              : "<<W<<" × "<<H<<"\n";
    std::cout<<"kp1 / kp2               : "<<n1<<" / "<<n2<<"\n";
    std::cout<<"GPU best inliers        : "<<inlGPU<<"\n\n";
    std::cout<<"GPU H2D copy            : "<<tH2D   <<" ms\n";
    std::cout<<"GPU FAST kernels        : "<<tFast  <<" ms\n";
    std::cout<<"GPU BRIEF kernels       : "<<tBrief <<" ms\n";
    std::cout<<"GPU match kernel        : "<<tMatch <<" ms\n";
    std::cout<<"GPU RANSAC kernel       : "<<tRans  <<" ms\n";
    std::cout<<"GPU D2H copy            : "<<tD2H   <<" ms\n";
    std::cout<<"GPU total               : "<<tH2D+tFast+tBrief+tMatch+tRans+tD2H<<" ms\n";
    std::cout<<"CPU total               : "<<cpu_ms <<" ms\n";

    CUDA_CHECK(cudaFree(d_im1)); CUDA_CHECK(cudaFree(d_im2));
    CUDA_CHECK(cudaFree(d_kp1)); CUDA_CHECK(cudaFree(d_kp2));
    CUDA_CHECK(cudaFree(d_cnt1));CUDA_CHECK(cudaFree(d_cnt2));
    CUDA_CHECK(cudaFree(d_desc1));CUDA_CHECK(cudaFree(d_desc2));
    CUDA_CHECK(cudaFree(d_match));CUDA_CHECK(cudaFree(d_bestInl));
    return 0;
}

static void cpuFAST(const std::vector<uint8_t>& img,int W,int H,int thr,
                    std::vector<KP>& kp)
{
    static const int cx[16]={0,3,6,8,8,6,3,0,-3,-6,-8,-8,-6,-3,0,0};
    static const int cy[16]={8,6,3,0,-3,-6,-8,-8,-8,-6,-3,0,3,6,8,8};
    kp.clear(); kp.reserve(MAX_KP);
    auto at=[&](int x,int y){ return img[y*W+x]; };
    for(int y=16;y<H-16&&kp.size()<MAX_KP;++y)
        for(int x=16;x<W-16&&kp.size()<MAX_KP;++x){
            int p=at(x,y), b=0,d=0;
            if(at(x,y+9)>p+thr) ++b; else if(at(x,y+9)<p-thr) ++d;
            if(at(x,y-9)>p+thr) ++b; else if(at(x,y-9)<p-thr) ++d;
            if(at(x+9,y)>p+thr) ++b; else if(at(x+9,y)<p-thr) ++d;
            if(at(x-9,y)>p+thr) ++b; else if(at(x-9,y)<p-thr) ++d;
            if(b<3&&d<3) continue;
            int consec=0,maxc=0;
            for(int i=0;i<24;++i){
                int q=at(x+cx[i&15],y+cy[i&15]);
                if(q>p+thr||q<p-thr) ++consec; else consec=0;
                maxc=std::max(maxc,consec);
                if(maxc>=9) break;
            }
            if(maxc>=9) kp.push_back({short(x),short(y)});
        }
}

static void cpuBRIEF(const std::vector<uint8_t>& img,int W,int H,
                     const std::vector<KP>& kp,std::vector<uint32_t>& desc,
                     const int8_t* pat)
{
    desc.resize(kp.size()*8);
    for(size_t i=0;i<kp.size();++i){
        const KP& k=kp[i];
        int m01=0,m10=0;
        for(int dy=-PATCH_R;dy<=PATCH_R;++dy){
            int yy=iclamp(k.y+dy,0,H-1);
            for(int dx=-PATCH_R;dx<=PATCH_R;++dx){
                int xx=iclamp(k.x+dx,0,W-1);
                int v=img[yy*W+xx];
                m01+=v*dy; m10+=v*dx;
            }
        }
        float ang=atan2f(float(m01),float(m10));
        float ca=cosf(ang), sa=sinf(ang);
        uint32_t bits[8]={0};
        for(int b=0;b<256;++b){
            const int8_t* p=&pat[4*b];
            int x1=int(ca*p[0]-sa*p[1]+0.5f);
            int y1=int(sa*p[0]+ca*p[1]+0.5f);
            int x2=int(ca*p[2]-sa*p[3]+0.5f);
            int y2=int(sa*p[2]+ca*p[3]+0.5f);
            int v1=img[iclamp(k.y+y1,0,H-1)*W+iclamp(k.x+x1,0,W-1)];
            int v2=img[iclamp(k.y+y2,0,H-1)*W+iclamp(k.x+x2,0,W-1)];
            if(v1<v2) bits[b>>5] |= 1u<<(b&31);
        }
        std::memcpy(&desc[i*8],bits,32);
    }
}

static void cpuMatch(const std::vector<uint32_t>& d1,const std::vector<uint32_t>& d2,
                     float ratio,std::vector<uint16_t>& best)
{
    int n1=d1.size()/8, n2=d2.size()/8;
    best.assign(n1,0xFFFF);
    for(int i=0;i<n1;++i){
        int bestD=512,sec=512,bestJ=-1;
        for(int j=0;j<n2;++j){
            int d=0;
            for(int w=0;w<8;++w) d+=pop32(d1[i*8+w]^d2[j*8+w]);
            if(d<bestD){ sec=bestD; bestD=d; bestJ=j; }
            else if(d<sec) sec=d;
        }
        if(bestD<ratio*sec) best[i]=uint16_t(bestJ);
    }
}