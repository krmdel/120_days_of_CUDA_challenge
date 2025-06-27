#include <cuda_runtime.h>
#include <vector>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <chrono>
#include <random>
#include <cstring>
#include <cmath>

#if __has_include(<opencv2/imgcodecs.hpp>)
# define USE_OPENCV 1
# include <opencv2/imgcodecs.hpp>
#else
# define USE_OPENCV 0
#endif

// Helpers
#define CUDA_CHECK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){                 \
    std::cerr<<"CUDA error: "<<cudaGetErrorString(e)<<" ("<<__LINE__<<")\n";     \
    std::exit(1);} }while(0)
template<class TP> inline double ms(TP a,TP b){
    return std::chrono::duration<double,std::milli>(b-a).count(); }

// PGM
#if !USE_OPENCV
static bool readPGM(const char* fn,std::vector<uint8_t>& img,int& W,int& H){
    FILE* fp=fopen(fn,"rb"); if(!fp) return false;
    char m[3]; if(fscanf(fp,"%2s",m)!=1||strcmp(m,"P5")){fclose(fp);return false;}
    int maxv; if(fscanf(fp,"%d%d%d",&W,&H,&maxv)!=3||maxv!=255){fclose(fp);return false;}
    fgetc(fp);
    img.resize(size_t(W)*H);
    size_t rd=fread(img.data(),1,img.size(),fp);
    fclose(fp);
    return rd==img.size();
}
#endif

// Constants
constexpr int MAX_KP  = 4096;    // cap to keep runtime small
constexpr int PATCH_R = 16;

// FAST circle offsets (radius ≈9 px)
__device__ __constant__ int8_t c_cx[16]={0,3,6,8,8,6,3,0,-3,-6,-8,-8,-6,-3,0,0};
__device__ __constant__ int8_t c_cy[16]={8,6,3,0,-3,-6,-8,-8,-8,-6,-3,0,3,6,8,8};

// 256-bit BRIEF pattern (4 × 256 = 1024 B)
__constant__ int8_t c_pat[1024];

struct KP{ short x,y; };

// Clamp
__host__ __device__ inline int iclamp(int v,int l,int h){
    return v<l?l:(v>h?h:v);
}

// FAST-9 GPU kernel
__global__ void fast9Ker(const uint8_t* img,int W,int H,int thr,
                         KP* out,int* d_cnt)
{
    int x=blockIdx.x*blockDim.x+threadIdx.x;
    int y=blockIdx.y*blockDim.y+threadIdx.y;
    if(x<16||y<16||x>=W-16||y>=H-16) return;

    int idx=y*W+x, p=img[idx];
    int bright=0,dark=0;
    if(img[idx+ 9]   >p+thr) ++bright; else if(img[idx+ 9]   <p-thr) ++dark;
    if(img[idx- 9]   >p+thr) ++bright; else if(img[idx- 9]   <p-thr) ++dark;
    if(img[idx+ 9*W] >p+thr) ++bright; else if(img[idx+ 9*W] <p-thr) ++dark;
    if(img[idx- 9*W] >p+thr) ++bright; else if(img[idx- 9*W] <p-thr) ++dark;
    if(bright<3 && dark<3) return;

    int consec=0,maxc=0;
#pragma unroll
    for(int i=0;i<24;++i){
        int ii=i&15;
        int q=img[(y+c_cy[ii])*W + (x+c_cx[ii])];
        if(q>p+thr||q<p-thr) ++consec; else consec=0;
        if(consec>maxc) maxc=consec;
        if(maxc>=9) break;
    }
    if(maxc<9) return;

    int id=atomicAdd(d_cnt,1);
    if(id<MAX_KP) out[id]={short(x),short(y)};
}

// BRIEF-256 GPU kernel
__global__ void briefKer(const KP* kp,int n,const uint8_t* img,
                         uint32_t* desc,int W,int H)
{
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return;
    KP k=kp[i];

    // Orientation (intensity centroid)
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
        const int8_t* p=&c_pat[4*b];
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

// Brute-force Hamming matcher (tiled)
__global__ void matchKer(const uint32_t* desc,int n,int* best)
{
    __shared__ uint32_t sh[32][8];
    int tid=threadIdx.x;
    int i=blockIdx.x; if(i>=n) return;
    const uint32_t* q=&desc[i*8];
    int bestD=256;
    for(int base=0;base<n;base+=32){
        int idx=base+tid;
        if(idx<n)
#pragma unroll
            for(int w=0;w<8;++w) sh[tid][w]=desc[idx*8+w];
        __syncthreads();
        int lim=min(32,n-base);
        for(int j=0;j<lim;++j){
            if(base+j==i) continue;
            int d=0;
#pragma unroll
            for(int w=0;w<8;++w) d+=__popc(q[w]^sh[j][w]);
            if(d<bestD) bestD=d;
        }
        __syncthreads();
    }
    if(tid==0) best[i]=bestD;
}

// CPU reference pipeline (scalar, capped)
static inline uint32_t pop32(uint32_t x){ return __builtin_popcount(x); }

static void cpuFAST(const std::vector<uint8_t>& img,int W,int H,int thr,
                    std::vector<KP>& kp)
{
    static const int cx[16]={0,3,6,8,8,6,3,0,-3,-6,-8,-8,-6,-3,0,0};
    static const int cy[16]={8,6,3,0,-3,-6,-8,-8,-8,-6,-3,0,3,6,8,8};
    kp.clear(); kp.reserve(MAX_KP);
    auto at=[&](int x,int y){ return img[y*W+x]; };
    for(int y=16;y<H-16 && (int)kp.size()<MAX_KP;++y)
        for(int x=16;x<W-16 && (int)kp.size()<MAX_KP;++x){
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
                if(consec>maxc) maxc=consec;
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

static void cpuMatch(const std::vector<uint32_t>& desc,int n,std::vector<int>& best)
{
    best.assign(n,256);
    for(int i=0;i<n;++i){
        const uint32_t* A=&desc[i*8];
        int bd=256;
        for(int j=0;j<n;++j){ if(i==j) continue;
            const uint32_t* B=&desc[j*8];
            int d=0; for(int w=0;w<8;++w) d+=pop32(A[w]^B[w]);
            if(d<bd) bd=d;
        }
        best[i]=bd;
    }
}

// Pattern initialiser
static void initPattern(){
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> d(-PATCH_R,PATCH_R);
    int8_t host[1024]; for(auto& v:host) v=int8_t(d(rng));
    CUDA_CHECK(cudaMemcpyToSymbol(c_pat,host,1024));
}

int main(int argc,char** argv)
{
    int thr=(argc>=3)?std::stoi(argv[2]):25;

    int W,H; std::vector<uint8_t> img;
    if(argc>=2){
#if USE_OPENCV
        auto m=cv::imread(argv[1],cv::IMREAD_GRAYSCALE);
        if(m.empty()){ std::cerr<<"load err\n"; return 1; }
        W=m.cols; H=m.rows; img.assign(m.datastart,m.dataend);
#else
        if(!readPGM(argv[1],img,W,H)){ std::cerr<<"PGM err\n"; return 1; }
#endif
    }else{
        W=1920; H=1080; img.resize(size_t(W)*H);
        std::mt19937 rng(0); std::uniform_int_distribution<int>d(0,255);
        for(auto& v:img) v=d(rng);
        std::cout<<"[Info] random frame "<<W<<"×"<<H<<"\n";
    }
    initPattern();

    // CPU pipeline
    auto tc0=std::chrono::high_resolution_clock::now();

    std::vector<KP>   cpuKP;  cpuFAST(img,W,H,thr,cpuKP);
    // Cap to MAX_KP for fair match size
    if(cpuKP.size()>MAX_KP) cpuKP.resize(MAX_KP);

    int8_t hostPat[1024]; CUDA_CHECK(cudaMemcpyFromSymbol(hostPat,c_pat,1024));
    std::vector<uint32_t> cpuDesc; cpuBRIEF(img,W,H,cpuKP,cpuDesc,hostPat);

    std::vector<int> cpuBest; cpuMatch(cpuDesc,cpuKP.size(),cpuBest);

    auto tc1=std::chrono::high_resolution_clock::now();
    double cpu_ms=ms(tc0,tc1);

    // GPU pipeline
    const int N=W*H;
    uint8_t  *d_img ; CUDA_CHECK(cudaMalloc(&d_img ,N));
    KP       *d_kp  ; CUDA_CHECK(cudaMalloc(&d_kp  ,MAX_KP*sizeof(KP)));
    int      *d_cnt ; CUDA_CHECK(cudaMalloc(&d_cnt ,4));
    uint32_t *d_desc; CUDA_CHECK(cudaMalloc(&d_desc,MAX_KP*8*4));
    int      *d_best; CUDA_CHECK(cudaMalloc(&d_best,MAX_KP*4));

    cudaEvent_t evH0,evH1,evF0,evF1,evB0,evB1,evM0,evM1,evD0,evD1,evTot0,evTot1;
    for(auto p:{&evH0,&evH1,&evF0,&evF1,&evB0,&evB1,&evM0,&evM1,&evD0,&evD1,&evTot0,&evTot1})
        CUDA_CHECK(cudaEventCreate(p));

    cudaEventRecord(evTot0);

    cudaEventRecord(evH0);
    CUDA_CHECK(cudaMemcpy(d_img,img.data(),N,cudaMemcpyHostToDevice));
    cudaEventRecord(evH1);

    CUDA_CHECK(cudaMemset(d_cnt,0,4));
    dim3 blk(16,16), grd((W+15)/16,(H+15)/16);

    cudaEventRecord(evF0);
    fast9Ker<<<grd,blk>>>(d_img,W,H,thr,d_kp,d_cnt); CUDA_CHECK(cudaGetLastError());
    cudaEventRecord(evF1);

    int h_cnt; CUDA_CHECK(cudaMemcpy(&h_cnt,d_cnt,4,cudaMemcpyDeviceToHost));
    if(h_cnt>MAX_KP) h_cnt=MAX_KP;

    cudaEventRecord(evB0);
    if(h_cnt)
        briefKer<<<(h_cnt+255)/256,256>>>(d_kp,h_cnt,d_img,d_desc,W,H);
    cudaEventRecord(evB1);

    cudaEventRecord(evM0);
    if(h_cnt)
        matchKer<<<h_cnt,32>>>(d_desc,h_cnt,d_best);
    cudaEventRecord(evM1);

    std::vector<int> gpuBest(h_cnt);
    cudaEventRecord(evD0);
    if(h_cnt)
        CUDA_CHECK(cudaMemcpy(gpuBest.data(),d_best,h_cnt*4,cudaMemcpyDeviceToHost));
    cudaEventRecord(evD1);

    cudaEventRecord(evTot1); CUDA_CHECK(cudaEventSynchronize(evTot1));

    // Timings
    float tH2D,tFast,tBrief,tMatch,tD2H,gpuTot;
    cudaEventElapsedTime(&tH2D ,evH0,evH1);
    cudaEventElapsedTime(&tFast,evF0,evF1);
    cudaEventElapsedTime(&tBrief,evB0,evB1);
    cudaEventElapsedTime(&tMatch,evM0,evM1);
    cudaEventElapsedTime(&tD2H ,evD0,evD1);
    cudaEventElapsedTime(&gpuTot,evTot0,evTot1);

    std::cout<<"\nImage                   : "<<W<<" × "<<H<<"\n";
    std::cout<<"FAST threshold          : "<<thr<<"\n";
    std::cout<<"Keypoints (GPU)         : "<<h_cnt<<"\n";
    std::cout<<"Keypoints (CPU)         : "<<cpuKP.size()<<"\n\n";
    std::cout<<"GPU H2D copy            : "<<tH2D   <<" ms\n";
    std::cout<<"GPU FAST kernel         : "<<tFast  <<" ms\n";
    std::cout<<"GPU BRIEF kernel        : "<<tBrief <<" ms\n";
    std::cout<<"GPU Match kernel        : "<<tMatch <<" ms\n";
    std::cout<<"GPU D2H copy            : "<<tD2H   <<" ms\n";
    std::cout<<"GPU total               : "<<gpuTot <<" ms\n";
    std::cout<<"CPU total               : "<<cpu_ms <<" ms\n";

    CUDA_CHECK(cudaFree(d_img));
    CUDA_CHECK(cudaFree(d_kp));
    CUDA_CHECK(cudaFree(d_cnt));
    CUDA_CHECK(cudaFree(d_desc));
    CUDA_CHECK(cudaFree(d_best));
    return 0;
}