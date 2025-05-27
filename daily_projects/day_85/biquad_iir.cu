#include <cuda_runtime.h>

#include <vector>
#include <random>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <cstdlib>
#include <algorithm>          // std::max

// Helpers
#define THREADS 256
#define CUDA_CHECK(x)  do{ cudaError_t rc=(x); if(rc!=cudaSuccess){            \
    std::cerr<<"CUDA "<<cudaGetErrorString(rc)<<" @"<<__FILE__<<":"<<__LINE__;\
    std::exit(EXIT_FAILURE);} }while(0)

// Coefficients kept in constant memory
__constant__ float d_b[3];          // b0 b1 b2
__constant__ float d_a[2];          // a1 a2 (a0 = 1)

// 1) serial recursion – one thread processes the whole stream
__global__ void biquad_serial(const float* __restrict__ x,
                              float*       __restrict__ y,
                              int N)
{
    float x1 = 0.f, x2 = 0.f;
    float y1 = 0.f, y2 = 0.f;
    const float b0=d_b[0], b1=d_b[1], b2=d_b[2];
    const float a1=d_a[0], a2=d_a[1];

    for(int n=0;n<N;++n){
        float out = b0*x[n] + b1*x1 + b2*x2 - a1*y1 - a2*y2;
        y[n] = out;
        x2 = x1; x1 = x[n];
        y2 = y1; y1 = out;
    }
}

// 2) frame recursion – one thread per frame
struct Hist { float x1,x2,y1,y2; };

__global__ void biquad_frame(const float* __restrict__ x,
                             float*       __restrict__ y,
                             const Hist*  __restrict__ hPrev,
                             int F,int N,int Nframes)
{
    int f = blockIdx.x*blockDim.x + threadIdx.x;
    if(f >= Nframes) return;

    const int start=f*F;
    const int stop =min(start+F,N);

    float x1=hPrev[f].x1, x2=hPrev[f].x2;
    float y1=hPrev[f].y1, y2=hPrev[f].y2;

    const float b0=d_b[0], b1=d_b[1], b2=d_b[2];
    const float a1=d_a[0], a2=d_a[1];

    for(int n=start;n<stop;++n){
        float xn  = x[n];
        float out = b0*xn + b1*x1 + b2*x2 - a1*y1 - a2*y2;
        y[n]=out;
        x2=x1; x1=xn;
        y2=y1; y1=out;
    }
}

// Timing struct
struct Times{ float h2d, ker, d2h; float total()const{return h2d+ker+d2h;} };

int main(int argc,char**argv)
{
    const int FRAME = (argc>1)? std::atoi(argv[1]) : 8192;
    const int N     = (argc>2)? std::atoi(argv[2]) : 1'048'576;
    const int Nframes = (N + FRAME - 1) / FRAME;

    std::cout<<"Samples "<<N<<",   frame "<<FRAME<<",   #frames "<<Nframes<<"\n\n";

    // Host signal
    std::vector<float> h_x(N);
    std::mt19937 rng(0); std::uniform_real_distribution<float> dist(-1,1);
    for(float& v:h_x) v=dist(rng);

    // Biquad coeffs (Butterworth LP, fc = 0.1 · Fs)
    float b[3] = {0.2929f, 0.5858f, 0.2929f};
    float a[2] = {-0.0000f, 0.1716f};
    CUDA_CHECK(cudaMemcpyToSymbol(d_b,b,sizeof(b)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_a,a,sizeof(a)));

    // Device buffers
    float *d_in,*d_y_ser,*d_y_fr;
    CUDA_CHECK(cudaMalloc(&d_in ,sizeof(float)*N));
    CUDA_CHECK(cudaMalloc(&d_y_ser,sizeof(float)*N));
    CUDA_CHECK(cudaMalloc(&d_y_fr ,sizeof(float)*N));

    // Host→Device copy (shared by both engines)
    cudaEvent_t eH0,eH1; cudaEventCreate(&eH0); cudaEventCreate(&eH1);
    cudaEventRecord(eH0);
    CUDA_CHECK(cudaMemcpy(d_in,h_x.data(),sizeof(float)*N,cudaMemcpyHostToDevice));
    cudaEventRecord(eH1); CUDA_CHECK(cudaDeviceSynchronize());
    float h2d_ms; cudaEventElapsedTime(&h2d_ms,eH0,eH1);
    cudaEventDestroy(eH0); cudaEventDestroy(eH1);

    // 1) Serial recursion
    Times serial;
    {
        cudaEvent_t k0,k1,c0,c1; cudaEventCreate(&k0); cudaEventCreate(&k1);
        cudaEventCreate(&c0); cudaEventCreate(&c1);

        cudaEventRecord(k0);
        biquad_serial<<<1,1>>>(d_in,d_y_ser,N);
        cudaEventRecord(k1);

        std::vector<float> y_ser(N);
        cudaEventRecord(c0);
        CUDA_CHECK(cudaMemcpy(y_ser.data(),d_y_ser,sizeof(float)*N,
                              cudaMemcpyDeviceToHost));
        cudaEventRecord(c1); CUDA_CHECK(cudaDeviceSynchronize());

        cudaEventElapsedTime(&serial.ker ,k0,k1);
        cudaEventElapsedTime(&serial.d2h,c0,c1);
        serial.h2d = h2d_ms;

        cudaEventDestroy(k0); cudaEventDestroy(k1);
        cudaEventDestroy(c0); cudaEventDestroy(c1);

        // Keep y_ser for accuracy check later
        std::swap(h_x, y_ser);        // Reuse memory to save space: h_x now holds ref
    }

    // Prepare history for frame engine
    std::vector<Hist> h_hist(Nframes);
    {
        float x1=0,x2=0,y1=0,y2=0;
        for(int f=0;f<Nframes;++f){
            h_hist[f]={x1,x2,y1,y2};
            int start=f*FRAME, stop=std::min(start+FRAME,N);
            for(int n=start;n<stop;++n){
                float xn=h_x[n];                      // h_x currently holds serial output!
                float out=b[0]*xn+b[1]*x1+b[2]*x2 - a[0]*y1 - a[1]*y2;
                x2=x1; x1=xn; y2=y1; y1=out;
            }
        }
    }
    Hist* d_hist; CUDA_CHECK(cudaMalloc(&d_hist,sizeof(Hist)*Nframes));
    CUDA_CHECK(cudaMemcpy(d_hist,h_hist.data(),sizeof(Hist)*Nframes,
                          cudaMemcpyHostToDevice));

    // 2) Frame recursion
    Times frame;
    {
        cudaEvent_t k0,k1,c0,c1; cudaEventCreate(&k0); cudaEventCreate(&k1);
        cudaEventCreate(&c0); cudaEventCreate(&c1);

        dim3 grid((Nframes+THREADS-1)/THREADS), block(THREADS);

        cudaEventRecord(k0);
        biquad_frame<<<grid,block>>>(d_in,d_y_fr,d_hist,FRAME,N,Nframes);
        cudaEventRecord(k1);

        std::vector<float> y_fr(N);
        cudaEventRecord(c0);
        CUDA_CHECK(cudaMemcpy(y_fr.data(),d_y_fr,sizeof(float)*N,
                              cudaMemcpyDeviceToHost));
        cudaEventRecord(c1); CUDA_CHECK(cudaDeviceSynchronize());

        cudaEventElapsedTime(&frame.ker ,k0,k1);
        cudaEventElapsedTime(&frame.d2h,c0,c1);
        frame.h2d = h2d_ms;

        /* accuracy check vs serial */
        double maxDiff=0;
        for(int i=0;i<N;++i)
            maxDiff=std::max(maxDiff,
                             std::fabs(static_cast<double>(h_x[i])-y_fr[i]));
        std::cout<<"Max |serial-frame| = "<<maxDiff<<"\n\n";

        cudaEventDestroy(k0); cudaEventDestroy(k1);
        cudaEventDestroy(c0); cudaEventDestroy(c1);
    }

    // Timing report
    auto line=[&](const char* n,const Times& t){
        std::cout<<std::setw(17)<<std::left<<n
                 <<" H2D "<<std::setw(7)<<t.h2d
                 <<" Kern "<<std::setw(8)<<t.ker
                 <<" D2H "<<std::setw(7)<<t.d2h
                 <<" Total "<<t.total()<<" ms\n";
    };
    std::cout<<std::fixed<<std::setprecision(3);
    std::cout<<"                    H2D      Kern     D2H     Total\n";
    line("Serial recursion", serial);
    line("Frame  recursion", frame);

    cudaFree(d_in); cudaFree(d_y_ser); cudaFree(d_y_fr); cudaFree(d_hist);
    return 0;
}