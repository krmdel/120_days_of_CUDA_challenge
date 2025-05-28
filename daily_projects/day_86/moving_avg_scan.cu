#include <cuda_runtime.h>

#include <thrust/device_ptr.h>
#include <thrust/scan.h>

#include <vector>
#include <random>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <algorithm>

#define THREADS 256
#define CUDA_CHECK(x)  do{ cudaError_t rc=(x); if(rc!=cudaSuccess){            \
    std::cerr<<"CUDA "<<cudaGetErrorString(rc)<<" @"<<__FILE__<<":"<<__LINE__;\
    std::exit(EXIT_FAILURE);} }while(0)

// CPU reference
void cpu_prefix(const float* in, float* pref, int N)
{
    float acc = 0.0f;
    for (int i = 0; i < N; ++i) {
        acc += in[i];
        pref[i] = acc;
    }
}
void cpu_mavg(const float* pref, float* avg, int N, int W)
{
    for (int i = 0; i < N; ++i) {
        float sum = pref[i] - (i >= W ? pref[i - W] : 0.0f);
        avg[i] = sum / W;
    }
}

// GPU kernels
__global__ void prefix_sum_naive(const float* in, float* pref, int N)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        float acc = 0.0f;
        for (int i = 0; i < N; ++i) {
            acc += in[i];
            pref[i] = acc;
        }
    }
}

__global__ void moving_avg_kernel(const float* pref, float* avg, int N, int W)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float sum = pref[idx] - (idx >= W ? pref[idx - W] : 0.0f);
        avg[idx] = sum / W;
    }
}

struct Times { float h2d = 0, ker = 0, d2h = 0; float total() const { return h2d + ker + d2h; } };

void print_line(const char* name, const Times& t)
{
    std::cout << std::setw(18) << std::left << name
              << " H2D "  << std::setw(7) << t.h2d
              << " Kern " << std::setw(8) << t.ker
              << " D2H "  << std::setw(7) << t.d2h
              << " Total " << t.total() << " ms\n";
}

int main(int argc, char** argv)
{
    const int N = (argc > 1) ? std::atoi(argv[1]) : 1'048'576;
    const int W = (argc > 2) ? std::atoi(argv[2]) : 64;
    std::cout << "Samples " << N << ", window " << W << "\n\n";

    // Host data
    std::vector<float> h_x(N);
    std::mt19937 rng(0); std::uniform_real_distribution<float> dist(-1, 1);
    for (float& v : h_x) v = dist(rng);

    // CPU reference
    std::vector<float> h_pref_cpu(N), h_mavg_cpu(N);
    cpu_prefix(h_x.data(), h_pref_cpu.data(), N);
    cpu_mavg(h_pref_cpu.data(), h_mavg_cpu.data(), N, W);

    // 1) Naïve engine
    Times naive;
    {
        float *d_in, *d_pref, *d_avg;
        CUDA_CHECK(cudaMalloc(&d_in  , sizeof(float) * N));
        CUDA_CHECK(cudaMalloc(&d_pref, sizeof(float) * N));
        CUDA_CHECK(cudaMalloc(&d_avg , sizeof(float) * N));

        cudaEvent_t eH0, eH1, k0, k1, c0, c1;
        cudaEventCreate(&eH0); cudaEventCreate(&eH1);
        cudaEventCreate(&k0 ); cudaEventCreate(&k1 );
        cudaEventCreate(&c0 ); cudaEventCreate(&c1 );

        // H2D copy
        cudaEventRecord(eH0);
        CUDA_CHECK(cudaMemcpy(d_in, h_x.data(), sizeof(float) * N, cudaMemcpyHostToDevice));
        cudaEventRecord(eH1); CUDA_CHECK(cudaDeviceSynchronize());
        cudaEventElapsedTime(&naive.h2d, eH0, eH1);

        // Kernel
        dim3 block(THREADS), grid((N + THREADS - 1) / THREADS);
        cudaEventRecord(k0);
        prefix_sum_naive<<<1, 1>>>(d_in, d_pref, N);
        moving_avg_kernel<<<grid, block>>>(d_pref, d_avg, N, W);
        cudaEventRecord(k1); CUDA_CHECK(cudaDeviceSynchronize());
        cudaEventElapsedTime(&naive.ker, k0, k1);

        // D2H copy
        std::vector<float> h_mavg_naive(N);
        cudaEventRecord(c0);
        CUDA_CHECK(cudaMemcpy(h_mavg_naive.data(), d_avg, sizeof(float) * N,
                              cudaMemcpyDeviceToHost));
        cudaEventRecord(c1); CUDA_CHECK(cudaDeviceSynchronize());
        cudaEventElapsedTime(&naive.d2h, c0, c1);

        // Accuracy check
        float maxDiff = 0.0f;
        for (int i = 0; i < N; ++i)
            maxDiff = std::max(maxDiff, fabsf(h_mavg_naive[i] - h_mavg_cpu[i]));
        std::cout << "Max |CPU - naïve| = " << maxDiff << "\n";

        // Cleanup
        cudaFree(d_in); cudaFree(d_pref); cudaFree(d_avg);
        cudaEventDestroy(eH0); cudaEventDestroy(eH1);
        cudaEventDestroy(k0);  cudaEventDestroy(k1);
        cudaEventDestroy(c0);  cudaEventDestroy(c1);
    }

    std::cout << '\n';

    // 2) thrust engine
    Times thrustT;
    {
        float *d_in, *d_pref, *d_avg;
        CUDA_CHECK(cudaMalloc(&d_in  , sizeof(float) * N));
        CUDA_CHECK(cudaMalloc(&d_pref, sizeof(float) * N));
        CUDA_CHECK(cudaMalloc(&d_avg , sizeof(float) * N));

        cudaEvent_t eH0, eH1, k0, k1, c0, c1;
        cudaEventCreate(&eH0); cudaEventCreate(&eH1);
        cudaEventCreate(&k0 ); cudaEventCreate(&k1 );
        cudaEventCreate(&c0 ); cudaEventCreate(&c1 );

        // H2D copy
        cudaEventRecord(eH0);
        CUDA_CHECK(cudaMemcpy(d_in, h_x.data(), sizeof(float) * N, cudaMemcpyHostToDevice));
        cudaEventRecord(eH1); CUDA_CHECK(cudaDeviceSynchronize());
        cudaEventElapsedTime(&thrustT.h2d, eH0, eH1);

        // Prefix scan + moving average
        dim3 block(THREADS), grid((N + THREADS - 1) / THREADS);
        cudaEventRecord(k0);

        thrust::device_ptr<float> d_in_ptr(d_in);
        thrust::device_ptr<float> d_pref_ptr(d_pref);
        thrust::inclusive_scan(d_in_ptr, d_in_ptr + N, d_pref_ptr);

        moving_avg_kernel<<<grid, block>>>(d_pref, d_avg, N, W);

        cudaEventRecord(k1); CUDA_CHECK(cudaDeviceSynchronize());
        cudaEventElapsedTime(&thrustT.ker, k0, k1);

        // D2H copy
        std::vector<float> h_mavg_thr(N);
        cudaEventRecord(c0);
        CUDA_CHECK(cudaMemcpy(h_mavg_thr.data(), d_avg, sizeof(float) * N,
                              cudaMemcpyDeviceToHost));
        cudaEventRecord(c1); CUDA_CHECK(cudaDeviceSynchronize());
        cudaEventElapsedTime(&thrustT.d2h, c0, c1);

        // Accuracy check
        float maxDiff = 0.0f;
        for (int i = 0; i < N; ++i)
            maxDiff = std::max(maxDiff, fabsf(h_mavg_thr[i] - h_mavg_cpu[i]));
        std::cout << "Max |CPU - thrust| = " << maxDiff << "\n";

        // Cleanup
        cudaFree(d_in); cudaFree(d_pref); cudaFree(d_avg);
        cudaEventDestroy(eH0); cudaEventDestroy(eH1);
        cudaEventDestroy(k0);  cudaEventDestroy(k1);
        cudaEventDestroy(c0);  cudaEventDestroy(c1);
    }

    // Reports
    std::cout << std::fixed << std::setprecision(3) << '\n';
    std::cout << "                       H2D      Kern      D2H     Total\n";
    print_line("Naïve (1-thread)", naive);
    print_line("Thrust scan    ", thrustT);

    return 0;
}