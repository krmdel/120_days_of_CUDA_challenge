#include <cuda_runtime.h>
#include <cuComplex.h>
#include <device_launch_parameters.h>

#include <chrono>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = (call);                                            \
        if (err != cudaSuccess) {                                            \
            std::cerr << "CUDA error: " << cudaGetErrorString(err)           \
                      << " (" << __FILE__ << ":" << __LINE__ << ")\n";       \
            std::exit(EXIT_FAILURE);                                         \
        }                                                                    \
    } while (0)

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// GPU kernels with tiling

constexpr int TILE = 256;     // must be power-of-two ≤ shared-mem / 8 bytes

// Forward transform: real input → complex output
__global__ void dft1d_fwd_shared(const float* __restrict__ x,
                                 cuFloatComplex* __restrict__ X,
                                 int N)
{
    extern __shared__ float sh[];                   // TILE floats
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= N) return;

    float re = 0.f, im = 0.f;
    const float w = -2.f * M_PI / N;

    for (int base = 0; base < N; base += TILE) {
        // Cooperative load
        int tid = threadIdx.x;
        if (tid < TILE && base + tid < N)
            sh[tid] = x[base + tid];
        __syncthreads();

        // Aaccumulate partial sum
        #pragma unroll
        for (int j = 0; j < TILE && base + j < N; ++j) {
            float ang = w * k * (base + j);
            float s, c;  sincosf(ang, &s, &c);
            float v = sh[j];
            re += v * c;
            im += v * s;
        }
        __syncthreads();
    }
    X[k] = make_cuFloatComplex(re, im);
}

// Inverse transform: complex input → complex output (divides by N)
__global__ void dft1d_inv_shared(const cuFloatComplex* __restrict__ X,
                                 cuFloatComplex* __restrict__ x,
                                 int N)
{
    extern __shared__ cuFloatComplex shc[];         // TILE complex samples
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= N) return;

    float re = 0.f, im = 0.f;
    const float w =  2.f * M_PI / N;

    for (int base = 0; base < N; base += TILE) {
        int tid = threadIdx.x;
        if (tid < TILE && base + tid < N)
            shc[tid] = X[base + tid];
        __syncthreads();

        #pragma unroll
        for (int j = 0; j < TILE && base + j < N; ++j) {
            float ang = w * k * (base + j);
            float s, c;  sincosf(ang, &s, &c);
            cuFloatComplex v = shc[j];
            re += v.x * c - v.y * s;
            im += v.x * s + v.y * c;
        }
        __syncthreads();
    }
    x[k] = make_cuFloatComplex(re / N, im / N);     // 1/N scaling
}

// CPU
void dft1d_cpu(const std::vector<float>& in,
               std::vector<std::complex<float>>& out)
{
    const int N = static_cast<int>(in.size());
    out.resize(N);
    const float w = -2.f * M_PI / N;

    for (int k = 0; k < N; ++k) {
        float re = 0.f, im = 0.f;
        for (int n = 0; n < N; ++n) {
            float ang = w * k * n;
            re += in[n] * std::cos(ang);
            im += in[n] * std::sin(ang);
        }
        out[k] = {re, im};
    }
}

void idft1d_cpu(const std::vector<std::complex<float>>& in,
                std::vector<std::complex<float>>& out)
{
    const int N = static_cast<int>(in.size());
    out.resize(N);
    const float w =  2.f * M_PI / N;

    for (int k = 0; k < N; ++k) {
        float re = 0.f, im = 0.f;
        for (int n = 0; n < N; ++n) {
            float ang = w * k * n;
            re +=  in[n].real() * std::cos(ang) - in[n].imag() * std::sin(ang);
            im +=  in[n].real() * std::sin(ang) + in[n].imag() * std::cos(ang);
        }
        out[k] = {re / N, im / N};
    }
}

int main(int argc, char** argv)
{
    const int N = (argc > 1) ? std::atoi(argv[1]) : 16'384;
    std::cout << "Vector length: " << N << " (shared-mem TILE = "
              << TILE << ")\n";

    // Host input
    std::vector<float> h_x(N);
    std::mt19937 rng(0);
    std::uniform_real_distribution<float> dist(0.f, 1.f);
    for (float& v : h_x) v = dist(rng);

    // CPU
    std::vector<std::complex<float>> h_X_cpu, h_x_inv_cpu;
    auto t0 = std::chrono::high_resolution_clock::now();
    dft1d_cpu(h_x, h_X_cpu);
    auto t1 = std::chrono::high_resolution_clock::now();
    idft1d_cpu(h_X_cpu, h_x_inv_cpu);
    auto t2 = std::chrono::high_resolution_clock::now();

    double cpu_f_ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count();
    double cpu_i_ms =
        std::chrono::duration<double, std::milli>(t2 - t1).count();

    std::cout << "CPU forward DFT : " << cpu_f_ms << " ms\n";
    std::cout << "CPU inverse DFT : " << cpu_i_ms << " ms\n\n";

    // GPU buffers
    float*          d_x  = nullptr;
    cuFloatComplex* d_X  = nullptr;
    cuFloatComplex* d_xi = nullptr;
    CUDA_CHECK(cudaMalloc(&d_x,  N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_X,  N * sizeof(cuFloatComplex)));
    CUDA_CHECK(cudaMalloc(&d_xi, N * sizeof(cuFloatComplex)));

    // Timing events
    cudaEvent_t eH2D, eKf0, eKf1, eKi0, eKi1, eD2Hf, eD2Hi;
    cudaEventCreate(&eH2D);  cudaEventCreate(&eKf0); cudaEventCreate(&eKf1);
    cudaEventCreate(&eKi0);  cudaEventCreate(&eKi1);
    cudaEventCreate(&eD2Hf); cudaEventCreate(&eD2Hi);

    // Launch parameters
    constexpr int THREADS = 256;
    const int BLOCKS = (N + THREADS - 1) / THREADS;

    // Forward DFT
    cudaEventRecord(eH2D);
    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), N * sizeof(float),
                          cudaMemcpyHostToDevice));

    cudaEventRecord(eKf0);
    dft1d_fwd_shared<<<BLOCKS, THREADS, TILE * sizeof(float)>>>(d_x, d_X, N);
    cudaEventRecord(eKf1);

    // Inverse DFT
    cudaEventRecord(eKi0);
    dft1d_inv_shared<<<BLOCKS, THREADS, TILE * sizeof(cuFloatComplex)>>>(
        d_X, d_xi, N);
    cudaEventRecord(eKi1);

    // Copy back
    std::vector<cuFloatComplex> h_X_gpu(N), h_x_inv_gpu(N);
    cudaEventRecord(eD2Hf);
    CUDA_CHECK(cudaMemcpy(h_X_gpu.data(), d_X,
                          N * sizeof(cuFloatComplex),
                          cudaMemcpyDeviceToHost));
    cudaEventRecord(eD2Hi);
    CUDA_CHECK(cudaMemcpy(h_x_inv_gpu.data(), d_xi,
                          N * sizeof(cuFloatComplex),
                          cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaDeviceSynchronize());

    // GPU timings
    float h2d_ms, kf_ms, ki_ms, d2hf_ms, d2hi_ms;
    cudaEventElapsedTime(&h2d_ms, eH2D, eKf0);
    cudaEventElapsedTime(&kf_ms,  eKf0, eKf1);
    cudaEventElapsedTime(&ki_ms,  eKi0, eKi1);
    cudaEventElapsedTime(&d2hf_ms, eKf1, eD2Hf);
    cudaEventElapsedTime(&d2hi_ms, eD2Hf, eD2Hi);

    std::cout << "GPU H2D copy     : " << h2d_ms  << " ms\n";
    std::cout << "GPU forward kern : " << kf_ms   << " ms\n";
    std::cout << "GPU forward D2H  : " << d2hf_ms << " ms\n";
    std::cout << "GPU inverse kern : " << ki_ms   << " ms\n";
    std::cout << "GPU inverse D2H  : " << d2hi_ms << " ms\n";

    // Clean-up
    cudaFree(d_x);  cudaFree(d_X);  cudaFree(d_xi);
    cudaEventDestroy(eH2D);  cudaEventDestroy(eKf0); cudaEventDestroy(eKf1);
    cudaEventDestroy(eKi0);  cudaEventDestroy(eKi1);
    cudaEventDestroy(eD2Hf); cudaEventDestroy(eD2Hi);

    return 0;
}