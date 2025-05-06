#include <cuda_runtime.h>
#include <cuComplex.h>
#include <device_launch_parameters.h>

#include <chrono>
#include <cmath>
#include <complex>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#define CUDA_CHECK(x)                                                         \
    do {                                                                      \
        cudaError_t rc = (x);                                                 \
        if (rc != cudaSuccess) {                                              \
            std::cerr << "CUDA error: " << cudaGetErrorString(rc)             \
                      << " (" << __FILE__ << ":" << __LINE__ << ")\n";        \
            std::exit(EXIT_FAILURE);                                          \
        }                                                                     \
    } while (0)

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// GPU kernel: each thread computes one output coefficient k
__global__ void dft1d_kernel(const float* __restrict__ x,
                             cuFloatComplex* __restrict__ X,
                             int N) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= N) return;

    float real = 0.f, imag = 0.f;
    const float twopi_over_N = -2.f * M_PI / N;

    for (int n = 0; n < N; ++n) {
        float angle = twopi_over_N * k * n;
        float s, c;
        sincosf(angle, &s, &c);
        float val = x[n];
        real += val * c;
        imag += val * s;
    }
    X[k] = make_cuFloatComplex(real, imag);
}

// CPU reference (single-thread, baseline)
void dft1d_cpu(const std::vector<float>& in,
               std::vector<std::complex<float>>& out) {
    const int N = static_cast<int>(in.size());
    out.resize(N);
    const float twopi_over_N = -2.f * M_PI / N;

    for (int k = 0; k < N; ++k) {
        float real = 0.f, imag = 0.f;
        for (int n = 0; n < N; ++n) {
            float angle = twopi_over_N * k * n;
            real += in[n] * std::cos(angle);
            imag += in[n] * std::sin(angle);
        }
        out[k] = {real, imag};
    }
}

int main(int argc, char** argv) {
    const int N = (argc > 1) ? std::atoi(argv[1]) : 16384; // 128 x 128
    std::cout << "1-D signal length: " << N << "\n\n";

    // Generate random real signal
    std::vector<float> h_signal(N);
    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(0.f, 1.f);
    for (float& v : h_signal) v = dist(rng);

    // CPU baseline
    std::vector<std::complex<float>> h_dft_cpu;
    auto t0 = std::chrono::high_resolution_clock::now();
    dft1d_cpu(h_signal, h_dft_cpu);
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpu_ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "CPU DFT     : " << std::fixed << std::setprecision(3)
              << cpu_ms << " ms\n";

    // Allocate GPU buffers
    float*            d_signal = nullptr;
    cuFloatComplex*   d_dft    = nullptr;
    CUDA_CHECK(cudaMalloc(&d_signal, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dft,    N * sizeof(cuFloatComplex)));

    // Events for fine-grained timing
    cudaEvent_t eH2D0, eH2D1, eK0, eK1, eD2H0, eD2H1;
    cudaEventCreate(&eH2D0); cudaEventCreate(&eH2D1);
    cudaEventCreate(&eK0);   cudaEventCreate(&eK1);
    cudaEventCreate(&eD2H0); cudaEventCreate(&eD2H1);

    // H2D copy
    cudaEventRecord(eH2D0);
    CUDA_CHECK(cudaMemcpy(d_signal, h_signal.data(),
                          N * sizeof(float), cudaMemcpyHostToDevice));
    cudaEventRecord(eH2D1);

    // Launch kernel (1-D grid)
    const int THREADS = 256;
    const int BLOCKS  = (N + THREADS - 1) / THREADS;

    cudaEventRecord(eK0);
    dft1d_kernel<<<BLOCKS, THREADS>>>(d_signal, d_dft, N);
    cudaEventRecord(eK1);

    // Copy result back
    std::vector<cuFloatComplex> h_dft_gpu(N);
    cudaEventRecord(eD2H0);
    CUDA_CHECK(cudaMemcpy(h_dft_gpu.data(), d_dft,
                          N * sizeof(cuFloatComplex),
                          cudaMemcpyDeviceToHost));
    cudaEventRecord(eD2H1);

    CUDA_CHECK(cudaDeviceSynchronize());

    // GPU timings
    float h2d_ms, k_ms, d2h_ms;
    cudaEventElapsedTime(&h2d_ms, eH2D0, eH2D1);
    cudaEventElapsedTime(&k_ms,   eK0,   eK1);
    cudaEventElapsedTime(&d2h_ms, eD2H0, eD2H1);

    std::cout << "GPU H2D     : " << h2d_ms << " ms\n";
    std::cout << "GPU Kernel  : " << k_ms   << " ms\n";
    std::cout << "GPU D2H     : " << d2h_ms << " ms\n";
    std::cout << "GPU Total   : " << (h2d_ms + k_ms + d2h_ms) << " ms\n";

    // Cleanup
    cudaFree(d_signal);
    cudaFree(d_dft);
    cudaEventDestroy(eH2D0); cudaEventDestroy(eH2D1);
    cudaEventDestroy(eK0);   cudaEventDestroy(eK1);
    cudaEventDestroy(eD2H0); cudaEventDestroy(eD2H1);

    return 0;
}
