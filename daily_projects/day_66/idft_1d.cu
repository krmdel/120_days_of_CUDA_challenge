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

// GPU kernel:
// DIR = -1 → forward (-j 2π kn/N), real input, no scale
// DIR = +1 → inverse (+j 2π kn/N), complex input, ÷N scale
template<int DIR>
__global__ void dft1d_kernel(const float*          __restrict__ in_real,
                             const cuFloatComplex* __restrict__ in_cplx,
                             cuFloatComplex*       __restrict__ out,
                             int N)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= N) return;

    float re = 0.f, im = 0.f;
    const float w = DIR * 2.f * M_PI / N;        // sets sign

    if constexpr (DIR == -1) {                   // forward
        for (int n = 0; n < N; ++n) {
            float ang = w * k * n;
            float s, c;  sincosf(ang, &s, &c);
            float v = in_real[n];
            re += v * c;
            im += v * s;
        }
        out[k] = make_cuFloatComplex(re, im);
    } else {                                     // inverse
        for (int n = 0; n < N; ++n) {
            float ang = w * k * n;
            float s, c;  sincosf(ang, &s, &c);
            cuFloatComplex v = in_cplx[n];
            re += v.x * c - v.y * s;
            im += v.x * s + v.y * c;
        }
        out[k] = make_cuFloatComplex(re / N, im / N);
    }
}

// CPU
void dft1d_cpu_forward(const std::vector<float>& in,
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

void dft1d_cpu_inverse(const std::vector<std::complex<float>>& in,
                       std::vector<std::complex<float>>& out)
{
    const int N = static_cast<int>(in.size());
    out.resize(N);
    const float w = 2.f * M_PI / N;
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
    std::cout << "Vector length: " << N << "\n";

    // Create random real input
    std::vector<float> h_sig(N);
    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(0.f, 1.f);
    for (auto& v : h_sig) v = dist(rng);

    // CPU timings
    std::vector<std::complex<float>> h_dft_cpu, h_idft_cpu;
    auto t0 = std::chrono::high_resolution_clock::now();
    dft1d_cpu_forward(h_sig, h_dft_cpu);
    auto t1 = std::chrono::high_resolution_clock::now();
    dft1d_cpu_inverse(h_dft_cpu, h_idft_cpu);
    auto t2 = std::chrono::high_resolution_clock::now();

    double cpu_f_ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count();
    double cpu_i_ms =
        std::chrono::duration<double, std::milli>(t2 - t1).count();

    std::cout << "CPU forward DFT : " << cpu_f_ms << " ms\n";
    std::cout << "CPU inverse DFT : " << cpu_i_ms << " ms\n\n";

    // GPU setup
    float*          d_sig  = nullptr;
    cuFloatComplex* d_dft  = nullptr;
    cuFloatComplex* d_idft = nullptr;
    CUDA_CHECK(cudaMalloc(&d_sig,  N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dft,  N * sizeof(cuFloatComplex)));
    CUDA_CHECK(cudaMalloc(&d_idft, N * sizeof(cuFloatComplex)));

    const int THREADS = 256;
    const int BLOCKS  = (N + THREADS - 1) / THREADS;
    const size_t BYTES_R = N * sizeof(float);
    const size_t BYTES_C = N * sizeof(cuFloatComplex);

    cudaEvent_t eH2D, eKf0, eKf1, eKi0, eKi1, eD2Hf, eD2Hi;
    cudaEventCreate(&eH2D);  cudaEventCreate(&eKf0); cudaEventCreate(&eKf1);
    cudaEventCreate(&eKi0);  cudaEventCreate(&eKi1);
    cudaEventCreate(&eD2Hf); cudaEventCreate(&eD2Hi);

    // Forward DFT
    cudaEventRecord(eH2D);
    CUDA_CHECK(cudaMemcpy(d_sig, h_sig.data(), BYTES_R,
                          cudaMemcpyHostToDevice));

    cudaEventRecord(eKf0);
    dft1d_kernel<-1><<<BLOCKS, THREADS>>>(d_sig, nullptr, d_dft, N);
    cudaEventRecord(eKf1);

    // Inverse DFT
    cudaEventRecord(eKi0);
    dft1d_kernel<+1><<<BLOCKS, THREADS>>>(nullptr, d_dft, d_idft, N);
    cudaEventRecord(eKi1);

    // Copy results back
    std::vector<cuFloatComplex> h_dft_gpu(N), h_idft_gpu(N);
    cudaEventRecord(eD2Hf);
    CUDA_CHECK(cudaMemcpy(h_dft_gpu.data(), d_dft,  BYTES_C,
                          cudaMemcpyDeviceToHost));
    cudaEventRecord(eD2Hi);
    CUDA_CHECK(cudaMemcpy(h_idft_gpu.data(), d_idft, BYTES_C,
                          cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaDeviceSynchronize());

    // Timings
    float h2d_ms, kf_ms, ki_ms, d2hf_ms, d2hi_ms;
    cudaEventElapsedTime(&h2d_ms,  eH2D,  eKf0);
    cudaEventElapsedTime(&kf_ms,   eKf0,  eKf1);
    cudaEventElapsedTime(&ki_ms,   eKi0,  eKi1);
    cudaEventElapsedTime(&d2hf_ms, eKf1,  eD2Hf);
    cudaEventElapsedTime(&d2hi_ms, eD2Hf, eD2Hi);

    std::cout << "GPU H2D copy     : " << h2d_ms  << " ms\n";
    std::cout << "GPU forward kern : " << kf_ms   << " ms\n";
    std::cout << "GPU forward D2H  : " << d2hf_ms << " ms\n";
    std::cout << "GPU inverse kern : " << ki_ms   << " ms\n";
    std::cout << "GPU inverse D2H  : " << d2hi_ms << " ms\n";

    // Clean-up
    cudaFree(d_sig);  cudaFree(d_dft);  cudaFree(d_idft);
    cudaEventDestroy(eH2D);  cudaEventDestroy(eKf0); cudaEventDestroy(eKf1);
    cudaEventDestroy(eKi0);  cudaEventDestroy(eKi1);
    cudaEventDestroy(eD2Hf); cudaEventDestroy(eD2Hi);

    return 0;
}
