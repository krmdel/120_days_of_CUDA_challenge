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

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// CUDA check macro
#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                            \
            std::cerr << "CUDA error " << cudaGetErrorString(err)            \
                      << " (" << __FILE__ << ":" << __LINE__ << ")\n";       \
            std::exit(EXIT_FAILURE);                                         \
        }                                                                    \
    } while (0)


// CUDA Kernel
// 2-D baseline DFT: Each thread computes one output frequency bin (u,v)
// ─────────────────────────────────────────────────────────────────────────────
__global__ void dft2d_kernel(const float* __restrict__ src,
                             cuFloatComplex* __restrict__ dst,
                             int width, int height) {
    int u = blockIdx.x * blockDim.x + threadIdx.x;  // frequency coordinate
    int v = blockIdx.y * blockDim.y + threadIdx.y;
    if (u >= width || v >= height) return;

    float real = 0.0f;
    float imag = 0.0f;

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float val = src[y * width + x];
            float angle = -2.0f * M_PI * (static_cast<float>(u) * x / width +
                                          static_cast<float>(v) * y / height);
            float s, c;
            sincosf(angle, &s, &c);
            real += val * c;
            imag += val * s;
        }
    }
    dst[v * width + u] = make_cuFloatComplex(real, imag);
}

// CPU baseline 2-D DFT (single-precision)
void dft2d_cpu(const std::vector<float>& src,
               std::vector<std::complex<float>>& dst,
               int width, int height) {
    const float wInv = 1.0f / static_cast<float>(width);
    const float hInv = 1.0f / static_cast<float>(height);

    for (int v = 0; v < height; ++v) {
        for (int u = 0; u < width; ++u) {
            float real = 0.0f;
            float imag = 0.0f;
            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                    float angle = -2.0f * M_PI * (u * x * wInv + v * y * hInv);
                    float s = std::sin(angle);
                    float c = std::cos(angle);
                    float val = src[y * width + x];
                    real += val * c;
                    imag += val * s;
                }
            }
            dst[v * width + u] = {real, imag};
        }
    }
}

int main(int argc, char** argv) {
    const int W = (argc > 1) ? std::atoi(argv[1]) : 128;   // image width
    const int H = (argc > 2) ? std::atoi(argv[2]) : 128;   // image height
    const size_t N = static_cast<size_t>(W) * H;

    std::cout << "Image size: " << W << " x " << H << " (" << N
              << " pixels)\n\n";

    // Generate random image
    std::vector<float> h_img(N);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (auto& p : h_img) p = dist(rng);

    // CPU DFT
    std::vector<std::complex<float>> h_dft_cpu(N);
    auto t0 = std::chrono::high_resolution_clock::now();
    dft2d_cpu(h_img, h_dft_cpu, W, H);
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpu_ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "CPU DFT     : " << std::fixed << std::setprecision(3)
              << cpu_ms << " ms\n";

    // Allocate and copy data to GPU
    float*  d_img  = nullptr;
    cuFloatComplex* d_dft = nullptr;
    CUDA_CHECK(cudaMalloc(&d_img,  N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dft, N * sizeof(cuFloatComplex)));

    cudaEvent_t evH2D_start, evH2D_end, evK_start, evK_end, evD2H_start,
        evD2H_end;
    cudaEventCreate(&evH2D_start);
    cudaEventCreate(&evH2D_end);
    cudaEventCreate(&evK_start);
    cudaEventCreate(&evK_end);
    cudaEventCreate(&evD2H_start);
    cudaEventCreate(&evD2H_end);

    cudaEventRecord(evH2D_start);
    CUDA_CHECK(cudaMemcpy(d_img, h_img.data(), N * sizeof(float),
                          cudaMemcpyHostToDevice));
    cudaEventRecord(evH2D_end);

    // Launch kernel
    dim3 block(16, 16);
    dim3 grid((W + block.x - 1) / block.x, (H + block.y - 1) / block.y);

    cudaEventRecord(evK_start);
    dft2d_kernel<<<grid, block>>>(d_img, d_dft, W, H);
    cudaEventRecord(evK_end);

    // Copy data back to CPU
    std::vector<cuFloatComplex> h_dft_gpu(N);
    cudaEventRecord(evD2H_start);
    CUDA_CHECK(cudaMemcpy(h_dft_gpu.data(), d_dft,
                          N * sizeof(cuFloatComplex),
                          cudaMemcpyDeviceToHost));
    cudaEventRecord(evD2H_end);

    CUDA_CHECK(cudaDeviceSynchronize());

    // Timings
    float h2d_ms, k_ms, d2h_ms;
    cudaEventElapsedTime(&h2d_ms, evH2D_start, evH2D_end);
    cudaEventElapsedTime(&k_ms,   evK_start,   evK_end);
    cudaEventElapsedTime(&d2h_ms, evD2H_start, evD2H_end);

    std::cout << "GPU H2D     : " << h2d_ms << " ms\n";
    std::cout << "GPU Kernel  : " << k_ms   << " ms\n";
    std::cout << "GPU D2H     : " << d2h_ms << " ms\n";
    std::cout << "GPU Total   : " << (h2d_ms + k_ms + d2h_ms) << " ms\n";

    // Cleanup
    cudaFree(d_img);
    cudaFree(d_dft);
    cudaEventDestroy(evH2D_start);
    cudaEventDestroy(evH2D_end);
    cudaEventDestroy(evK_start);
    cudaEventDestroy(evK_end);
    cudaEventDestroy(evD2H_start);
    cudaEventDestroy(evD2H_end);

    return 0;
}