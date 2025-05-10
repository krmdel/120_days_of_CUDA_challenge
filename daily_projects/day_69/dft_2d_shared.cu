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
#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err = call;                                            \
        if (err != cudaSuccess) {                                          \
            std::cerr << "CUDA error " << cudaGetErrorString(err)          \
                      << " (" << __FILE__ << ":" << __LINE__ << ")\n";     \
            std::exit(EXIT_FAILURE);                                       \
        }                                                                  \
    } while (0)

// Incremental-rotation helper
__device__ __forceinline__ void rotate(float& c, float& s,
                                       const float cD, const float sD)
{
    const float t = c * cD - s * sD;   // cos(a+Δ)
    s             = c * sD + s * cD;   // sin(a+Δ)
    c             = t;
}

// Tiled shared-memory kernel
constexpr int TILE = 16;

__global__ void dft2d_kernel_tiled(const float* __restrict__ src,
                                   cuFloatComplex* __restrict__ dst,
                                   int W, int H)
{
    const int u = blockIdx.x * blockDim.x + threadIdx.x;
    const int v = blockIdx.y * blockDim.y + threadIdx.y;
    if (u >= W || v >= H) return;

    const float kx = -2.f * M_PI * static_cast<float>(u) / W;
    const float ky = -2.f * M_PI * static_cast<float>(v) / H;
    float s_dx, c_dx, s_dy, c_dy;
    sincosf(kx, &s_dx, &c_dx);
    sincosf(ky, &s_dy, &c_dy);

    __shared__ float tile[TILE][TILE];

    float accRe = 0.f, accIm = 0.f;

    for (int y0 = 0; y0 < H; y0 += TILE) {
        float s_yRow, c_yRow;
        sincosf(ky * static_cast<float>(y0), &s_yRow, &c_yRow);

        for (int x0 = 0; x0 < W; x0 += TILE) {

            // load TILE×TILE block into shared memory
            int gX = x0 + threadIdx.x;
            int gY = y0 + threadIdx.y;
            tile[threadIdx.y][threadIdx.x] =
                (gX < W && gY < H) ? src[gY * W + gX] : 0.f;
            __syncthreads();

            float s_xCol, c_xCol;
            sincosf(kx * static_cast<float>(x0), &s_xCol, &c_xCol);

            float s_y = s_yRow, c_y = c_yRow;
            #pragma unroll
            for (int ty = 0; ty < TILE && (y0 + ty) < H; ++ty) {

                float s_x = s_xCol, c_x = c_xCol;
                #pragma unroll
                for (int tx = 0; tx < TILE && (x0 + tx) < W; ++tx) {
                    const float cosTot = c_x * c_y - s_x * s_y;
                    const float sinTot = c_x * s_y + s_x * c_y;
                    const float val    = tile[ty][tx];
                    accRe += val * cosTot;
                    accIm += val * sinTot;

                    rotate(c_x, s_x, c_dx, s_dx);  // +1 in x
                }
                rotate(c_y, s_y, c_dy, s_dy);      // +1 in y
            }
            __syncthreads();
        }
    }
    dst[v * W + u] = make_cuFloatComplex(accRe, accIm);
}

int main(int argc, char** argv)
{
    const int W = (argc > 1) ? std::atoi(argv[1]) : 128;
    const int H = (argc > 2) ? std::atoi(argv[2]) : 128;
    const size_t N = static_cast<size_t>(W) * H;

    std::cout << "Image " << W << "×" << H
              << " (" << N << " px)\n\n";

    // Host image
    std::vector<float> h_img(N);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.f, 1.f);
    for (float& p : h_img) p = dist(rng);

    // Device buffers
    float* d_img = nullptr;
    cuFloatComplex* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_img, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(cuFloatComplex)));

    // Events for full timing
    cudaEvent_t eH2D_beg, eH2D_end,
                eKernel_beg, eKernel_end,
                eD2H_beg, eD2H_end;
    cudaEventCreate(&eH2D_beg);  cudaEventCreate(&eH2D_end);
    cudaEventCreate(&eKernel_beg); cudaEventCreate(&eKernel_end);
    cudaEventCreate(&eD2H_beg);  cudaEventCreate(&eD2H_end);

    // H2D
    cudaEventRecord(eH2D_beg);
    CUDA_CHECK(cudaMemcpy(d_img, h_img.data(),
                          N * sizeof(float), cudaMemcpyHostToDevice));
    cudaEventRecord(eH2D_end);

    // Kernel
    dim3 block(16, 16);
    dim3 grid((W + 15) / 16, (H + 15) / 16);
    cudaEventRecord(eKernel_beg);
    dft2d_kernel_tiled<<<grid, block>>>(d_img, d_out, W, H);
    cudaEventRecord(eKernel_end);

    // D2H
    std::vector<cuFloatComplex> h_out(N);
    cudaEventRecord(eD2H_beg);
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out,
                          N * sizeof(cuFloatComplex),
                          cudaMemcpyDeviceToHost));
    cudaEventRecord(eD2H_end);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Timings
    float h2d_ms  = 0.f, k_ms = 0.f, d2h_ms = 0.f;
    cudaEventElapsedTime(&h2d_ms,  eH2D_beg,   eH2D_end);
    cudaEventElapsedTime(&k_ms,    eKernel_beg, eKernel_end);
    cudaEventElapsedTime(&d2h_ms,  eD2H_beg,   eD2H_end);

    std::cout << "GPU H2D     : " << h2d_ms << " ms\n";
    std::cout << "GPU Kernel  : " << k_ms   << " ms\n";
    std::cout << "GPU D2H     : " << d2h_ms << " ms\n";
    std::cout << "GPU Total   : " << (h2d_ms + k_ms + d2h_ms) << " ms\n";

    // Clean up
    cudaFree(d_img); cudaFree(d_out);
    cudaEventDestroy(eH2D_beg);   cudaEventDestroy(eH2D_end);
    cudaEventDestroy(eKernel_beg); cudaEventDestroy(eKernel_end);
    cudaEventDestroy(eD2H_beg);   cudaEventDestroy(eD2H_end);
    return 0;
}
