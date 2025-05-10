#include <cuda_runtime.h>
#include <cuComplex.h>
#include <device_launch_parameters.h>
#include <cufft.h>

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

// Error macros
#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                            \
            std::cerr << "CUDA error " << cudaGetErrorString(err)            \
                      << " (" << __FILE__ << ":" << __LINE__ << ")\n";       \
            std::exit(EXIT_FAILURE);                                         \
        }                                                                    \
    } while (0)

#define CUFFT_CHECK(call)                                                    \
    do {                                                                     \
        cufftResult err = call;                                              \
        if (err != CUFFT_SUCCESS) {                                          \
            std::cerr << "cuFFT error " << err                               \
                      << " (" << __FILE__ << ":" << __LINE__ << ")\n";       \
            std::exit(EXIT_FAILURE);                                         \
        }                                                                    \
    } while (0)

// Incremental rotation helper
__device__ __forceinline__ void rotate(float& c, float& s,
                                       const float cD, const float sD)
{
    const float t = c * cD - s * sD;
    s             = c * sD + s * cD;
    c             = t;
}

// Tiled kernel
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
            // Load tile to shared memory
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

                    rotate(c_x, s_x, c_dx, s_dx); // +1 pixel in x
                }
                rotate(c_y, s_y, c_dy, s_dy);     // +1 pixel in y
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

    // Generate random image (real)
    std::vector<float> h_img(N);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.f, 1.f);
    for (float& p : h_img) p = dist(rng);

    
    // Buffer Allocation
    float*           d_img      = nullptr;            // input for tiled kernel
    cuFloatComplex*  d_out_tile = nullptr;            // output tiled
    cuFloatComplex*  d_fft_in   = nullptr;            // cuFFT: C2C input
    cuFloatComplex*  d_fft_out  = nullptr;            // cuFFT output
    CUDA_CHECK(cudaMalloc(&d_img     , N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out_tile, N * sizeof(cuFloatComplex)));
    CUDA_CHECK(cudaMalloc(&d_fft_in  , N * sizeof(cuFloatComplex)));
    CUDA_CHECK(cudaMalloc(&d_fft_out , N * sizeof(cuFloatComplex)));

    // Host complex buffer for cuFFT
    std::vector<cuFloatComplex> h_img_c(N);
    for (size_t i = 0; i < N; ++i)
        h_img_c[i] = make_cuFloatComplex(h_img[i], 0.f);

    // CUDA events
    cudaEvent_t H2D_tile_beg, H2D_tile_end,
                Kern_tile_beg, Kern_tile_end,
                D2H_tile_beg, D2H_tile_end,
                H2D_fft_beg,  H2D_fft_end,
                Kern_fft_beg, Kern_fft_end,
                D2H_fft_beg,  D2H_fft_end;
    cudaEventCreate(&H2D_tile_beg);  cudaEventCreate(&H2D_tile_end);
    cudaEventCreate(&Kern_tile_beg); cudaEventCreate(&Kern_tile_end);
    cudaEventCreate(&D2H_tile_beg);  cudaEventCreate(&D2H_tile_end);
    cudaEventCreate(&H2D_fft_beg);   cudaEventCreate(&H2D_fft_end);
    cudaEventCreate(&Kern_fft_beg);  cudaEventCreate(&Kern_fft_end);
    cudaEventCreate(&D2H_fft_beg);   cudaEventCreate(&D2H_fft_end);

    // Tiled path
    cudaEventRecord(H2D_tile_beg);
    CUDA_CHECK(cudaMemcpy(d_img, h_img.data(),
                          N * sizeof(float), cudaMemcpyHostToDevice));
    cudaEventRecord(H2D_tile_end);

    dim3 block(16, 16);
    dim3 grid((W + 15) / 16, (H + 15) / 16);
    cudaEventRecord(Kern_tile_beg);
    dft2d_kernel_tiled<<<grid, block>>>(d_img, d_out_tile, W, H);
    cudaEventRecord(Kern_tile_end);

    std::vector<cuFloatComplex> h_out_tile(N);
    cudaEventRecord(D2H_tile_beg);
    CUDA_CHECK(cudaMemcpy(h_out_tile.data(), d_out_tile,
                          N * sizeof(cuFloatComplex),
                          cudaMemcpyDeviceToHost));
    cudaEventRecord(D2H_tile_end);

    // cuFFT path
    cudaEventRecord(H2D_fft_beg);
    CUDA_CHECK(cudaMemcpy(d_fft_in, h_img_c.data(),
                          N * sizeof(cuFloatComplex), cudaMemcpyHostToDevice));
    cudaEventRecord(H2D_fft_end);

    cufftHandle plan;
    CUFFT_CHECK(cufftPlan2d(&plan, H, W, CUFFT_C2C));

    cudaEventRecord(Kern_fft_beg);
    CUFFT_CHECK(cufftExecC2C(plan, d_fft_in, d_fft_out, CUFFT_FORWARD));
    cudaEventRecord(Kern_fft_end);

    std::vector<cuFloatComplex> h_out_fft(N);
    cudaEventRecord(D2H_fft_beg);
    CUDA_CHECK(cudaMemcpy(h_out_fft.data(), d_fft_out,
                          N * sizeof(cuFloatComplex),
                          cudaMemcpyDeviceToHost));
    cudaEventRecord(D2H_fft_end);
    CUDA_CHECK(cudaDeviceSynchronize());
    cufftDestroy(plan);

    // Timing results
    float h2d_tile_ms, k_tile_ms, d2h_tile_ms;
    float h2d_fft_ms , k_fft_ms , d2h_fft_ms ;
    cudaEventElapsedTime(&h2d_tile_ms, H2D_tile_beg, H2D_tile_end);
    cudaEventElapsedTime(&k_tile_ms  , Kern_tile_beg, Kern_tile_end);
    cudaEventElapsedTime(&d2h_tile_ms, D2H_tile_beg, D2H_tile_end);
    cudaEventElapsedTime(&h2d_fft_ms , H2D_fft_beg , H2D_fft_end );
    cudaEventElapsedTime(&k_fft_ms   , Kern_fft_beg, Kern_fft_end);
    cudaEventElapsedTime(&d2h_fft_ms , D2H_fft_beg , D2H_fft_end);

    std::cout << std::fixed << std::setprecision(3);

    std::cout << "Tiled kernel:\n";
    std::cout << "GPU H2D     : " << h2d_tile_ms << " ms\n";
    std::cout << "GPU Kernel  : " << k_tile_ms   << " ms\n";
    std::cout << "GPU D2H     : " << d2h_tile_ms << " ms\n";
    std::cout << "GPU Total   : " << (h2d_tile_ms + k_tile_ms + d2h_tile_ms)
              << " ms\n\n";

    std::cout << "cuFFT       :\n";
    std::cout << "GPU H2D     : " << h2d_fft_ms  << " ms\n";
    std::cout << "GPU Kernel  : " << k_fft_ms    << " ms\n";
    std::cout << "GPU D2H     : " << d2h_fft_ms  << " ms\n";
    std::cout << "GPU Total   : " << (h2d_fft_ms + k_fft_ms + d2h_fft_ms)
              << " ms\n";

    // Clean-up
    cudaFree(d_img); cudaFree(d_out_tile);
    cudaFree(d_fft_in); cudaFree(d_fft_out);

    for (auto ev : {H2D_tile_beg, H2D_tile_end, Kern_tile_beg, Kern_tile_end,
                    D2H_tile_beg, D2H_tile_end, H2D_fft_beg,  H2D_fft_end,
                    Kern_fft_beg, Kern_fft_end, D2H_fft_beg,  D2H_fft_end})
        cudaEventDestroy(ev);

    return 0;
}
