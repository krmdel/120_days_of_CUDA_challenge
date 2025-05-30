#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <chrono>
#include <fstream>
#include <iostream>
#include <vector>

#define CUDA_CHECK(call)                                                         \
    do {                                                                         \
        cudaError_t err = (call);                                                \
        if (err != cudaSuccess) {                                                \
            std::cerr << "CUDA error: " << cudaGetErrorString(err)               \
                      << " (" << __FILE__ << ':' << __LINE__ << ")\n";           \
            std::exit(EXIT_FAILURE);                                             \
        }                                                                        \
    } while (0)

// Utility: read / write raw (binary) PGM
bool readPGM(const char* fname, std::vector<uint8_t>& img, int& w, int& h)
{
    std::ifstream ifs(fname, std::ios::binary);
    if (!ifs) return false;
    std::string magic;  ifs >> magic;
    if (magic != "P5") return false;
    ifs >> w >> h;
    int maxval; ifs >> maxval;
    ifs.get();                        // consume single whitespace
    if (maxval != 255) return false;  // only 8-bit supported
    img.resize(size_t(w) * h);
    ifs.read(reinterpret_cast<char*>(img.data()), img.size());
    return bool(ifs);
}
bool writePGM(const char* fname, const std::vector<uint8_t>& img, int w, int h)
{
    std::ofstream ofs(fname, std::ios::binary);
    if (!ofs) return false;
    ofs << "P5\n" << w << ' ' << h << "\n255\n";
    ofs.write(reinterpret_cast<const char*>(img.data()), img.size());
    return bool(ofs);
}

// DCT constants
__host__ __device__ inline float alpha(int u)            // normalisation factor
{
    return u == 0 ? 0.353553390593273762f /* √(1/8) */ : 0.5f /* √(2/8) */;
}
__host__ __device__ inline float cos8(int k, int n)      // cos((2k+1) n π / 16)
{
    return cosf((float)(2 * k + 1) * n * 0.196349540849362077f); // π/16
}

// CPU reference: forward & inverse 8 × 8 DCT on a single block
static void dct8x8_cpu_block(const float in[8][8], float out[8][8])
{
    for (int u = 0; u < 8; ++u)
        for (int v = 0; v < 8; ++v) {
            float sum = 0.f;
            for (int x = 0; x < 8; ++x)
                for (int y = 0; y < 8; ++y)
                    sum += in[x][y] * cos8(x, u) * cos8(y, v);
            out[u][v] = alpha(u) * alpha(v) * sum;
        }
}
static void idct8x8_cpu_block(const float in[8][8], float out[8][8])
{
    for (int x = 0; x < 8; ++x)
        for (int y = 0; y < 8; ++y) {
            float sum = 0.f;
            for (int u = 0; u < 8; ++u)
                for (int v = 0; v < 8; ++v)
                    sum += alpha(u) * alpha(v) * in[u][v] * cos8(x, u) * cos8(y, v);
            out[x][y] = sum;
        }
}

// Process entire image (row-major, pitch == width)
static void forwardDCT_CPU(const float* src, float* dst, int w, int h)
{
    for (int by = 0; by < h; by += 8)
        for (int bx = 0; bx < w; bx += 8) {
            float blk[8][8], coeff[8][8];
            for (int x = 0; x < 8; ++x)
                for (int y = 0; y < 8; ++y)
                    blk[x][y] = src[(by + y) * w + bx + x];
            dct8x8_cpu_block(blk, coeff);
            for (int u = 0; u < 8; ++u)
                for (int v = 0; v < 8; ++v)
                    dst[(by + v) * w + bx + u] = coeff[u][v];
        }
}
static void inverseDCT_CPU(const float* src, float* dst, int w, int h)
{
    for (int by = 0; by < h; by += 8)
        for (int bx = 0; bx < w; bx += 8) {
            float coeff[8][8], blk[8][8];
            for (int u = 0; u < 8; ++u)
                for (int v = 0; v < 8; ++v)
                    coeff[u][v] = src[(by + v) * w + bx + u];
            idct8x8_cpu_block(coeff, blk);
            for (int x = 0; x < 8; ++x)
                for (int y = 0; y < 8; ++y)
                    dst[(by + y) * w + bx + x] = blk[x][y];
        }
}

// GPU: naïve forward / inverse 8 × 8 kernels (global memory only)
__global__ void dct8x8_forward_naive(const float* __restrict__ src,
                                     float* __restrict__ dst,
                                     int width, int height)
{
    int bx = blockIdx.x * 8;
    int by = blockIdx.y * 8;
    int u  = threadIdx.x;   // 0-7
    int v  = threadIdx.y;   // 0-7
    if (bx + u >= width || by + v >= height) return;

    float sum = 0.f;
    for (int x = 0; x < 8; ++x)
        for (int y = 0; y < 8; ++y) {
            float pixel = src[(by + y) * width + (bx + x)];
            sum += pixel * cos8(x, u) * cos8(y, v);
        }
    dst[(by + v) * width + (bx + u)] = alpha(u) * alpha(v) * sum;
}

__global__ void dct8x8_inverse_naive(const float* __restrict__ src,
                                     float* __restrict__ dst,
                                     int width, int height)
{
    int bx = blockIdx.x * 8;
    int by = blockIdx.y * 8;
    int x  = threadIdx.x;   // 0-7
    int y  = threadIdx.y;   // 0-7
    if (bx + x >= width || by + y >= height) return;

    float sum = 0.f;
    for (int u = 0; u < 8; ++u)
        for (int v = 0; v < 8; ++v) {
            float coeff = src[(by + v) * width + (bx + u)];
            sum += alpha(u) * alpha(v) * coeff * cos8(x, u) * cos8(y, v);
        }
    dst[(by + y) * width + (bx + x)] = sum;
}

// PSNR
double psnr(const std::vector<float>& a, const std::vector<float>& b)
{
    double mse = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        double diff = a[i] - b[i];
        mse += diff * diff;
    }
    mse /= a.size();
    return 10.0 * std::log10(255.0 * 255.0 / mse);
}

int main(int argc, char** argv)
{
    // Load or generate image
    int width = 1024, height = 1024;
    std::vector<uint8_t> img8;
    if (argc >= 2) {
        if (!readPGM(argv[1], img8, width, height)) {
            std::cerr << "Failed to read PGM '" << argv[1] << "'\n";
            return EXIT_FAILURE;
        }
        std::cout << "Loaded " << width << "×" << height << " PGM\n";
    } else {
        img8.resize(size_t(width) * height);
        std::srand(1234);
        for (auto& p : img8)
            p = static_cast<uint8_t>(std::rand() & 0xFF);
        std::cout << "Generated random " << width << "×" << height << " image\n";
    }
    if (width % 8 || height % 8) {
        std::cerr << "Image dimensions must be multiples of 8\n";
        return EXIT_FAILURE;
    }

    // Convert to float
    std::vector<float> h_src(width * height);
    for (size_t i = 0; i < h_src.size(); ++i) h_src[i] = static_cast<float>(img8[i]);

    // CPU reference
    std::vector<float> h_coeff_cpu(width * height), h_recon_cpu(width * height);
    auto t0 = std::chrono::high_resolution_clock::now();
    forwardDCT_CPU(h_src.data(), h_coeff_cpu.data(), width, height);
    inverseDCT_CPU(h_coeff_cpu.data(), h_recon_cpu.data(), width, height);
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    double psnr_cpu = psnr(h_src, h_recon_cpu);

    // GPU buffers
    float *d_src, *d_coeff, *d_recon;
    size_t bytes = size_t(width) * height * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_src,   bytes));
    CUDA_CHECK(cudaMalloc(&d_coeff, bytes));
    CUDA_CHECK(cudaMalloc(&d_recon, bytes));

    // Events
    cudaEvent_t ev_total_start, ev_total_stop,
                ev_h2d_start,   ev_h2d_stop,
                ev_k_start,     ev_k_stop,
                ev_d2h_start,   ev_d2h_stop;
    CUDA_CHECK(cudaEventCreate(&ev_total_start));
    CUDA_CHECK(cudaEventCreate(&ev_total_stop));
    CUDA_CHECK(cudaEventCreate(&ev_h2d_start));
    CUDA_CHECK(cudaEventCreate(&ev_h2d_stop));
    CUDA_CHECK(cudaEventCreate(&ev_k_start));
    CUDA_CHECK(cudaEventCreate(&ev_k_stop));
    CUDA_CHECK(cudaEventCreate(&ev_d2h_start));
    CUDA_CHECK(cudaEventCreate(&ev_d2h_stop));

    CUDA_CHECK(cudaEventRecord(ev_total_start));

    // H2D copy
    CUDA_CHECK(cudaEventRecord(ev_h2d_start));
    CUDA_CHECK(cudaMemcpy(d_src, h_src.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(ev_h2d_stop));
    CUDA_CHECK(cudaEventSynchronize(ev_h2d_stop));

    // Kernel launch
    dim3 block(8, 8);
    dim3 grid(width / 8, height / 8);
    CUDA_CHECK(cudaEventRecord(ev_k_start));
    dct8x8_forward_naive<<<grid, block>>>(d_src, d_coeff, width, height);
    dct8x8_inverse_naive<<<grid, block>>>(d_coeff, d_recon, width, height);
    CUDA_CHECK(cudaEventRecord(ev_k_stop));
    CUDA_CHECK(cudaEventSynchronize(ev_k_stop));

    // D2H copy
    std::vector<float> h_recon_gpu(width * height);
    CUDA_CHECK(cudaEventRecord(ev_d2h_start));
    CUDA_CHECK(cudaMemcpy(h_recon_gpu.data(), d_recon, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(ev_d2h_stop));
    CUDA_CHECK(cudaEventSynchronize(ev_d2h_stop));

    CUDA_CHECK(cudaEventRecord(ev_total_stop));
    CUDA_CHECK(cudaEventSynchronize(ev_total_stop));

    float h2d_ms = 0.f, k_ms = 0.f, d2h_ms = 0.f, total_ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&h2d_ms, ev_h2d_start, ev_h2d_stop));
    CUDA_CHECK(cudaEventElapsedTime(&k_ms,   ev_k_start,   ev_k_stop));
    CUDA_CHECK(cudaEventElapsedTime(&d2h_ms, ev_d2h_start, ev_d2h_stop));
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, ev_total_start, ev_total_stop));

    double psnr_gpu = psnr(h_src, h_recon_gpu);

    // Timings
    std::cout << "CPU   forward+inverse    : " << cpu_ms << " ms\n";
    std::cout << "GPU   H2D copy           : " << h2d_ms << " ms\n";
    std::cout << "GPU   kernel (Fwd+Inv)   : " << k_ms   << " ms\n";
    std::cout << "GPU   D2H copy           : " << d2h_ms << " ms\n";
    std::cout << "GPU   total time         : " << total_ms << " ms\n";
    std::cout << "PSNR  CPU reconstruction : " << psnr_cpu << " dB\n";
    std::cout << "PSNR  GPU reconstruction : " << psnr_gpu << " dB\n";

    CUDA_CHECK(cudaFree(d_src));
    CUDA_CHECK(cudaFree(d_coeff));
    CUDA_CHECK(cudaFree(d_recon));
    CUDA_CHECK(cudaEventDestroy(ev_total_start));
    CUDA_CHECK(cudaEventDestroy(ev_total_stop));
    CUDA_CHECK(cudaEventDestroy(ev_h2d_start));
    CUDA_CHECK(cudaEventDestroy(ev_h2d_stop));
    CUDA_CHECK(cudaEventDestroy(ev_k_start));
    CUDA_CHECK(cudaEventDestroy(ev_k_stop));
    CUDA_CHECK(cudaEventDestroy(ev_d2h_start));
    CUDA_CHECK(cudaEventDestroy(ev_d2h_stop));
    return EXIT_SUCCESS;
}