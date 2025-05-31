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

constexpr float PI = 3.14159265358979323846f;

//  Device-side cosine table
__constant__ float d_cos[64];

//  Host-side duplicate cosine table (for CPU reference)
static float h_cos[64];

//  Minimal PGM I/O helpers
bool readPGM(const char* fname, std::vector<uint8_t>& img, int& w, int& h)
{
    std::ifstream ifs(fname, std::ios::binary);
    if (!ifs) return false;
    std::string magic;  ifs >> magic;
    if (magic != "P5") return false;
    int maxval; ifs >> w >> h >> maxval;
    ifs.get();
    if (maxval != 255) return false;
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
__host__ __device__ inline float alpha(int k) { return k == 0 ? 0.353553390593273762f : 0.5f; }

// CPU reference (uses host-side cosine table)
static void dct8_block_cpu(const float in[8][8], float out[8][8])
{
    for (int u = 0; u < 8; ++u)
        for (int v = 0; v < 8; ++v) {
            float sum = 0.f;
            for (int x = 0; x < 8; ++x)
                for (int y = 0; y < 8; ++y)
                    sum += in[x][y] *
                           h_cos[u * 8 + x] *  // cos((2x+1)uπ/16)
                           h_cos[v * 8 + y];   // cos((2y+1)vπ/16)
            out[u][v] = alpha(u) * alpha(v) * sum;
        }
}
static void idct8_block_cpu(const float in[8][8], float out[8][8])
{
    for (int x = 0; x < 8; ++x)
        for (int y = 0; y < 8; ++y) {
            float sum = 0.f;
            for (int u = 0; u < 8; ++u)
                for (int v = 0; v < 8; ++v)
                    sum += alpha(u) * alpha(v) * in[u][v] *
                           h_cos[u * 8 + x] *  // cos((2x+1)uπ/16)
                           h_cos[v * 8 + y];   // cos((2y+1)vπ/16)
            out[x][y] = sum;
        }
}
static void forwardDCT_CPU(const float* s, float* d, int w, int h)
{
    for (int by = 0; by < h; by += 8)
        for (int bx = 0; bx < w; bx += 8) {
            float blk[8][8], coeff[8][8];
            for (int x = 0; x < 8; ++x)
                for (int y = 0; y < 8; ++y)
                    blk[x][y] = s[(by + y) * w + bx + x];
            dct8_block_cpu(blk, coeff);
            for (int u = 0; u < 8; ++u)
                for (int v = 0; v < 8; ++v)
                    d[(by + v) * w + bx + u] = coeff[u][v];
        }
}
static void inverseDCT_CPU(const float* s, float* d, int w, int h)
{
    for (int by = 0; by < h; by += 8)
        for (int bx = 0; bx < w; bx += 8) {
            float coeff[8][8], blk[8][8];
            for (int u = 0; u < 8; ++u)
                for (int v = 0; v < 8; ++v)
                    coeff[u][v] = s[(by + v) * w + bx + u];
            idct8_block_cpu(coeff, blk);
            for (int x = 0; x < 8; ++x)
                for (int y = 0; y < 8; ++y)
                    d[(by + y) * w + bx + x] = blk[x][y];
        }
}

//  GPU kernels: shared-mem separable forward & inverse 8 × 8 DCT
__global__ void dct8x8_forward_shared(const float* __restrict__ src,
                                      float* __restrict__ dst,
                                      int width)
{
    __shared__ float sh[8][8];
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int gx = blockIdx.x * 8 + tx;
    const int gy = blockIdx.y * 8 + ty;

    // Load tile
    sh[ty][tx] = src[gy * width + gx];
    __syncthreads();

    // Pass 1: row DCT
    float row = 0.f;
#pragma unroll
    for (int n = 0; n < 8; ++n)
        row += sh[ty][n] * d_cos[tx * 8 + n];
    row *= alpha(tx);
    sh[ty][tx] = row;                 // write transposed position
    __syncthreads();

    // Pass 2: column DCT
    float col = 0.f;
#pragma unroll
    for (int n = 0; n < 8; ++n)
        col += sh[n][tx] * d_cos[ty * 8 + n];
    col *= alpha(ty);

    dst[gy * width + gx] = col;
}

__global__ void dct8x8_inverse_shared(const float* __restrict__ src,
                                      float* __restrict__ dst,
                                      int width)
{
    __shared__ float sh[8][8];
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int gx = blockIdx.x * 8 + tx;
    const int gy = blockIdx.y * 8 + ty;

    // Load coefficients
    sh[ty][tx] = src[gy * width + gx];
    __syncthreads();

    // Pass 1: column inverse
    float tmp = 0.f;
#pragma unroll
    for (int v = 0; v < 8; ++v)
        tmp += alpha(v) * sh[v][tx] * d_cos[v * 8 + ty];
    sh[ty][tx] = tmp;
    __syncthreads();

    // Pass 2: row inverse
    float pix = 0.f;
#pragma unroll
    for (int u = 0; u < 8; ++u)
        pix += alpha(u) * sh[ty][u] * d_cos[u * 8 + tx];

    dst[gy * width + gx] = pix;
}

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
    // Pre-compute cosine LUT
    for (int k = 0; k < 8; ++k)
        for (int n = 0; n < 8; ++n)
            h_cos[k * 8 + n] = std::cos((2 * n + 1) * k * PI / 16.f);
    CUDA_CHECK(cudaMemcpyToSymbol(d_cos, h_cos, sizeof(h_cos)));

    // Load or generate image
    int width = 1024, height = 1024;
    std::vector<uint8_t> img8;
    if (argc >= 2) {
        if (!readPGM(argv[1], img8, width, height)) {
            std::cerr << "PGM load failed\n"; return 1;
        }
        std::cout << "Loaded " << width << "×" << height << " PGM\n";
    } else {
        img8.resize(size_t(width) * height);
        std::srand(1234);
        for (auto& p : img8) p = uint8_t(std::rand() & 0xFF);
        std::cout << "Generated random " << width << "×" << height << " image\n";
    }
    if (width % 8 || height % 8) { std::cerr << "Dims must be multiples of 8\n"; return 1; }

    std::vector<float> h_src(width * height);
    for (size_t i = 0; i < h_src.size(); ++i) h_src[i] = float(img8[i]);

    // CPU reference
    std::vector<float> h_coeff_cpu(width * height), h_rec_cpu(width * height);
    auto c0 = std::chrono::high_resolution_clock::now();
    forwardDCT_CPU(h_src.data(), h_coeff_cpu.data(), width, height);
    inverseDCT_CPU(h_coeff_cpu.data(), h_rec_cpu.data(), width, height);
    auto c1 = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(c1 - c0).count();
    double psnr_cpu = psnr(h_src, h_rec_cpu);

    // Allocate GPU
    float *d_src, *d_coeff, *d_rec;
    size_t bytes = size_t(width) * height * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_src, bytes));
    CUDA_CHECK(cudaMalloc(&d_coeff, bytes));
    CUDA_CHECK(cudaMalloc(&d_rec, bytes));

    // Events for timing
    cudaEvent_t t0, t1, tH2D0, tH2D1, tK0, tK1, tD2H0, tD2H1;
    for (cudaEvent_t* ev : {&t0,&t1,&tH2D0,&tH2D1,&tK0,&tK1,&tD2H0,&tD2H1})
        CUDA_CHECK(cudaEventCreate(ev));

    CUDA_CHECK(cudaEventRecord(t0));

    // H2D copy
    CUDA_CHECK(cudaEventRecord(tH2D0));
    CUDA_CHECK(cudaMemcpy(d_src, h_src.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(tH2D1));

    // Kernels
    dim3 block(8, 8), grid(width / 8, height / 8);
    CUDA_CHECK(cudaEventRecord(tK0));
    dct8x8_forward_shared<<<grid, block>>>(d_src, d_coeff, width);
    dct8x8_inverse_shared<<<grid, block>>>(d_coeff, d_rec, width);
    CUDA_CHECK(cudaEventRecord(tK1));

    // D2H copy
    std::vector<float> h_rec_gpu(width * height);
    CUDA_CHECK(cudaEventRecord(tD2H0));
    CUDA_CHECK(cudaMemcpy(h_rec_gpu.data(), d_rec, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(tD2H1));

    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));

    float h2d_ms, k_ms, d2h_ms, total_ms;
    CUDA_CHECK(cudaEventElapsedTime(&h2d_ms, tH2D0, tH2D1));
    CUDA_CHECK(cudaEventElapsedTime(&k_ms,   tK0,   tK1));
    CUDA_CHECK(cudaEventElapsedTime(&d2h_ms, tD2H0, tD2H1));
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, t0,  t1));

    double psnr_gpu = psnr(h_src, h_rec_gpu);

    // Timings
    std::cout << "CPU forward+inverse      : " << cpu_ms << " ms\n";
    std::cout << "GPU H2D copy             : " << h2d_ms << " ms\n";
    std::cout << "GPU kernels (Fwd+Inv)    : " << k_ms   << " ms\n";
    std::cout << "GPU D2H copy             : " << d2h_ms << " ms\n";
    std::cout << "GPU total                : " << total_ms << " ms\n";
    std::cout << "PSNR CPU reconstruction  : " << psnr_cpu << " dB\n";
    std::cout << "PSNR GPU reconstruction  : " << psnr_gpu << " dB\n";

    // Clean-up
    CUDA_CHECK(cudaFree(d_src));
    CUDA_CHECK(cudaFree(d_coeff));
    CUDA_CHECK(cudaFree(d_rec));
    return 0;
}