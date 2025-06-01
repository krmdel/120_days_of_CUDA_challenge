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

// Constant-memory lookup tables
__constant__ uint8_t  d_zigzag[64];               // row-major-idx -> zigzag-idx
__constant__ uint8_t  d_qtab[64];                 // JPEG luminance Q-matrix
__constant__ float    d_cos[64];                  // cos((2n+1)kπ/16) (k major)

// Host duplicates for CPU path & memcpyToSymbol
static uint8_t  h_zigzag[64] = {                 // ISO/IEC 10918-1 zig-zag
     0,  1,  5,  6, 14, 15, 27, 28,
     2,  4,  7, 13, 16, 26, 29, 42,
     3,  8, 12, 17, 25, 30, 41, 43,
     9, 11, 18, 24, 31, 40, 44, 53,
    10, 19, 23, 32, 39, 45, 52, 54,
    20, 22, 33, 38, 46, 51, 55, 60,
    21, 34, 37, 47, 50, 56, 59, 61,
    35, 36, 48, 49, 57, 58, 62, 63 };
static uint8_t  h_qtab[64] = {                  // standard JPEG luminance
    16, 11, 10, 16, 24,  40,  51,  61,
    12, 12, 14, 19, 26,  58,  60,  55,
    14, 13, 16, 24, 40,  57,  69,  56,
    14, 17, 22, 29, 51,  87,  80,  62,
    18, 22, 37, 56, 68, 109, 103,  77,
    24, 35, 55, 64, 81, 104, 113,  92,
    49, 64, 78, 87,103, 121, 120, 101,
    72, 92, 95, 98,112, 100, 103,  99 };
static float    h_cos[64];

// Utility: minimal PGM I/O
bool readPGM(const char* f, std::vector<uint8_t>& img, int& w, int& h)
{
    std::ifstream ifs(f, std::ios::binary); if (!ifs) return false;
    std::string m; ifs >> m; if (m != "P5") return false;
    int maxv; ifs >> w >> h >> maxv; ifs.get(); if (maxv != 255) return false;
    img.resize(size_t(w) * h); ifs.read(reinterpret_cast<char*>(img.data()), img.size());
    return bool(ifs);
}

// Helpers
__host__ __device__ inline float alpha(int k) { return k ? 0.5f : 0.353553390593273762f; }

// CPU reference DCT
static void dct8_block_cpu(const float in[8][8], float out[8][8])
{
    for (int u = 0; u < 8; ++u)
        for (int v = 0; v < 8; ++v) {
            float sum = 0.f;
            for (int x = 0; x < 8; ++x)
                for (int y = 0; y < 8; ++y)
                    sum += in[x][y] * h_cos[u * 8 + x] * h_cos[v * 8 + y];
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
                           h_cos[u * 8 + x] * h_cos[v * 8 + y];
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

// Shared-memory forward / inverse DCT kernels
__global__ void dct8x8_forward_shared(const float* __restrict__ src,
                                      float* __restrict__ dst, int width)
{
    __shared__ float sh[8][8];
    int tx = threadIdx.x, ty = threadIdx.y;
    int gx = blockIdx.x * 8 + tx, gy = blockIdx.y * 8 + ty;
    sh[ty][tx] = src[gy * width + gx]; __syncthreads();

    float row = 0.f;  // row DCT
#pragma unroll
    for (int n = 0; n < 8; ++n) row += sh[ty][n] * d_cos[tx * 8 + n];
    row *= alpha(tx);
    sh[ty][tx] = row; __syncthreads();

    float col = 0.f;  // col DCT
#pragma unroll
    for (int n = 0; n < 8; ++n) col += sh[n][tx] * d_cos[ty * 8 + n];
    col *= alpha(ty);
    dst[gy * width + gx] = col;
}
__global__ void dct8x8_inverse_shared(const float* __restrict__ src,
                                      float* __restrict__ dst, int width)
{
    __shared__ float sh[8][8];
    int tx = threadIdx.x, ty = threadIdx.y;
    int gx = blockIdx.x * 8 + tx, gy = blockIdx.y * 8 + ty;
    sh[ty][tx] = src[gy * width + gx]; __syncthreads();

    float t = 0.f;     // inverse col
#pragma unroll
    for (int v = 0; v < 8; ++v) t += alpha(v) * sh[v][tx] * d_cos[v * 8 + ty];
    sh[ty][tx] = t; __syncthreads();

    float p = 0.f;     // inverse row
#pragma unroll
    for (int u = 0; u < 8; ++u) p += alpha(u) * sh[ty][u] * d_cos[u * 8 + tx];
    dst[gy * width + gx] = p;
}

// Zig-zag scan  (float  →  float[64])
__global__ void zigzag_kernel(const float* __restrict__ coeff,
                              float* __restrict__ zz,
                              int width, int blocksPerRow)
{
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;           // 0..7
    int gx = bx * 8 + tx, gy = by * 8 + ty;
    int blockIdxLinear = by * blocksPerRow + bx;      // tile id
    int lutIdx = ty * 8 + tx;                         // row-major index 0..63
    uint8_t zzPos = d_zigzag[lutIdx];                 // where to write

    float val = coeff[gy * width + gx];
    zz[blockIdxLinear * 64 + zzPos] = val;
}

// Quantisation  (float  →  int16)
__global__ void quantise_kernel(const float* __restrict__ in,
                                int16_t* __restrict__ out,
                                float qf, int blocks)          // one block = 64 threads
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;  // coefficient id over all tiles
    if (tid >= blocks * 64) return;
    int cidx = tid % 64;
    float scale = qf * static_cast<float>(d_qtab[cidx]);
    out[tid] = static_cast<int16_t>(roundf(in[tid] / scale));
}

// Dequant + inverse zig-zag  (int16 → float[8×8])
__global__ void dequant_invzig_kernel(const int16_t* __restrict__ in,
                                      float* __restrict__ outCoeff,
                                      float qf, int width, int blocksPerRow)
{
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;           // 0..7
    int gx = bx * 8 + tx, gy = by * 8 + ty;
    int tileId = by * blocksPerRow + bx;
    int lutIdx = ty * 8 + tx;                         // row-major 0..63
    uint8_t zzPos = d_zigzag[lutIdx];                 // location in vector

    float scale = qf * static_cast<float>(d_qtab[lutIdx]);
    int16_t qv  = in[tileId * 64 + zzPos];
    outCoeff[gy * width + gx] = scale * static_cast<float>(qv);
}

// PSNR                                                                       *
double psnr(const std::vector<float>& a, const std::vector<float>& b)
{
    double mse = 0.0; for (size_t i = 0; i < a.size(); ++i) { double d = a[i] - b[i]; mse += d * d; }
    mse /= a.size(); return 10.0 * std::log10(255.0 * 255.0 / mse);
}

int main(int argc, char** argv)
{
    // Quality factor
    float qf = 1.0f;                                     // default
    if (argc >= 3) qf = std::stof(argv[2]);

    // Pre-compute tables and upload
    for (int k = 0; k < 8; ++k)
        for (int n = 0; n < 8; ++n)
            h_cos[k * 8 + n] = std::cos((2 * n + 1) * k * PI / 16.f);
    CUDA_CHECK(cudaMemcpyToSymbol(d_zigzag, h_zigzag, sizeof(h_zigzag)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_qtab,   h_qtab,   sizeof(h_qtab)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_cos,    h_cos,    sizeof(h_cos)));

    // Load / generate image
    int W = 1024, H = 1024;
    std::vector<uint8_t> img8;
    if (argc >= 2) {
        if (!readPGM(argv[1], img8, W, H)) { std::cerr << "PGM load failed\n"; return 1; }
        std::cout << "Loaded " << W << "×" << H << " PGM\n";
    } else {
        img8.resize(size_t(W) * H); std::srand(1234);
        for (auto& p : img8) p = uint8_t(std::rand() & 0xFF);
        std::cout << "Generated random " << W << "×" << H << " image\n";
    }
    if (W % 8 || H % 8) { std::cerr << "Image dims must be multiples of 8\n"; return 1; }
    std::vector<float> h_src(W * H); for (size_t i = 0; i < h_src.size(); ++i) h_src[i] = float(img8[i]);

    // CPU reference for PSNR
    std::vector<float> h_coeff_cpu(W * H), h_rec_cpu(W * H);
    auto c0 = std::chrono::high_resolution_clock::now();
    forwardDCT_CPU(h_src.data(), h_coeff_cpu.data(), W, H);
    inverseDCT_CPU(h_coeff_cpu.data(), h_rec_cpu.data(), W, H);
    auto c1 = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double,std::milli>(c1 - c0).count();
    double psnr_cpu = psnr(h_src, h_rec_cpu);

    // Allocate GPU memory
    float   *d_src, *d_coeff, *d_deqCoeff;
    float   *d_zzF;                  // float zig-zag before quant
    int16_t *d_qcoeff;               // quantised
    size_t bytesImg      = size_t(W) * H * sizeof(float);
    int blocksPerRow     = W / 8;
    int blocksPerCol     = H / 8;
    int totalBlocks      = blocksPerRow * blocksPerCol;
    size_t bytesBlockF   = size_t(totalBlocks) * 64 * sizeof(float);
    size_t bytesBlockI16 = size_t(totalBlocks) * 64 * sizeof(int16_t);

    CUDA_CHECK(cudaMalloc(&d_src,      bytesImg));
    CUDA_CHECK(cudaMalloc(&d_coeff,    bytesImg));
    CUDA_CHECK(cudaMalloc(&d_deqCoeff, bytesImg));
    CUDA_CHECK(cudaMalloc(&d_zzF,      bytesBlockF));
    CUDA_CHECK(cudaMalloc(&d_qcoeff,   bytesBlockI16));

    // Events for timing
    cudaEvent_t evH2D0, evH2D1, evFwd0, evFwd1, evZZ0, evZZ1, evQ0, evQ1, evDQ0, evDQ1, evInv0, evInv1, evD2H0, evD2H1, evTot0, evTot1;
    for (cudaEvent_t* e : {&evH2D0,&evH2D1,&evFwd0,&evFwd1,&evZZ0,&evZZ1,&evQ0,&evQ1,&evDQ0,&evDQ1,&evInv0,&evInv1,&evD2H0,&evD2H1,&evTot0,&evTot1})
        CUDA_CHECK(cudaEventCreate(e));

    CUDA_CHECK(cudaEventRecord(evTot0));

    // H2D copy
    CUDA_CHECK(cudaEventRecord(evH2D0));
    CUDA_CHECK(cudaMemcpy(d_src, h_src.data(), bytesImg, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(evH2D1));

    dim3 block(8,8), grid(blocksPerRow, blocksPerCol);

    // Forward DCT
    CUDA_CHECK(cudaEventRecord(evFwd0));
    dct8x8_forward_shared<<<grid, block>>>(d_src, d_coeff, W);
    CUDA_CHECK(cudaEventRecord(evFwd1));

    // Zig-zag
    CUDA_CHECK(cudaEventRecord(evZZ0));
    zigzag_kernel<<<grid, block>>>(d_coeff, d_zzF, W, blocksPerRow);
    CUDA_CHECK(cudaEventRecord(evZZ1));

    // Quantisation
    CUDA_CHECK(cudaEventRecord(evQ0));
    quantise_kernel<<<(totalBlocks*64 + 255)/256, 256>>>(d_zzF, d_qcoeff, qf, totalBlocks);
    CUDA_CHECK(cudaEventRecord(evQ1));

    // Dequant + inverse zig-zag
    CUDA_CHECK(cudaEventRecord(evDQ0));
    dequant_invzig_kernel<<<grid, block>>>(d_qcoeff, d_deqCoeff, qf, W, blocksPerRow);
    CUDA_CHECK(cudaEventRecord(evDQ1));

    // Inverse DCT
    CUDA_CHECK(cudaEventRecord(evInv0));
    dct8x8_inverse_shared<<<grid, block>>>(d_deqCoeff, d_coeff, W); // reuse d_coeff for recon
    CUDA_CHECK(cudaEventRecord(evInv1));

    // D2H copy
    std::vector<float> h_rec_gpu(W * H);
    CUDA_CHECK(cudaEventRecord(evD2H0));
    CUDA_CHECK(cudaMemcpy(h_rec_gpu.data(), d_coeff, bytesImg, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(evD2H1));

    CUDA_CHECK(cudaEventRecord(evTot1));
    CUDA_CHECK(cudaEventSynchronize(evTot1));

    // Timing
    auto et=[&](cudaEvent_t a, cudaEvent_t b){ float ms; CUDA_CHECK(cudaEventElapsedTime(&ms,a,b)); return ms; };
    float h2d  = et(evH2D0, evH2D1);
    float fwd  = et(evFwd0, evFwd1);
    float zz   = et(evZZ0,  evZZ1);
    float qk   = et(evQ0,   evQ1);
    float dquz = et(evDQ0,  evDQ1);
    float invd = et(evInv0, evInv1);
    float d2h  = et(evD2H0, evD2H1);
    float tot  = et(evTot0, evTot1);

    double psnr_gpu = psnr(h_src, h_rec_gpu);

    // Report
    std::cout << "\nqf=" << qf << "\n";
    std::cout << "CPU forward+inverse        : " << cpu_ms << " ms\n";
    std::cout << "GPU H2D copy               : " << h2d  << " ms\n";
    std::cout << "GPU Forward DCT            : " << fwd  << " ms\n";
    std::cout << "GPU Zig-zag                : " << zz   << " ms\n";
    std::cout << "GPU Quantisation           : " << qk   << " ms\n";
    std::cout << "GPU Dequant+Inv-zig-zag    : " << dquz << " ms\n";
    std::cout << "GPU Inverse DCT            : " << invd << " ms\n";
    std::cout << "GPU D2H copy               : " << d2h  << " ms\n";
    std::cout << "GPU total                  : " << tot  << " ms\n";
    std::cout << "PSNR CPU reconstruction    : " << psnr_cpu << " dB\n";
    std::cout << "PSNR GPU reconstruction    : " << psnr_gpu << " dB\n";

    // Cleanup
    CUDA_CHECK(cudaFree(d_src));
    CUDA_CHECK(cudaFree(d_coeff));
    CUDA_CHECK(cudaFree(d_deqCoeff));
    CUDA_CHECK(cudaFree(d_zzF));
    CUDA_CHECK(cudaFree(d_qcoeff));
    return 0;
}