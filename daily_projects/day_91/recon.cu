#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <chrono>
#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <cstring>

#define CUDA_CHECK(call)                                                         \
  do {                                                                           \
    cudaError_t _err = (call);                                                   \
    if (_err != cudaSuccess) {                                                   \
      std::cerr << "CUDA error: " << cudaGetErrorString(_err)                    \
                << " (" << __FILE__ << ':' << __LINE__ << ")\n";                 \
      std::exit(EXIT_FAILURE);                                                   \
    }                                                                            \
  } while (0)

constexpr float PI = 3.14159265358979323846f;

//  Minimal LodePNG
#define LODEPNG_IMPLEMENTATION
#define LODEPNG_COMPILE_CPP
#include <assert.h>
#include <stdint.h>
namespace lodepng {
enum { OK = 0, ERR_MEM = 83, ERR_WRITE = 79 };

static void write32(std::vector<unsigned char>& v, uint32_t x) {
  v.push_back((x >> 24) & 255);
  v.push_back((x >> 16) & 255);
  v.push_back((x >> 8)  & 255);
  v.push_back(x & 255);
}
static uint32_t crc32(const unsigned char* d, size_t l) {
  static uint32_t tbl[256];
  static bool init = false;
  if (!init) {
    for (uint32_t i = 0; i < 256; ++i) {
      uint32_t c = i;
      for (int k = 0; k < 8; ++k) c = (c & 1) ? 0xEDB88320u ^ (c >> 1) : c >> 1;
      tbl[i] = c;
    }
    init = true;
  }
  uint32_t c = 0xFFFFFFFFu;
  for (size_t n = 0; n < l; ++n) c = tbl[(c ^ d[n]) & 255] ^ (c >> 8);
  return c ^ 0xFFFFFFFFu;
}
static uint32_t adler32(const unsigned char* d, size_t l) {
  uint32_t a = 1, b = 0;
  for (size_t i = 0; i < l; ++i) { a = (a + d[i]) % 65521; b = (b + a) % 65521; }
  return (b << 16) | a;
}
// Store-only deflate
static void zlib_nocomp(std::vector<unsigned char>& out,
                        const unsigned char* in, size_t n) {
  out.push_back(0x78); out.push_back(0x01);               // zlib header
  size_t pos = 0, blk = 65535;
  while (pos < n) {
    size_t len = std::min(blk, n - pos);
    out.push_back((pos + len == n) ? 1 : 0);              // BFINAL
    out.push_back(len & 255); out.push_back(len >> 8);
    uint16_t nlen = ~len; out.push_back(nlen & 255); out.push_back(nlen >> 8);
    out.insert(out.end(), in + pos, in + pos + len);
    pos += len;
  }
  write32(out, adler32(in, n));
}
unsigned encode(std::vector<unsigned char>& png,
                const unsigned char* img, unsigned w, unsigned h,
                int /*ct*/, int /*bd*/) {
  const unsigned char sig[8] = {137, 80, 78, 71, 13, 10, 26, 10};
  png.insert(png.end(), sig, sig + 8);

  // IHDR
  std::vector<unsigned char> ih;
  write32(ih, w); write32(ih, h);
  ih.push_back(8);  // bit depth
  ih.push_back(0);  // GREY
  ih.push_back(0); ih.push_back(0); ih.push_back(0);
  write32(png, ih.size());
  png.insert(png.end(), {'I','H','D','R'});
  png.insert(png.end(), ih.begin(), ih.end());
  write32(png, crc32((unsigned char*)"IHDR", 4) ^ crc32(ih.data(), ih.size()));

  // IDAT
  std::vector<unsigned char> raw((w + 1) * h);
  for (unsigned y = 0; y < h; ++y) {
    raw[y * (w + 1)] = 0;  // filter type 0
    std::memcpy(&raw[y * (w + 1) + 1], &img[y * w], w);
  }
  std::vector<unsigned char> z;
  zlib_nocomp(z, raw.data(), raw.size());
  write32(png, z.size());
  png.insert(png.end(), {'I','D','A','T'});
  png.insert(png.end(), z.begin(), z.end());
  write32(png, crc32((unsigned char*)"IDAT", 4) ^ crc32(z.data(), z.size()));

  // IEND
  write32(png, 0); png.insert(png.end(), {'I','E','N','D'});
  write32(png, crc32((unsigned char*)"IEND", 4));
  return OK;
}
unsigned save_file(const std::vector<unsigned char>& buf,
                   const std::string& fn) {
  std::ofstream f(fn, std::ios::binary); if (!f) return ERR_WRITE;
  f.write(reinterpret_cast<const char*>(buf.data()), buf.size());
  return 0;
}
inline unsigned encodeFile(const std::string& fn, const unsigned char* img,
                           unsigned w, unsigned h,
                           int ct = 0, int bd = 8) {
  std::vector<unsigned char> png;
  unsigned err = encode(png, img, w, h, ct, bd);
  if (!err) err = save_file(png, fn);
  return err;
}
}

// Constant tables
__constant__ uint8_t d_zig[64], d_q[64];
__constant__ float   d_c[64];

static uint8_t h_zig[64] = {
  0, 1, 5, 6,14,15,27,28,  2, 4, 7,13,16,26,29,42,
  3, 8,12,17,25,30,41,43,  9,11,18,24,31,40,44,53,
 10,19,23,32,39,45,52,54, 20,22,33,38,46,51,55,60,
 21,34,37,47,50,56,59,61, 35,36,48,49,57,58,62,63 };
static uint8_t h_q[64] = {
 16,11,10,16,24,40,51,61, 12,12,14,19,26,58,60,55,
 14,13,16,24,40,57,69,56, 14,17,22,29,51,87,80,62,
 18,22,37,56,68,109,103,77,24,35,55,64,81,104,113,92,
 49,64,78,87,103,121,120,101,72,92,95,98,112,100,103,99 };
static float  h_c[64];

// PGM reader (8-bit)
bool readPGM(const char* fn, std::vector<uint8_t>& img, int& w, int& h) {
  std::ifstream f(fn, std::ios::binary); if (!f) return false;
  std::string m; f >> m; if (m != "P5") return false;
  int maxv; f >> w >> h >> maxv; f.get();
  if (maxv != 255) return false;
  img.resize(size_t(w) * h);
  f.read(reinterpret_cast<char*>(img.data()), img.size());
  return bool(f);
}

// Helpers
__host__ __device__ inline float a(int k) { return k ? 0.5f : 0.353553390593273762f; }

// Shared-mem DCT kernels
__global__ void fwdDCT(const float* __restrict__ s, float* __restrict__ d, int W) {
  __shared__ float sh[8][8];
  int tx = threadIdx.x, ty = threadIdx.y;
  int gx = blockIdx.x * 8 + tx, gy = blockIdx.y * 8 + ty;
  sh[ty][tx] = s[gy * W + gx]; __syncthreads();

  float row = 0.f;
#pragma unroll
  for (int n = 0; n < 8; ++n) row += sh[ty][n] * d_c[tx * 8 + n];
  row *= a(tx);
  sh[ty][tx] = row; __syncthreads();

  float col = 0.f;
#pragma unroll
  for (int n = 0; n < 8; ++n) col += sh[n][tx] * d_c[ty * 8 + n];
  col *= a(ty);
  d[gy * W + gx] = col;
}
__global__ void invDCT(const float* __restrict__ s, float* __restrict__ d, int W) {
  __shared__ float sh[8][8];
  int tx = threadIdx.x, ty = threadIdx.y;
  int gx = blockIdx.x * 8 + tx, gy = blockIdx.y * 8 + ty;
  sh[ty][tx] = s[gy * W + gx]; __syncthreads();

  float tmp = 0.f;
#pragma unroll
  for (int v = 0; v < 8; ++v) tmp += a(v) * sh[v][tx] * d_c[v * 8 + ty];
  sh[ty][tx] = tmp; __syncthreads();

  float pix = 0.f;
#pragma unroll
  for (int u = 0; u < 8; ++u) pix += a(u) * sh[ty][u] * d_c[u * 8 + tx];
  d[gy * W + gx] = pix;
}

// Zig-zag, Quant, Dequant kernels
__global__ void zigzag(const float* __restrict__ in, float* __restrict__ out,
                       int W, int bpr) {
  int bx = blockIdx.x, by = blockIdx.y;
  int tx = threadIdx.x, ty = threadIdx.y;
  int tile = by * bpr + bx;
  int gx = bx * 8 + tx, gy = by * 8 + ty;
  out[tile * 64 + d_zig[ty * 8 + tx]] = in[gy * W + gx];
}
__global__ void quant(const float* in, int16_t* q, float qf, int N) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id >= N) return;
  q[id] = static_cast<int16_t>(roundf(in[id] / (qf * float(d_q[id % 64]))));
}
__global__ void dequant_invzig(const int16_t* q, float* out, float qf,
                               int W, int bpr) {
  int bx = blockIdx.x, by = blockIdx.y;
  int tx = threadIdx.x, ty = threadIdx.y;
  int tile = by * bpr + bx;
  int gx = bx * 8 + tx, gy = by * 8 + ty;
  out[gy * W + gx] =
      qf * float(d_q[ty * 8 + tx]) * float(q[tile * 64 + d_zig[ty * 8 + tx]]);
}

// MSE kernel
__global__ void mseK(const float* a, const float* b, double* g, int W, int H) {
  __shared__ double s[64];
  int tx = threadIdx.x, ty = threadIdx.y;
  int gx = blockIdx.x * 8 + tx, gy = blockIdx.y * 8 + ty;

  double diff = 0.0;
  if (gx < W && gy < H) {
    double delta = double(a[gy * W + gx]) - double(b[gy * W + gx]);
    diff = delta * delta;
  }
  s[ty * 8 + tx] = diff;
  __syncthreads();

  for (int stride = 32; stride; stride >>= 1) {
    if ((ty * 8 + tx) < stride) s[ty * 8 + tx] += s[ty * 8 + tx + stride];
    __syncthreads();
  }
  if (tx == 0 && ty == 0) atomicAdd(g, s[0]);
}
double psnr(double mse) { return 10.0 * std::log10(255.0 * 255.0 / mse); }

int main(int argc, char** argv) {
  float qf = 1.0f;
  std::string outPNG = "recon.png";
  if (argc >= 3) qf = std::stof(argv[2]);
  if (argc >= 4) outPNG = argv[3];

  // cosine table & upload
  for (int k = 0; k < 8; ++k)
    for (int n = 0; n < 8; ++n)
      h_c[k * 8 + n] = std::cos((2 * n + 1) * k * PI / 16.f);
  CUDA_CHECK(cudaMemcpyToSymbol(d_zig, h_zig, sizeof(h_zig)));
  CUDA_CHECK(cudaMemcpyToSymbol(d_q  , h_q , sizeof(h_q )));
  CUDA_CHECK(cudaMemcpyToSymbol(d_c  , h_c , sizeof(h_c )));

  // load image (or random)
  int W = 1024, H = 1024;
  std::vector<uint8_t> img8;
  if (argc >= 2) {
    if (!readPGM(argv[1], img8, W, H)) { std::cerr << "PGM load failed\n"; return 1; }
  } else {
    img8.resize(size_t(W) * H); std::srand(1234);
    for (auto& p : img8) p = std::rand() & 0xFF;
  }
  std::vector<float> h_src(W * H);
  for (size_t i = 0; i < h_src.size(); ++i) h_src[i] = float(img8[i]);

  // GPU buffers
  float *d_src, *d_coeff, *d_tmp, *d_rec, *d_zz;
  int16_t* d_qv; double* d_err;
  size_t imgB = size_t(W) * H * sizeof(float);
  int bpr = W / 8, bpc = H / 8, totalBlocks = bpr * bpc;
  CUDA_CHECK(cudaMalloc(&d_src , imgB));
  CUDA_CHECK(cudaMalloc(&d_coeff, imgB));
  CUDA_CHECK(cudaMalloc(&d_tmp , imgB));
  CUDA_CHECK(cudaMalloc(&d_rec , imgB));
  CUDA_CHECK(cudaMalloc(&d_zz  , size_t(totalBlocks) * 64 * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_qv  , size_t(totalBlocks) * 64 * sizeof(int16_t)));
  CUDA_CHECK(cudaMalloc(&d_err , sizeof(double)));
  CUDA_CHECK(cudaMemset(d_err, 0, sizeof(double)));

  // Timing
  cudaEvent_t e0, e1; CUDA_CHECK(cudaEventCreate(&e0)); CUDA_CHECK(cudaEventCreate(&e1));
  CUDA_CHECK(cudaEventRecord(e0));

  CUDA_CHECK(cudaMemcpy(d_src, h_src.data(), imgB, cudaMemcpyHostToDevice));

  dim3 block(8, 8), grid(bpr, bpc);
  fwdDCT<<<grid, block>>>(d_src,  d_coeff, W);
  zigzag <<<grid, block>>>(d_coeff, d_zz,  W, bpr);
  quant  <<< (totalBlocks * 64 + 255) / 256, 256 >>>(d_zz, d_qv, qf, totalBlocks * 64);
  dequant_invzig<<<grid, block>>>(d_qv, d_tmp, qf, W, bpr);
  invDCT<<<grid, block>>>(d_tmp, d_rec, W);
  mseK  <<<grid, block>>>(d_src, d_rec, d_err, W, H);

  std::vector<float> h_rec(W * H);
  CUDA_CHECK(cudaMemcpy(h_rec.data(), d_rec, imgB, cudaMemcpyDeviceToHost));
  double mse; CUDA_CHECK(cudaMemcpy(&mse, d_err, sizeof(double), cudaMemcpyDeviceToHost));
  mse /= double(W) * H;

  CUDA_CHECK(cudaEventRecord(e1)); CUDA_CHECK(cudaEventSynchronize(e1));
  float ms; CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));

  std::cout << "GPU total " << ms << " ms,  PSNR " << psnr(mse) << " dB\n";

  // Save PNG
  std::vector<unsigned char> raw(W * H);
  for (size_t i = 0; i < raw.size(); ++i) {
    float v = h_rec[i];
    v = (v < 0.f) ? 0.f : (v > 255.f ? 255.f : v);
    raw[i] = static_cast<unsigned char>(v);
  }
  if (lodepng::encodeFile(outPNG, raw.data(), W, H))
    std::cerr << "PNG write error\n";
  else
    std::cout << "Saved " << outPNG << '\n';

  // Cleanup
  CUDA_CHECK(cudaFree(d_src)); CUDA_CHECK(cudaFree(d_coeff)); CUDA_CHECK(cudaFree(d_tmp));
  CUDA_CHECK(cudaFree(d_rec)); CUDA_CHECK(cudaFree(d_zz)); CUDA_CHECK(cudaFree(d_qv));
  CUDA_CHECK(cudaFree(d_err));
  return 0;
}