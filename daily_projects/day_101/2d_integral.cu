#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <chrono>
#include <fstream>
#include <iostream>
#include <vector>
#include <string>

#define CUDA_CHECK(call)                                                          \
  do {                                                                            \
    cudaError_t _err = (call);                                                    \
    if (_err != cudaSuccess) {                                                    \
      std::cerr << "CUDA error: " << cudaGetErrorString(_err)                     \
                << " (" << __FILE__ << ':' << __LINE__ << ")\n"; std::exit(1);    \
    }                                                                             \
  } while (0)

template<class TP> inline double ms(TP a, TP b){
  return std::chrono::duration<double,std::milli>(b - a).count(); }

// Minimal PGM loader
bool readPGM(const char* fn, std::vector<uint8_t>& img, int& W, int& H)
{
  std::ifstream f(fn, std::ios::binary); if (!f) return false;
  std::string m; f >> m; if (m != "P5") return false;
  int maxv; f >> W >> H >> maxv; f.get(); if (maxv != 255) return false;
  img.resize(size_t(W) * H);
  f.read(reinterpret_cast<char*>(img.data()), img.size());
  return bool(f);
}

// CPU reference
void integralCPU(const std::vector<uint8_t>& src,
                 std::vector<uint32_t>& dst,
                 int W, int H)
{
  dst.resize(size_t(W) * H);
  for (int y = 0; y < H; ++y) {
    uint32_t row = 0;
    for (int x = 0; x < W; ++x) {
      row += src[y * W + x];
      dst[y * W + x] = row + (y ? dst[(y - 1) * W + x] : 0);
    }
  }
}

struct CastByte { __host__ __device__ uint32_t operator()(uint8_t v) const {
  return static_cast<uint32_t>(v); } };

struct RowKey { int W;
  __host__ __device__ RowKey(int w):W(w){}
  __host__ __device__ int operator()(int i) const { return i / W; } };

struct RowKeyStride { int stride;
  __host__ __device__ RowKeyStride(int s):stride(s){}
  __host__ __device__ int operator()(int i) const { return i / stride; } };

// 32×32 tiled transpose ───────────── */
__global__ void transpose32u(const uint32_t* __restrict__ in,
                             uint32_t* __restrict__ out,
                             int W, int H)
{
  __shared__ uint32_t tile[32][33];
  int x = blockIdx.x * 32 + threadIdx.x;
  int y = blockIdx.y * 32 + threadIdx.y;
  if (x < W && y < H)
    tile[threadIdx.y][threadIdx.x] = in[y * W + x];
  __syncthreads();
  x = blockIdx.y * 32 + threadIdx.x;
  y = blockIdx.x * 32 + threadIdx.y;
  if (x < H && y < W)
    out[y * H + x] = tile[threadIdx.x][threadIdx.y];
}

int main(int argc, char** argv)
{
  int W = 4096, H = 4096;
  std::vector<uint8_t> img8;
  if (argc >= 2) {
    if (!readPGM(argv[1], img8, W, H)) { std::cerr << "PGM load fail\n"; return 1; }
    std::cout << "Loaded " << W << "×" << H << " PGM\n";
  } else {
    img8.resize(size_t(W) * H);
    std::srand(1234);
    for (auto& p : img8) p = std::rand() & 0xFF;
    std::cout << "Random " << W << "×" << H << " image\n";
  }
  const size_t N = size_t(W) * H;

  // CPU reference
  std::vector<uint32_t> iiCPU;
  auto t0 = std::chrono::high_resolution_clock::now();
  integralCPU(img8, iiCPU, W, H);
  auto t1 = std::chrono::high_resolution_clock::now();
  double cpu_ms = ms(t0, t1);

  // GPU buffers
  uint8_t  *d_img;  CUDA_CHECK(cudaMalloc(&d_img,  N));
  uint32_t *d_row;  CUDA_CHECK(cudaMalloc(&d_row,  N * sizeof(uint32_t)));
  uint32_t *d_tr ;  CUDA_CHECK(cudaMalloc(&d_tr ,  N * sizeof(uint32_t)));
  uint32_t *d_int;  CUDA_CHECK(cudaMalloc(&d_int, N * sizeof(uint32_t)));

  // CUDA events
  cudaEvent_t beg, end, h0, h1, k0, k1, d0, d1;
  CUDA_CHECK(cudaEventCreate(&beg));
  CUDA_CHECK(cudaEventCreate(&end));
  CUDA_CHECK(cudaEventCreate(&h0));
  CUDA_CHECK(cudaEventCreate(&h1));
  CUDA_CHECK(cudaEventCreate(&k0));
  CUDA_CHECK(cudaEventCreate(&k1));
  CUDA_CHECK(cudaEventCreate(&d0));
  CUDA_CHECK(cudaEventCreate(&d1));

  CUDA_CHECK(cudaEventRecord(beg));

  // H2D
  CUDA_CHECK(cudaEventRecord(h0));
  CUDA_CHECK(cudaMemcpy(d_img, img8.data(), N, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaEventRecord(h1));

  // Wrappers
  thrust::device_ptr<uint8_t >  in8(d_img);
  thrust::device_ptr<uint32_t> row32(d_row);
  thrust::device_ptr<uint32_t> tr32 (d_tr );
  thrust::device_ptr<uint32_t> out32(d_int);

  CastByte caster;

  CUDA_CHECK(cudaEventRecord(k0));

  // Row scan (single segmented scan)
  auto keys_row = thrust::make_transform_iterator(
                    thrust::counting_iterator<int>(0), RowKey(W));
  auto vals_row = thrust::make_transform_iterator(in8, caster);
  thrust::inclusive_scan_by_key(keys_row, keys_row + N, vals_row, row32);

  // Transpose row-scan result
  dim3 blk(32, 32), grd((W + 31) / 32, (H + 31) / 32);
  transpose32u<<<grd, blk>>>(d_row, d_tr, W, H);

  // Column scan (rows of transposed)
  auto keys_col = thrust::make_transform_iterator(
                    thrust::counting_iterator<int>(0), RowKeyStride(H));
  thrust::inclusive_scan_by_key(keys_col, keys_col + N, tr32, tr32);

  // Transpose back to original layout
  dim3 grd2((H + 31) / 32, (W + 31) / 32);
  transpose32u<<<grd2, blk>>>(d_tr, d_int, H, W);

  CUDA_CHECK(cudaEventRecord(k1));

  // D2H
  std::vector<uint32_t> iiGPU(N);
  CUDA_CHECK(cudaEventRecord(d0));
  CUDA_CHECK(cudaMemcpy(iiGPU.data(), d_int, N * sizeof(uint32_t), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaEventRecord(d1));

  CUDA_CHECK(cudaEventRecord(end));
  CUDA_CHECK(cudaEventSynchronize(end));

  // Timings
  auto evms=[&](cudaEvent_t a,cudaEvent_t b){ float m; CUDA_CHECK(cudaEventElapsedTime(&m,a,b)); return double(m); };
  double h2d = evms(h0, h1);
  double ker = evms(k0, k1);
  double d2h = evms(d0, d1);
  double tot = evms(beg, end);

  // RMSE check
  double mse = 0; for (size_t i = 0; i < N; ++i) { double diff = double(iiGPU[i]) - double(iiCPU[i]); mse += diff * diff; }
  double rmse = std::sqrt(mse / N);

  std::cout << "CPU total               : " << cpu_ms << " ms\n";
  std::cout << "GPU H2D copy            : " << h2d    << " ms\n";
  std::cout << "GPU kernels             : " << ker    << " ms\n";
  std::cout << "GPU D2H copy            : " << d2h    << " ms\n";
  std::cout << "GPU total               : " << tot    << " ms\n";
  std::cout << "RMSE                    : " << rmse   << "\n";

  CUDA_CHECK(cudaFree(d_img));
  CUDA_CHECK(cudaFree(d_row));
  CUDA_CHECK(cudaFree(d_tr ));
  CUDA_CHECK(cudaFree(d_int));
  return 0;
}