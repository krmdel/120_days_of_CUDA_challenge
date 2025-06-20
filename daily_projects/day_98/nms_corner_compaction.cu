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
#include <algorithm>            // ← for std::max_element

// Error-checking macro (no name collisions)
#define CUDA_CHECK(call)                                                        \
  do {                                                                          \
    cudaError_t _err = (call);                                                  \
    if (_err != cudaSuccess) {                                                  \
      std::cerr << "CUDA error: " << cudaGetErrorString(_err)                   \
                << " (" << __FILE__ << ':' << __LINE__ << ")\n";               \
      std::exit(EXIT_FAILURE);                                                  \
    }                                                                           \
  } while (0)

// Chrono → ms helper
template<class TP>
inline double ms(TP a, TP b) {
  return std::chrono::duration<double, std::milli>(b - a).count();
}

// Raw float32 loader
bool readRawFloat(const char* fn, std::vector<float>& buf, size_t n) {
  std::ifstream f(fn, std::ios::binary); if (!f) return false;
  buf.resize(n);
  f.read(reinterpret_cast<char*>(buf.data()), n * sizeof(float));
  return bool(f);
}

// CPU scalar NMS (3×3)
static void cpuNMS(const std::vector<float>& R,
                   std::vector<uint8_t>& flags,
                   int W, int H, float th)
{
  flags.assign(size_t(W) * H, 0);
  auto idx = [&](int x, int y) { return y * W + x; };

  for (int y = 1; y < H - 1; ++y)
    for (int x = 1; x < W - 1; ++x) {
      float v = R[idx(x, y)];
      if (v < th) continue;
      bool isMax = true;
      for (int dy = -1; dy <= 1 && isMax; ++dy)
        for (int dx = -1; dx <= 1; ++dx)
          if (dx || dy) isMax &= (v >= R[idx(x + dx, y + dy)]);
      flags[idx(x, y)] = isMax;
    }
}

// GPU kernels
__global__ void nmsKernel(const float* __restrict__ R,
                          uint8_t* __restrict__ flag,
                          int W, int H, float th)
{
  int x = blockIdx.x * 16 + threadIdx.x;
  int y = blockIdx.y * 16 + threadIdx.y;
  if (x <= 0 || y <= 0 || x >= W - 1 || y >= H - 1) return;

  int i = y * W + x;
  float v = R[i];
  if (v < th) { flag[i] = 0; return; }

  bool isMax = true;
#pragma unroll
  for (int dy = -1; dy <= 1; ++dy)
#pragma unroll
    for (int dx = -1; dx <= 1; ++dx)
      if (dx || dy) isMax &= (v >= R[(y + dy) * W + x + dx]);

  flag[i] = isMax;
}

__global__ void compactKernel(const uint8_t* __restrict__ flag,
                              int*          __restrict__ list,
                              int*          __restrict__ counter,
                              int N)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;
  if (flag[i]) {
    int pos = atomicAdd(counter, 1);
    list[pos] = i;          // store linear index
  }
}

__global__ void flagCopyKernel(const uint8_t* src, uint8_t* dst, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) dst[i] = src[i];
}

int main(int argc, char** argv)
{
  int   W = 4096, H = 4096;
  float frac = 0.01f;                        // threshold = frac × max(R)
  std::vector<float> R;

  if (argc == 5) {                           // R.bin W H frac
    W = std::stoi(argv[2]); H = std::stoi(argv[3]); frac = std::stof(argv[4]);
    if (!readRawFloat(argv[1], R, size_t(W) * H)) { std::cerr << "read fail\n"; return 1; }
    std::cout << "Loaded response map " << W << "×" << H
              << "   (threshold = " << frac * 100 << " % of max)\n";
  } else {
    R.resize(size_t(W) * H);
    std::srand(1234);
    for (auto& v : R) v = float(std::rand()) / RAND_MAX * 1000.f;
    std::cout << "Random response map " << W << "×" << H
              << "   (threshold = " << frac * 100 << " % of max)\n";
  }
  const size_t N = R.size();

  // Determine absolute threshold
  float maxR = *std::max_element(R.begin(), R.end());
  float th   = maxR * frac;

  // CPU reference
  std::vector<uint8_t> flagCPU;
  auto c0 = std::chrono::high_resolution_clock::now();
  cpuNMS(R, flagCPU, W, H, th);
  auto c1 = std::chrono::high_resolution_clock::now();
  double cpu_ms = ms(c0, c1);

  // GPU path
  float   *d_R;
  uint8_t *d_flag;
  uint8_t *d_flagCopy;              // for accuracy check
  int     *d_list;
  int     *d_count;
  CUDA_CHECK(cudaMalloc(&d_R,        N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_flag,     N * sizeof(uint8_t)));
  CUDA_CHECK(cudaMalloc(&d_flagCopy, N * sizeof(uint8_t)));
  CUDA_CHECK(cudaMalloc(&d_list,     N * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_count,    sizeof(int)));

  // CUDA events
  cudaEvent_t eBeg, eEnd,
              eH2D0, eH2D1,
              eNMS0, eNMS1,
              eCmp0, eCmp1,
              eD2H0, eD2H1;
  for (cudaEvent_t* ev : { &eBeg, &eEnd, &eH2D0, &eH2D1,
                           &eNMS0, &eNMS1, &eCmp0, &eCmp1,
                           &eD2H0, &eD2H1 })
    CUDA_CHECK(cudaEventCreate(ev));

  CUDA_CHECK(cudaEventRecord(eBeg));

  // H2D
  CUDA_CHECK(cudaEventRecord(eH2D0));
  CUDA_CHECK(cudaMemcpy(d_R, R.data(), N * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaEventRecord(eH2D1));

  // NMS
  dim3 blk(16, 16), grd((W + 15) / 16, (H + 15) / 16);
  CUDA_CHECK(cudaEventRecord(eNMS0));
  nmsKernel<<<grd, blk>>>(d_R, d_flag, W, H, th);
  CUDA_CHECK(cudaEventRecord(eNMS1));

  // Compaction
  CUDA_CHECK(cudaMemset(d_count, 0, sizeof(int)));
  int threads = 256, blocks = (N + threads - 1) / threads;
  CUDA_CHECK(cudaEventRecord(eCmp0));
  compactKernel<<<blocks, threads>>>(d_flag, d_list, d_count, N);
  CUDA_CHECK(cudaEventRecord(eCmp1));

  // Copy back compact list and flag map for validation
  int h_cnt;
  CUDA_CHECK(cudaEventRecord(eD2H0));
  CUDA_CHECK(cudaMemcpy(&h_cnt, d_count, sizeof(int), cudaMemcpyDeviceToHost));
  std::vector<int> list(h_cnt);
  if (h_cnt)
    CUDA_CHECK(cudaMemcpy(list.data(), d_list, h_cnt * sizeof(int),
                          cudaMemcpyDeviceToHost));
  // Copy flags for error check
  flagCopyKernel<<<blocks, threads>>>(d_flag, d_flagCopy, N);
  std::vector<uint8_t> flagGPU(N);
  CUDA_CHECK(cudaMemcpy(flagGPU.data(), d_flagCopy, N * sizeof(uint8_t),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaEventRecord(eD2H1));

  CUDA_CHECK(cudaEventRecord(eEnd));
  CUDA_CHECK(cudaEventSynchronize(eEnd));

  // Timings
  auto evms = [&](cudaEvent_t a, cudaEvent_t b) {
    float m; CUDA_CHECK(cudaEventElapsedTime(&m, a, b)); return double(m);
  };
  double h2d = evms(eH2D0, eH2D1);
  double nms = evms(eNMS0, eNMS1);
  double cmp = evms(eCmp0, eCmp1);
  double d2h = evms(eD2H0, eD2H1);
  double tot = evms(eBeg , eEnd );

  // Accuracy
  size_t diff = 0; size_t cpuCnt = 0;
  for (size_t i = 0; i < N; ++i) {
    if (flagCPU[i]) ++cpuCnt;
    if (flagCPU[i] != flagGPU[i]) ++diff;
  }
  double errPct = 100.0 * diff / N;

  // Timings
  std::cout << "CPU total (NMS)          : " << cpu_ms << " ms"
            << "   (corners = " << cpuCnt << ")\n";
  std::cout << "GPU H2D copy             : " << h2d   << " ms\n";
  std::cout << "GPU NMS kernel           : " << nms   << " ms\n";
  std::cout << "GPU compaction kernel    : " << cmp   << " ms"
            << "   (corners = " << h_cnt << ")\n";
  std::cout << "GPU D2H copy             : " << d2h   << " ms\n";
  std::cout << "GPU total                : " << tot   << " ms\n";
  std::cout << "Pixel disagreement       : " << errPct << " %\n";

  CUDA_CHECK(cudaFree(d_R)); CUDA_CHECK(cudaFree(d_flag));
  CUDA_CHECK(cudaFree(d_flagCopy));
  CUDA_CHECK(cudaFree(d_list)); CUDA_CHECK(cudaFree(d_count));
  return 0;
}
