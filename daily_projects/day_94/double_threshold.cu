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

#define CUDA_CHECK(call)                                                          \
  do {                                                                            \
    cudaError_t _e = (call);                                                      \
    if (_e != cudaSuccess) {                                                      \
      std::cerr << "CUDA error: " << cudaGetErrorString(_e)                       \
                << " (" << __FILE__ << ':' << __LINE__ << ")\n"; std::exit(1);    \
    }                                                                             \
  } while (0)

template<typename TP>
inline double msec(TP a, TP b) {
  return std::chrono::duration<double, std::milli>(b - a).count();
}

// Raw float32 helper
bool readRawFloat(const char* fn, std::vector<float>& buf, size_t count)
{
  std::ifstream f(fn, std::ios::binary);
  if (!f) return false;
  buf.resize(count);
  f.read(reinterpret_cast<char*>(buf.data()), count * sizeof(float));
  return bool(f);
}

// CPU reference implementation
static void hysteresisCPU(const std::vector<float>& mag,
                          std::vector<uint8_t>& out,
                          float highT, float lowT,
                          int W, int H)
{
  out.assign(W * H, 0);
  // Double threshold
  for (size_t i = 0; i < mag.size(); ++i)
    out[i] = (mag[i] >= highT) ? 2 : (mag[i] >= lowT) ? 1 : 0;

  // Hysteresis
  bool changed = true;
  auto id = [&](int x, int y) { return y * W + x; };
  while (changed) {
    changed = false;
    for (int y = 1; y < H - 1; ++y)
      for (int x = 1; x < W - 1; ++x)
        if (out[id(x, y)] == 1) {
          for (int dy = -1; dy <= 1 && out[id(x, y)] == 1; ++dy)
            for (int dx = -1; dx <= 1; ++dx)
              if (out[id(x + dx, y + dy)] == 2) {
                out[id(x, y)] = 2; changed = true; break;
              }
        }
  }
  // Final binary edges
  for (auto& v : out) v = (v == 2) ? 1 : 0;
}


// Classify: 0/1/2
__global__ void classifyKernel(const float* mag, uint8_t* lab,
                               float highT, float lowT, int N)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;
  float m = mag[i];
  lab[i] = (m >= highT) ? 2 : (m >= lowT) ? 1 : 0;
}

// One hysteresis sweep
__global__ void hystIterKernel(const uint8_t* in, uint8_t* out,
                               int W, int H, int* d_flag)
{
  int x = blockIdx.x * 16 + threadIdx.x;
  int y = blockIdx.y * 16 + threadIdx.y;
  if (x <= 0 || y <= 0 || x >= W - 1 || y >= H - 1) return;

  int idx = y * W + x;
  uint8_t v = in[idx];
  if (v != 1) { out[idx] = v; return; }

  bool strong = false;
  for (int dy = -1; dy <= 1 && !strong; ++dy)
    for (int dx = -1; dx <= 1 && !strong; ++dx)
      strong = (in[idx + dy * W + dx] == 2);

  if (strong) { out[idx] = 2; atomicAdd(d_flag, 1); }
  else        { out[idx] = 1; }
}

__global__ void convertStrongToBinary(uint8_t* buf, int N)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) buf[i] = (buf[i] == 2) ? 1 : 0;
}

int main(int argc, char** argv)
{
  // Parameters
  const float kHigh = 1.2f;   // high threshold = kHigh * mean
  const float kLowFactor = 0.4f;

  // Load or make magnitude
  int W = 4096, H = 4096;
  std::vector<float> mag;
  if (argc == 4) {
    W = std::stoi(argv[2]); H = std::stoi(argv[3]);
    if (!readRawFloat(argv[1], mag, size_t(W) * H)) {
      std::cerr << "Cannot read raw magnitude\n"; return 1;
    }
    std::cout << "Loaded magnitude " << W << "×" << H << '\n';
  } else {
    mag.resize(size_t(W) * H);
    std::srand(1234);
    for (auto& m : mag) m = float(std::rand() & 0x3FF);
    std::cout << "Generated random magnitude " << W << "×" << H << '\n';
  }
  size_t N = mag.size();

  // Thresholds
  double sum = 0; for (float v : mag) sum += v;
  float mean  = float(sum / N);
  float highT = kHigh * mean;
  float lowT  = kLowFactor * highT;

  // CPU reference
  std::vector<uint8_t> edgeCPU;
  auto t0 = std::chrono::high_resolution_clock::now();
  hysteresisCPU(mag, edgeCPU, highT, lowT, W, H);
  auto t1 = std::chrono::high_resolution_clock::now();
  double cpu_ms = msec(t0, t1);

  // GPU path
  float*    d_mag;
  uint8_t  *d_labA, *d_labB;
  int*      d_changed;
  CUDA_CHECK(cudaMalloc(&d_mag, N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_labA, N * sizeof(uint8_t)));
  CUDA_CHECK(cudaMalloc(&d_labB, N * sizeof(uint8_t)));
  CUDA_CHECK(cudaMalloc(&d_changed, sizeof(int)));

  // Events
  cudaEvent_t eBeg, eEnd, evH2D0, evH2D1, evCls0, evCls1,
              evHys0, evHys1, evD2H0, evD2H1;
  CUDA_CHECK(cudaEventCreate(&eBeg));
  CUDA_CHECK(cudaEventCreate(&eEnd));
  CUDA_CHECK(cudaEventCreate(&evH2D0));
  CUDA_CHECK(cudaEventCreate(&evH2D1));
  CUDA_CHECK(cudaEventCreate(&evCls0));
  CUDA_CHECK(cudaEventCreate(&evCls1));
  CUDA_CHECK(cudaEventCreate(&evHys0));
  CUDA_CHECK(cudaEventCreate(&evHys1));
  CUDA_CHECK(cudaEventCreate(&evD2H0));
  CUDA_CHECK(cudaEventCreate(&evD2H1));

  CUDA_CHECK(cudaEventRecord(eBeg));

  // H2D
  CUDA_CHECK(cudaEventRecord(evH2D0));
  CUDA_CHECK(cudaMemcpy(d_mag, mag.data(), N * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaEventRecord(evH2D1));

  // Classification
  int tpb = 256, bpg = (N + tpb - 1) / tpb;
  CUDA_CHECK(cudaEventRecord(evCls0));
  classifyKernel<<<bpg, tpb>>>(d_mag, d_labA, highT, lowT, N);
  CUDA_CHECK(cudaEventRecord(evCls1));

  // Iterative hysteresis
  dim3 blk(16, 16), grd((W + 15) / 16, (H + 15) / 16);
  uint8_t* cur = d_labA;
  uint8_t* nxt = d_labB;
  int iterations = 0;
  CUDA_CHECK(cudaEventRecord(evHys0));
  while (true) {
    CUDA_CHECK(cudaMemset(d_changed, 0, sizeof(int)));
    hystIterKernel<<<grd, blk>>>(cur, nxt, W, H, d_changed);
    iterations++;

    int h_flag;
    CUDA_CHECK(cudaMemcpy(&h_flag, d_changed, sizeof(int), cudaMemcpyDeviceToHost));
    if (h_flag == 0) {
      cur = nxt;
      break;
    }
    std::swap(cur, nxt);
  }
  CUDA_CHECK(cudaEventRecord(evHys1));

  // Convert 2 → 1
  bpg = (N + tpb - 1) / tpb;
  convertStrongToBinary<<<bpg, tpb>>>(cur, N);

  // D2H
  std::vector<uint8_t> edgeGPU(N);
  CUDA_CHECK(cudaEventRecord(evD2H0));
  CUDA_CHECK(cudaMemcpy(edgeGPU.data(), cur, N * sizeof(uint8_t), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaEventRecord(evD2H1));

  CUDA_CHECK(cudaEventRecord(eEnd));
  CUDA_CHECK(cudaEventSynchronize(eEnd));

  // Timings
  auto evms = [&](cudaEvent_t a, cudaEvent_t b) {
    float m; CUDA_CHECK(cudaEventElapsedTime(&m, a, b)); return double(m);
  };
  double h2d = evms(evH2D0, evH2D1);
  double cls = evms(evCls0, evCls1);
  double hys = evms(evHys0, evHys1);
  double d2h = evms(evD2H0, evD2H1);
  double tot = evms(eBeg , eEnd );

  // Accuracy
  size_t diff = 0;
  for (size_t i = 0; i < N; ++i)
    if (edgeGPU[i] != edgeCPU[i]) ++diff;
  double errPct = 100.0 * diff / N;

  // Report
  std::cout << "CPU total                 : " << cpu_ms << " ms\n";
  std::cout << "GPU H2D copy              : " << h2d    << " ms\n";
  std::cout << "GPU classify kernel       : " << cls    << " ms\n";
  std::cout << "GPU hysteresis kernels    : " << hys    << " ms   (iterations = "
            << iterations << ")\n";
  std::cout << "GPU D2H copy              : " << d2h    << " ms\n";
  std::cout << "GPU total                 : " << tot    << " ms\n";
  std::cout << "Pixel disagreement        : " << errPct << " %\n";

  CUDA_CHECK(cudaFree(d_mag));
  CUDA_CHECK(cudaFree(d_labA));
  CUDA_CHECK(cudaFree(d_labB));
  CUDA_CHECK(cudaFree(d_changed));
  return 0;
}
