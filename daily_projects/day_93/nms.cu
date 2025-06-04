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

#define CUDA_CHECK(call)                                                           \
  do {                                                                             \
    cudaError_t err_ = (call);                                                     \
    if (err_ != cudaSuccess) {                                                     \
      std::cerr << "CUDA error: " << cudaGetErrorString(err_)                      \
                << " (" << __FILE__ << ':' << __LINE__ << ")\n";                   \
      std::exit(EXIT_FAILURE);                                                     \
    }                                                                              \
  } while (0)

template<typename TP>
inline double msec(TP a, TP b) {
  return std::chrono::duration<double, std::milli>(b - a).count();
}

// CPU reference NMS
static void nmsCPU(const std::vector<float>& mag,
                   const std::vector<uint8_t>& ang,
                   std::vector<float>& out,
                   int W, int H)
{
  out.assign(W * H, 0.f);
  for (int y = 1; y < H - 1; ++y)
    for (int x = 1; x < W - 1; ++x) {
      float m = mag[y * W + x];
      uint8_t d = ang[y * W + x];
      float n1, n2;
      switch (d) {                       // 0°, 45°, 90°, 135°
        case 0:  n1 = mag[y * W + x - 1];   n2 = mag[y * W + x + 1];     break;
        case 1:  n1 = mag[(y - 1) * W + x - 1]; n2 = mag[(y + 1) * W + x + 1]; break;
        case 2:  n1 = mag[(y - 1) * W + x]; n2 = mag[(y + 1) * W + x];   break;
        default: n1 = mag[(y - 1) * W + x + 1]; n2 = mag[(y + 1) * W + x - 1]; break;
      }
      out[y * W + x] = (m >= n1 && m >= n2) ? m : 0.f;
    }
}

// GPU NMS kernel
__global__ void nmsKernel(const float* __restrict__ mag,
                          const uint8_t* __restrict__ ang,
                          float* __restrict__ out,
                          int W, int H)
{
  __shared__ float sh[18][18];            // 16×16 tile + 1-pixel halo
  int tx = threadIdx.x, ty = threadIdx.y;
  int gx = blockIdx.x * 16 + tx;
  int gy = blockIdx.y * 16 + ty;
  int lx = tx + 1, ly = ty + 1;

  if (gx < W && gy < H) sh[ly][lx] = mag[gy * W + gx];

  // Halo loads
  if (tx == 0  && gx > 0  )                sh[ly][0 ]  = mag[gy * W + gx - 1];
  if (tx == 15 && gx + 1 < W)              sh[ly][17]  = mag[gy * W + gx + 1];
  if (ty == 0  && gy > 0  )                sh[0 ][lx]  = mag[(gy - 1) * W + gx];
  if (ty == 15 && gy + 1 < H)              sh[17][lx]  = mag[(gy + 1) * W + gx];
  if (tx == 0  && ty == 0  && gx > 0  && gy > 0 )
    sh[0 ][0 ]  = mag[(gy - 1) * W + gx - 1];
  if (tx == 15 && ty == 0  && gx + 1 < W && gy > 0 )
    sh[0 ][17]  = mag[(gy - 1) * W + gx + 1];
  if (tx == 0  && ty == 15 && gx > 0  && gy + 1 < H)
    sh[17][0 ]  = mag[(gy + 1) * W + gx - 1];
  if (tx == 15 && ty == 15 && gx + 1 < W && gy + 1 < H)
    sh[17][17] = mag[(gy + 1) * W + gx + 1];

  __syncthreads();

  if (gx == 0 || gy == 0 || gx == W - 1 || gy == H - 1) {
    if (gx < W && gy < H) out[gy * W + gx] = 0.f;
    return;
  }

  float m = sh[ly][lx];
  uint8_t d = ang[gy * W + gx];
  float n1, n2;
  switch (d) {
    case 0:  n1 = sh[ly][lx - 1];   n2 = sh[ly][lx + 1];   break;
    case 1:  n1 = sh[ly - 1][lx - 1]; n2 = sh[ly + 1][lx + 1]; break;
    case 2:  n1 = sh[ly - 1][lx];   n2 = sh[ly + 1][lx];   break;
    default: n1 = sh[ly - 1][lx + 1]; n2 = sh[ly + 1][lx - 1]; break;
  }
  out[gy * W + gx] = (m >= n1 && m >= n2) ? m : 0.f;
}

template<typename T>
bool readRaw(const char* fn, std::vector<T>& buf, size_t count)
{
  std::ifstream f(fn, std::ios::binary);
  if (!f) return false;
  buf.resize(count);
  f.read(reinterpret_cast<char*>(buf.data()), count * sizeof(T));
  return bool(f);
}

int main(int argc, char** argv)
{
  int W = 4096, H = 4096;
  std::vector<float>   mag;
  std::vector<uint8_t> ang;

  if (argc == 5) {
    W = std::stoi(argv[3]); H = std::stoi(argv[4]);
    if (!readRaw<float>(argv[1], mag, size_t(W) * H) ||
        !readRaw<uint8_t>(argv[2], ang, size_t(W) * H)) {
      std::cerr << "Error reading raw input buffers\n";
      return 1;
    }
    std::cout << "Loaded raw magnitude + angle (" << W << "×" << H << ")\n";
  } else {
    /* random buffers for benchmark */
    mag.resize(size_t(W) * H);
    ang.resize(size_t(W) * H);
    std::srand(1234);
    for (size_t i = 0; i < mag.size(); ++i) {
      mag[i] = static_cast<float>(std::rand() & 0x3FF);   // 0–1023
      ang[i] = static_cast<uint8_t>(std::rand() & 3);     // 0–3
    }
    std::cout << "Generated random test data (" << W << "×" << H << ")\n";
  }

  // CPU reference NMS
  std::vector<float> nmsCPUout;
  auto c0 = std::chrono::high_resolution_clock::now();
  nmsCPU(mag, ang, nmsCPUout, W, H);
  auto c1 = std::chrono::high_resolution_clock::now();
  double cpu_ms = msec(c0, c1);

  // GPU setup
  float *d_mag, *d_nms; uint8_t *d_ang;
  size_t magB = size_t(W) * H * sizeof(float);
  size_t angB = size_t(W) * H * sizeof(uint8_t);
  CUDA_CHECK(cudaMalloc(&d_mag, magB));
  CUDA_CHECK(cudaMalloc(&d_ang, angB));
  CUDA_CHECK(cudaMalloc(&d_nms, magB));

  cudaEvent_t ev0, ev1, h2d0, h2d1, k0, k1, d2h0, d2h1;
  CUDA_CHECK(cudaEventCreate(&ev0));
  CUDA_CHECK(cudaEventCreate(&ev1));
  CUDA_CHECK(cudaEventCreate(&h2d0));
  CUDA_CHECK(cudaEventCreate(&h2d1));
  CUDA_CHECK(cudaEventCreate(&k0));
  CUDA_CHECK(cudaEventCreate(&k1));
  CUDA_CHECK(cudaEventCreate(&d2h0));
  CUDA_CHECK(cudaEventCreate(&d2h1));

  CUDA_CHECK(cudaEventRecord(ev0));

  // H2D
  CUDA_CHECK(cudaEventRecord(h2d0));
  CUDA_CHECK(cudaMemcpy(d_mag, mag.data(), magB, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_ang, ang.data(), angB, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaEventRecord(h2d1));

  // Kernel launch
  dim3 block(16, 16);
  dim3 grid((W + 15) / 16, (H + 15) / 16);
  CUDA_CHECK(cudaEventRecord(k0));
  nmsKernel<<<grid, block>>>(d_mag, d_ang, d_nms, W, H);
  CUDA_CHECK(cudaEventRecord(k1));

  // D2H
  std::vector<float> nmsGPU(W * H);
  CUDA_CHECK(cudaEventRecord(d2h0));
  CUDA_CHECK(cudaMemcpy(nmsGPU.data(), d_nms, magB, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaEventRecord(d2h1));

  CUDA_CHECK(cudaEventRecord(ev1));
  CUDA_CHECK(cudaEventSynchronize(ev1));

  // Timing
  auto evms = [&](cudaEvent_t a, cudaEvent_t b) {
    float m; CUDA_CHECK(cudaEventElapsedTime(&m, a, b)); return double(m);
  };
  double h2d = evms(h2d0, h2d1);
  double ker = evms(k0, k1);
  double d2h = evms(d2h0, d2h1);
  double tot = evms(ev0, ev1);

  // Accuracy
  double mse = 0.0;
  for (size_t i = 0; i < nmsGPU.size(); ++i) {
    double d = nmsGPU[i] - nmsCPUout[i]; mse += d * d;
  }
  double rmse = std::sqrt(mse / nmsGPU.size());

  // Report
  std::cout << "CPU NMS total           : " << cpu_ms << " ms\n";
  std::cout << "GPU H2D copy            : " << h2d    << " ms\n";
  std::cout << "GPU NMS kernel          : " << ker    << " ms\n";
  std::cout << "GPU D2H copy            : " << d2h    << " ms\n";
  std::cout << "GPU total               : " << tot    << " ms\n";
  std::cout << "RMSE (CPU vs GPU)       : " << rmse   << "\n";

  CUDA_CHECK(cudaFree(d_mag));
  CUDA_CHECK(cudaFree(d_ang));
  CUDA_CHECK(cudaFree(d_nms));
  return 0;
}
