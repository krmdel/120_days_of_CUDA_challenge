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

#define CUDA_CHECK(call)                                                            \
  do {                                                                              \
    cudaError_t _e = (call);                                                        \
    if (_e != cudaSuccess) {                                                        \
      std::cerr << "CUDA error: " << cudaGetErrorString(_e)                         \
                << " (" << __FILE__ << ':' << __LINE__ << ")\n"; std::exit(1);      \
    }                                                                               \
  } while (0)

template<typename T>
inline double msec(T a, T b) {
  return std::chrono::duration<double, std::milli>(b - a).count();
}

// Raw reader
bool readRawFloat(const char* fn, std::vector<float>& buf, size_t count) {
  std::ifstream f(fn, std::ios::binary);
  if (!f) return false;
  buf.resize(count);
  f.read(reinterpret_cast<char*>(buf.data()), count * sizeof(float));
  return bool(f);
}

// CPU reference
static void harrisCPU(const std::vector<float>& Ix2,
                      const std::vector<float>& Iy2,
                      const std::vector<float>& Ixy,
                      std::vector<float>& R,
                      float k, int W, int H)
{
  R.resize(size_t(W) * H);
  for (size_t i = 0; i < R.size(); ++i) {
    float a = Ix2[i];
    float b = Ixy[i];
    float c = Iy2[i];
    float det = a * c - b * b;
    float trace = a + c;
    R[i] = det - k * trace * trace;
  }
}

// GPU kernel
__constant__ float d_k;        // Harris k (0.04)

__global__ void harrisKernel(const float* __restrict__ Ix2,
                             const float* __restrict__ Iy2,
                             const float* __restrict__ Ixy,
                             float* __restrict__ R,
                             float* __restrict__ blockMax,
                             int N)
{
  extern __shared__ float sh[];      // dynamic shared mem for reduction
  int tid = threadIdx.x;
  int i   = blockIdx.x * blockDim.x + tid;

  float r = -1e30f;
  if (i < N) {
    float a = Ix2[i];
    float c = Iy2[i];
    float b = Ixy[i];
    float det = a * c - b * b;
    float trace = a + c;
    r = det - d_k * trace * trace;
    R[i] = r;
  }
  sh[tid] = r;
  __syncthreads();

  // reduction: max within block
  for (int stride = blockDim.x / 2; stride; stride >>= 1) {
    if (tid < stride)
      sh[tid] = fmaxf(sh[tid], sh[tid + stride]);
    __syncthreads();
  }
  if (tid == 0) blockMax[blockIdx.x] = sh[0];
}

int main(int argc, char** argv)
{
  // Parameters
  const float kHarris = 0.04f;

  // Load or generate tensors
  int W = 4096, H = 4096;
  std::vector<float> Ix2, Iy2, Ixy;

  if (argc == 6) {
    W = std::stoi(argv[4]); H = std::stoi(argv[5]);
    size_t N = size_t(W) * H;
    if (!readRawFloat(argv[1], Ix2, N) ||
        !readRawFloat(argv[2], Iy2, N) ||
        !readRawFloat(argv[3], Ixy, N)) {
      std::cerr << "Error reading raw tensors\n"; return 1;
    }
    std::cout << "Loaded tensors " << W << "×" << H << '\n';
  } else {
    size_t N = size_t(W) * H;
    Ix2.resize(N); Iy2.resize(N); Ixy.resize(N);
    std::srand(1234);
    for (size_t i = 0; i < N; ++i) {
      float a = float(std::rand() & 0x1FFFF) / 655.0f;      // random positives
      float b = float(std::rand() & 0x1FFFF) / 655.0f;
      float c = float(std::rand() & 0x1FFFF) / 655.0f;
      Ix2[i] = a; Iy2[i] = c; Ixy[i] = b;
    }
    std::cout << "Generated random tensors " << W << "×" << H << '\n';
  }
  const size_t N = size_t(W) * H;

  // CPU reference
  std::vector<float> Rcpu;
  auto c0 = std::chrono::high_resolution_clock::now();
  harrisCPU(Ix2, Iy2, Ixy, Rcpu, kHarris, W, H);
  auto c1 = std::chrono::high_resolution_clock::now();
  double cpu_ms = msec(c0, c1);

  // GPU buffers
  float *d_Ix2, *d_Iy2, *d_Ixy, *d_R, *d_blockMax;
  CUDA_CHECK(cudaMalloc(&d_Ix2, N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_Iy2, N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_Ixy, N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_R,   N * sizeof(float)));

  // Each block writes one max
  int threads = 256;
  int blocks  = (N + threads - 1) / threads;
  CUDA_CHECK(cudaMalloc(&d_blockMax, blocks * sizeof(float)));

  // Events
  cudaEvent_t eBeg, eEnd, eH2D0, eH2D1, eK0, eK1, eD2H0, eD2H1;
  CUDA_CHECK(cudaEventCreate(&eBeg));
  CUDA_CHECK(cudaEventCreate(&eEnd));
  CUDA_CHECK(cudaEventCreate(&eH2D0));
  CUDA_CHECK(cudaEventCreate(&eH2D1));
  CUDA_CHECK(cudaEventCreate(&eK0));
  CUDA_CHECK(cudaEventCreate(&eK1));
  CUDA_CHECK(cudaEventCreate(&eD2H0));
  CUDA_CHECK(cudaEventCreate(&eD2H1));

  CUDA_CHECK(cudaEventRecord(eBeg));

  // H2D
  CUDA_CHECK(cudaEventRecord(eH2D0));
  CUDA_CHECK(cudaMemcpy(d_Ix2, Ix2.data(), N * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_Iy2, Iy2.data(), N * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_Ixy, Ixy.data(), N * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaEventRecord(eH2D1));

  // Upload k
  CUDA_CHECK(cudaMemcpyToSymbol(d_k, &kHarris, sizeof(float)));

  // Kernel
  size_t shBytes = threads * sizeof(float);
  CUDA_CHECK(cudaEventRecord(eK0));
  harrisKernel<<<blocks, threads, shBytes>>>(d_Ix2, d_Iy2, d_Ixy,
                                             d_R, d_blockMax, (int)N);
  CUDA_CHECK(cudaEventRecord(eK1));

  // D2H
  std::vector<float> Rgpu(N);
  CUDA_CHECK(cudaEventRecord(eD2H0));
  CUDA_CHECK(cudaMemcpy(Rgpu.data(), d_R, N * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaEventRecord(eD2H1));

  CUDA_CHECK(cudaEventRecord(eEnd));
  CUDA_CHECK(cudaEventSynchronize(eEnd));

  // Timings
  auto evms = [&](cudaEvent_t a, cudaEvent_t b) {
    float ms; CUDA_CHECK(cudaEventElapsedTime(&ms, a, b)); return double(ms);
  };
  double h2d = evms(eH2D0, eH2D1);
  double ker = evms(eK0 , eK1 );
  double d2h = evms(eD2H0, eD2H1);
  double tot = evms(eBeg , eEnd);

  // RMSE check
  double mse = 0.0;
  for (size_t i = 0; i < N; ++i) {
    double d = double(Rgpu[i]) - double(Rcpu[i]);
    mse += d * d;
  }
  double rmse = std::sqrt(mse / N);

  // Timings
  std::cout << "CPU total               : " << cpu_ms << " ms\n";
  std::cout << "GPU H2D copy            : " << h2d    << " ms\n";
  std::cout << "GPU kernel              : " << ker    << " ms\n";
  std::cout << "GPU D2H copy            : " << d2h    << " ms\n";
  std::cout << "GPU total               : " << tot    << " ms\n";
  std::cout << "RMSE (CPU vs GPU)       : " << rmse   << "\n";

  CUDA_CHECK(cudaFree(d_Ix2));
  CUDA_CHECK(cudaFree(d_Iy2));
  CUDA_CHECK(cudaFree(d_Ixy));
  CUDA_CHECK(cudaFree(d_R));
  CUDA_CHECK(cudaFree(d_blockMax));
  return 0;
}
