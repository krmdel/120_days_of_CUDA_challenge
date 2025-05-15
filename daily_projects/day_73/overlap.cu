#include <cuda_runtime.h>
#include <cufft.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <random>
#include <algorithm>           // std::copy_n

// Error-checking helpers
#define CUDA_CHECK(x)                                                             \
    do {                                                                          \
        cudaError_t rc = (x);                                                     \
        if (rc != cudaSuccess) {                                                  \
            std::cerr << "CUDA error: " << cudaGetErrorString(rc)                 \
                      << " (" << __FILE__ << ":" << __LINE__ << ")\n";            \
            std::exit(EXIT_FAILURE);                                              \
        }                                                                         \
    } while (0)

#define CUFFT_CHECK(x)                                                            \
    do {                                                                          \
        cufftResult rc = (x);                                                     \
        if (rc != CUFFT_SUCCESS) {                                                \
            std::cerr << "cuFFT error code " << rc                                \
                      << " (" << __FILE__ << ":" << __LINE__ << ")\n";            \
            std::exit(EXIT_FAILURE);                                              \
        }                                                                         \
    } while (0)

// Utility: next power of two
inline int next_pow2(int n)
{
    return 1 << static_cast<int>(std::ceil(std::log2(static_cast<float>(n))));
}

// Kernel 1: point-wise complex multiplication (batched)
//   data_freq   : size  (batch × freq_len)
//   kernel_freq : size  (freq_len) – same for every batch element
__global__ void multiply_complex_batch(cufftComplex*       data_freq,
                                       const cufftComplex* kernel_freq,
                                       int                 freq_len,
                                       int                 batch)
{
    int idx   = blockIdx.x * blockDim.x + threadIdx.x;
    int total = freq_len * batch;
    if (idx >= total) return;

    cufftComplex a = data_freq[idx];
    cufftComplex b = kernel_freq[idx % freq_len];   // broadcast kernel spectrum

    data_freq[idx] = make_cuFloatComplex(a.x * b.x - a.y * b.y,
                                         a.x * b.y + a.y * b.x);
}

// Kernel 2: overlap-add accumulation + normalisation
//   blocks   : (batch × N) real output from inverse FFTs
//   out_len  : total output length (signal_len + kernel_len − 1)
//   N        : FFT size
//   L        : hop size (non-overlapping part, == block length without overlap)
__global__ void overlap_add_accumulate(const float* __restrict__ blocks,
                                       float*       __restrict__ out,
                                       int          N,
                                       int          L,
                                       int          batch,
                                       size_t       out_len,
                                       float        inv_N)
{
    int idx   = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * N;
    if (idx >= total) return;

    int b        = idx / N;       // which block
    int within   = idx % N;       // index inside the block
    size_t o_idx = static_cast<size_t>(b) * L + within;
    if (o_idx < out_len) {        // guard against last block spill-over
        atomicAdd(out + o_idx, blocks[idx] * inv_N);
    }
}

// Overlap-Add convolution (all work stays on the GPU)
//  * L   – hop length
void overlap_add_convolve(const std::vector<float>& signal,
                          const std::vector<float>& kernel,
                          std::vector<float>&       out,
                          int                       L = 16384)   // tunable
{
    // Derived sizes
    const int M        = static_cast<int>(kernel.size());
    const int N        = next_pow2(L + M - 1);            // FFT size
    const int hop      = L;
    const int batch    = static_cast<int>((signal.size() + hop - 1) / hop);
    const int freq_len = N / 2 + 1;
    const size_t out_len = signal.size() + M - 1;
    const float inv_N   = 1.0f / static_cast<float>(N);

    // Host staging buffers
    std::vector<float> h_blocks(batch * N, 0.0f);         // zero-padded segments
    for (int b = 0; b < batch; ++b) {
        size_t offset   = static_cast<size_t>(b) * hop;
        size_t copy_len = std::min<size_t>(hop, signal.size() - offset);
        std::copy_n(signal.data() + offset, copy_len,
                    h_blocks.data() + static_cast<size_t>(b) * N);
    }

    std::vector<float> h_kernel_padded(N, 0.0f);
    std::copy(kernel.begin(), kernel.end(), h_kernel_padded.begin());

    // Device buffers
    float*        d_blocks    = nullptr;      // real input / IFFT output
    cufftComplex* d_blocks_F  = nullptr;      // frequency domain blocks
    cufftComplex* d_kernel_F  = nullptr;      // kernel spectrum
    float*        d_output    = nullptr;      // final convolution result
    float*        d_kernel_tmp = nullptr;     // temp real kernel (for FFT)

    CUDA_CHECK(cudaMalloc(&d_blocks,     sizeof(float)        * batch * N));
    CUDA_CHECK(cudaMalloc(&d_blocks_F,   sizeof(cufftComplex) * batch * freq_len));
    CUDA_CHECK(cudaMalloc(&d_kernel_F,   sizeof(cufftComplex) * freq_len));
    CUDA_CHECK(cudaMalloc(&d_output,     sizeof(float)        * out_len));
    CUDA_CHECK(cudaMalloc(&d_kernel_tmp,sizeof(float)         * N));
    CUDA_CHECK(cudaMemset(d_output, 0,   sizeof(float) * out_len));

    CUDA_CHECK(cudaMemcpy(d_blocks,      h_blocks.data(),
                          sizeof(float) * batch * N, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_kernel_tmp,  h_kernel_padded.data(),
                          sizeof(float) * N,          cudaMemcpyHostToDevice));

    // cuFFT plan (batched)
    cufftHandle planR2C, planC2R;
    int rank = 1, n[1] = {N};
    int istride = 1, ostride = 1;
    int inembed[1] = {N};
    int onembed[1] = {freq_len};
    int idist = N,  odist = freq_len;

    CUFFT_CHECK(cufftPlanMany(&planR2C, rank, n,
                              inembed, istride, idist,
                              onembed, ostride, odist,
                              CUFFT_R2C, batch));
    CUFFT_CHECK(cufftPlanMany(&planC2R, rank, n,
                              onembed, ostride, odist,
                              inembed, istride, idist,
                              CUFFT_C2R, batch));

    // Kernel FFT
    CUFFT_CHECK(cufftExecR2C(planR2C, d_kernel_tmp, d_kernel_F));
    cudaFree(d_kernel_tmp);

    // Forward FFT on all blocks
    CUFFT_CHECK(cufftExecR2C(planR2C, d_blocks, d_blocks_F));

    // Point-wise multiply
    {
        int threads = 256;
        int blocks  = (batch * freq_len + threads - 1) / threads;
        multiply_complex_batch<<<blocks, threads>>>(d_blocks_F, d_kernel_F,
                                                    freq_len, batch);
    }

    // Inverse FFT (batched)
    CUFFT_CHECK(cufftExecC2R(planC2R, d_blocks_F, d_blocks));

    // Overlap-add accumulation
    {
        int threads = 256;
        int blocks  = (batch * N + threads - 1) / threads;
        overlap_add_accumulate<<<blocks, threads>>>(d_blocks, d_output,
                                                    N, hop, batch, out_len,
                                                    inv_N);
    }

    // Copy result back
    out.resize(out_len);
    CUDA_CHECK(cudaMemcpy(out.data(), d_output,
                          sizeof(float) * out_len, cudaMemcpyDeviceToHost));

    // Cleanup --------------------------------------
    cufftDestroy(planR2C);
    cufftDestroy(planC2R);
    cudaFree(d_blocks);
    cudaFree(d_blocks_F);
    cudaFree(d_kernel_F);
    cudaFree(d_output);
}

int main()
{
    const size_t SIG_LEN = 1 << 16;   // 65 536 samples
    const int    KER_LEN = 257;
    std::cout << "Signal size: " << SIG_LEN << " \n\n";

    std::vector<float> signal(SIG_LEN), kernel(KER_LEN);
    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(0.f, 1.f);
    for (auto& v : signal) v = dist(rng);
    for (auto& v : kernel) v = dist(rng);

    std::vector<float> result;

    auto t0 = std::chrono::high_resolution_clock::now();
    overlap_add_convolve(signal, kernel, result, 16384);  // L = 16 384
    auto t1 = std::chrono::high_resolution_clock::now();

    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "Optimised Overlap-Add convolution: " << ms << " ms\n";

    return 0;
}