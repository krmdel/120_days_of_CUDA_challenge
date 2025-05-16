#include <cuda_runtime.h>
#include <cufft.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <random>
#include <algorithm>

// Error helpers
#define CUDA_CHECK(x)                                                            \
    do {                                                                         \
        cudaError_t rc = (x);                                                    \
        if (rc != cudaSuccess) {                                                 \
            std::cerr << "CUDA error: " << cudaGetErrorString(rc)                \
                      << " (" << __FILE__ << ":" << __LINE__ << ")\n";           \
            std::exit(EXIT_FAILURE);                                             \
        }                                                                        \
    } while (0)

#define CUFFT_CHECK(x)                                                           \
    do {                                                                         \
        cufftResult rc = (x);                                                    \
        if (rc != CUFFT_SUCCESS) {                                               \
            std::cerr << "cuFFT error code " << rc                                \
                      << " (" << __FILE__ << ":" << __LINE__ << ")\n";           \
            std::exit(EXIT_FAILURE);                                             \
        }                                                                        \
    } while (0)

// Constants
constexpr int  FFT_LEN   = 1024;         // window / FFT size
constexpr int  HOP       = 256;          // hop size (75% overlap)
constexpr int  FREQ_BINS = FFT_LEN / 2 + 1;
constexpr int  STREAMS   = 2;            // double-buffer
constexpr int  CHUNK_FR  = 512;          // frames processed per stream iteration

// Device constant Hann window
__constant__ float d_window[FFT_LEN];

// Kernel 1: windowing
__global__ void apply_window(float* frames, int frame_len, int frames_total)
{
    int tid  = blockIdx.x * blockDim.x + threadIdx.x;
    int size = frames_total * frame_len;
    if (tid >= size) return;

    int idx_in_frame = tid % frame_len;
    frames[tid] *= d_window[idx_in_frame];
}

// Kernel 2: magnitude²
__global__ void power_kernel(const cufftComplex* spectr,
                             float*              power,
                             int                 freq_bins,
                             int                 frames)
{
    int tid  = blockIdx.x * blockDim.x + threadIdx.x;
    int size = frames * freq_bins;
    if (tid >= size) return;

    cufftComplex v = spectr[tid];
    power[tid] = v.x * v.x + v.y * v.y;
}

// Spectrogram pipeline (host)
void compute_spectrogram(const std::vector<float>& signal,
                         std::vector<float>&       spec_out)
{
    // 1) Frame breakdown
    const size_t frames =
        (signal.size() < FFT_LEN) ? 0
        : 1 + (signal.size() - FFT_LEN) / HOP;
    if (frames == 0) {
        std::cerr << "Signal too short.\n";
        return;
    }
    spec_out.resize(frames * FREQ_BINS);

    // 2) Pre-compute Hann window on host and upload to constant memory
    std::vector<float> h_win(FFT_LEN);
    for (int n = 0; n < FFT_LEN; ++n)
        h_win[n] = 0.5f - 0.5f * std::cos(2.f * M_PI * n / (FFT_LEN - 1));
    CUDA_CHECK(cudaMemcpyToSymbol(d_window, h_win.data(),
                                  sizeof(float) * FFT_LEN));

    // 3) cuFFT batched plan (created once per stream)
    cufftHandle plan[STREAMS];
    for (int s = 0; s < STREAMS; ++s)
        CUFFT_CHECK(cufftPlan1d(&plan[s], FFT_LEN, CUFFT_R2C, CHUNK_FR));

    // 4) Allocate double-buffered device workspaces per stream
    float*        d_frames [STREAMS];
    cufftComplex* d_spectr [STREAMS];
    float*        d_power  [STREAMS];
    for (int s = 0; s < STREAMS; ++s) {
        CUDA_CHECK(cudaMalloc(&d_frames[s],
                   sizeof(float) * CHUNK_FR * FFT_LEN));
        CUDA_CHECK(cudaMalloc(&d_spectr[s],
                   sizeof(cufftComplex) * CHUNK_FR * FREQ_BINS));
        CUDA_CHECK(cudaMalloc(&d_power[s],
                   sizeof(float) * CHUNK_FR * FREQ_BINS));
        cufftSetStream(plan[s], nullptr);   // each stream set later in loop
    }

    // 5) Create CUDA streams
    cudaStream_t streams[STREAMS];
    for (int s = 0; s < STREAMS; ++s)
        CUDA_CHECK(cudaStreamCreate(&streams[s]));

    // 6) Host pointers to frame data (strided view)
    std::vector<float> host_frames(frames * FFT_LEN);
    for (size_t f = 0; f < frames; ++f)
        std::copy_n(signal.data() + f * HOP, FFT_LEN,
                    host_frames.data() + f * FFT_LEN);

    // 7) Process in chunks with double-buffer streams
    size_t frames_done = 0;
    while (frames_done < frames) {
        int s         = frames_done / CHUNK_FR % STREAMS;   // stream index
        size_t remain = frames - frames_done;
        int batch     = static_cast<int>(std::min<size_t>(CHUNK_FR, remain));

        // Async copy host→device (frames)
        CUDA_CHECK(cudaMemcpyAsync(d_frames[s],
                   host_frames.data() + frames_done * FFT_LEN,
                   sizeof(float) * batch * FFT_LEN,
                   cudaMemcpyHostToDevice, streams[s]));

        // Chain: window
        {
            int threads = 256;
            int blocks  = (batch * FFT_LEN + threads - 1) / threads;
            apply_window<<<blocks, threads, 0, streams[s]>>>
                (d_frames[s], FFT_LEN, batch);
        }

        // FFT (batched)
        cufftSetStream(plan[s], streams[s]);
        CUFFT_CHECK(cufftExecR2C(plan[s], d_frames[s], d_spectr[s]));

        // Magnitude²
        {
            int threads = 256;
            int blocks  = (batch * FREQ_BINS + threads - 1) / threads;
            power_kernel<<<blocks, threads, 0, streams[s]>>>
                (d_spectr[s], d_power[s], FREQ_BINS, batch);
        }

        // Async copy device→host of power
        CUDA_CHECK(cudaMemcpyAsync(spec_out.data() +
                   frames_done * FREQ_BINS,
                   d_power[s],
                   sizeof(float) * batch * FREQ_BINS,
                   cudaMemcpyDeviceToHost, streams[s]));

        frames_done += batch;
    }

    // 8) Sync & cleanup
    for (int s = 0; s < STREAMS; ++s) {
        CUDA_CHECK(cudaStreamSynchronize(streams[s]));
        cudaFree(d_frames[s]);
        cudaFree(d_spectr[s]);
        cudaFree(d_power[s]);
        cufftDestroy(plan[s]);
        cudaStreamDestroy(streams[s]);
    }
}

int main()
{
    // 2-second mono @ 48 kHz (96 000 samples)
    constexpr int SAMPLE_RATE = 48000;
    const int     SIG_LEN     = SAMPLE_RATE * 2;
    std::cout << "Singal length: " << SIG_LEN << "\n";

    std::vector<float> signal(SIG_LEN);
    std::mt19937 rng(0);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    for (float& v : signal) v = dist(rng);        // white noise

    std::vector<float> spectrogram;
    auto t0 = std::chrono::high_resolution_clock::now();
    compute_spectrogram(signal, spectrogram);
    auto t1 = std::chrono::high_resolution_clock::now();

    std::cout << "Spectrogram time: "
              << std::chrono::duration<double, std::milli>(t1 - t0).count()
              << " ms  (frames=" << spectrogram.size() / FREQ_BINS << ")\n";
}