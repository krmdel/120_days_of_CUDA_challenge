#include <cuda_runtime.h>
#include <cufft.h>
#include <cuComplex.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>

#define CHECK_CUDA(call) \
    { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    }

#define CHECK_CUFFT(call) \
    { \
        cufftResult err = call; \
        if (err != CUFFT_SUCCESS) { \
            std::cerr << "CUFFT error: " << err << " at line " << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    }

// Kernel for element-wise complex multiplication
__global__ void complexPointwiseMul2D(cufftComplex* a, const cufftComplex* b, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = y * width + x;

    if (x < width && y < height) {
        a[idx] = cuCmulf(a[idx], b[idx]);
    }
}

int main() {
    const int WIDTH = 128;
    const int HEIGHT = 128;
    const int N = WIDTH * HEIGHT;
    size_t size = sizeof(cufftComplex) * N;
        std::cout << "Image " << WIDTH << "×" << HEIGHT
              << " (" << N << " px)\n\n";
    // Host inputs
    std::vector<float> x(N), h(N);
    for (int i = 0; i < HEIGHT; ++i) {
        for (int j = 0; j < WIDTH; ++j) {
            x[i * WIDTH + j] = sinf(2 * M_PI * i / HEIGHT) * cosf(2 * M_PI * j / WIDTH);
            h[i * WIDTH + j] = cosf(2 * M_PI * i / HEIGHT) * sinf(2 * M_PI * j / WIDTH);
        }
    }

    // Convert to complex
    std::vector<cufftComplex> x_c(N), h_c(N);
    for (int i = 0; i < N; ++i) {
        x_c[i] = make_cuFloatComplex(x[i], 0.0f);
        h_c[i] = make_cuFloatComplex(h[i], 0.0f);
    }

    // Device memory
    cufftComplex *d_x, *d_h;
    CHECK_CUDA(cudaMalloc(&d_x, size));
    CHECK_CUDA(cudaMalloc(&d_h, size));
    CHECK_CUDA(cudaMemcpy(d_x, x_c.data(), size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_h, h_c.data(), size, cudaMemcpyHostToDevice));

    // cuFFT plan
    cufftHandle plan;
    CHECK_CUFFT(cufftPlan2d(&plan, HEIGHT, WIDTH, CUFFT_C2C));

    // Timing
    cudaEvent_t start, stop;
    float elapsed;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Forward FFTs
    CHECK_CUFFT(cufftExecC2C(plan, d_x, d_x, CUFFT_FORWARD));
    CHECK_CUFFT(cufftExecC2C(plan, d_h, d_h, CUFFT_FORWARD));

    // Element-wise multiply
    dim3 threads(16, 16);
    dim3 blocks((WIDTH + threads.x - 1) / threads.x, (HEIGHT + threads.y - 1) / threads.y);
    complexPointwiseMul2D<<<blocks, threads>>>(d_x, d_h, WIDTH, HEIGHT);
    CHECK_CUDA(cudaGetLastError());

    // Inverse FFT
    CHECK_CUFFT(cufftExecC2C(plan, d_x, d_x, CUFFT_INVERSE));

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);

    // Copy back and normalize
    std::vector<cufftComplex> result(N);
    CHECK_CUDA(cudaMemcpy(result.data(), d_x, size, cudaMemcpyDeviceToHost));
    for (int i = 0; i < N; ++i) {
        result[i].x /= N;
        result[i].y /= N;
    }

    // Printing inference time
    std::cout << "\nInference time: " << elapsed << " ms\n";

    // Cleanup
    cufftDestroy(plan);
    cudaFree(d_x);
    cudaFree(d_h);

    return 0;
}