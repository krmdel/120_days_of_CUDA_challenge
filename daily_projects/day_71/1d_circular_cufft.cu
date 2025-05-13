#include <cuda_runtime.h>
#include <cufft.h>
#include <cuComplex.h>
#include <iostream>
#include <vector>
#include <chrono>

#define CHECK_CUDA(call)                                                  \
    {                                                                     \
        cudaError_t err = call;                                           \
        if (err != cudaSuccess) {                                         \
            std::cerr << "CUDA error: " << cudaGetErrorString(err)       \
                      << " at line " << __LINE__ << std::endl;           \
            exit(EXIT_FAILURE);                                           \
        }                                                                 \
    }

#define CHECK_CUFFT(call)                                                 \
    {                                                                     \
        cufftResult err = call;                                           \
        if (err != CUFFT_SUCCESS) {                                       \
            std::cerr << "CUFFT error: " << err << " at line " << __LINE__ << std::endl; \
            exit(EXIT_FAILURE);                                           \
        }                                                                 \
    }

// Element-wise complex multiplication kernel
__global__ void complexPointwiseMul(cufftComplex* a, const cufftComplex* b, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        a[i] = cuCmulf(a[i], b[i]);
    }
}

int main() {
    const int N = 16'384;
    size_t size = sizeof(cufftComplex) * N;
    std::cout << "Vector length: " << N << "\n";

    // Host inputs
    std::vector<float> x(N, 0.0f), h(N, 0.0f);
    for (int i = 0; i < N; ++i) {
        x[i] = sinf(2 * M_PI * i / N);
        h[i] = cosf(2 * M_PI * i / N);
    }

    // Device buffers
    cufftComplex *d_x, *d_h;
    CHECK_CUDA(cudaMalloc(&d_x, size));
    CHECK_CUDA(cudaMalloc(&d_h, size));

    // Copy real input as complex
    std::vector<cufftComplex> x_c(N), h_c(N);
    for (int i = 0; i < N; ++i) {
        x_c[i] = make_cuFloatComplex(x[i], 0.0f);
        h_c[i] = make_cuFloatComplex(h[i], 0.0f);
    }

    CHECK_CUDA(cudaMemcpy(d_x, x_c.data(), size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_h, h_c.data(), size, cudaMemcpyHostToDevice));

    cufftHandle plan;
    CHECK_CUFFT(cufftPlan1d(&plan, N, CUFFT_C2C, 1));

    cudaEvent_t start, stop;
    float elapsed;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Forward FFT
    CHECK_CUFFT(cufftExecC2C(plan, d_x, d_x, CUFFT_FORWARD));
    CHECK_CUFFT(cufftExecC2C(plan, d_h, d_h, CUFFT_FORWARD));

    // Pointwise multiplication
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    complexPointwiseMul<<<blocks, threads>>>(d_x, d_h, N);

    // Inverse FFT
    CHECK_CUFFT(cufftExecC2C(plan, d_x, d_x, CUFFT_INVERSE));

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);

    // Copy back result and normalize
    std::vector<cufftComplex> result(N);
    CHECK_CUDA(cudaMemcpy(result.data(), d_x, size, cudaMemcpyDeviceToHost));
    for (int i = 0; i < N; ++i) {
        result[i].x /= N;
        result[i].y /= N;
    }

    // Printing inference time
    std::cout << "\nInference time: " << elapsed << " ms\n";

    // Cleanup
    cudaFree(d_x);
    cudaFree(d_h);
    cufftDestroy(plan);
    return 0;
}