Day 49: Implementation of forward diffusion in CUDA

1) Summary of the daily tutorial

The code implements the forward diffusion step of a diffusion model in CUDA. First, β‑scheduler is computed and its cumulative product ᾱₜ, then for a chosen timestep T_SAMPLE, generate Gaussian noise ε and blend it with the original signal x₀ according to the diffusion equation. Single templated kernel is initialized per‑thread CURAND states, sample noise, and compute the diffused output in one pass, while overlapping data transfers and computation via CUDA streams and events.

The diffusion update is:

```math
x_t[i] \mathrel{=} \sqrt{\bar\alpha_t}\;x_0[i] \;+\;\sqrt{1 - \bar\alpha_t}\;\epsilon[i]
```

- CUDA kernels:
  - fd_gpu_rand:
    - A templated global kernel where each thread:
      - Initializes a "curandStatePhilox4_32_10_t" with a fixed seed and its unique index.
      - Samples ε ~ N(0,1) via "curand_normal".
      - Writes ε into the "noise" buffer for later host‐side validation.
      - Computes the forward‐diffused value  
        
        ```math
        \text{out}[i] = \sqrt{\bar\alpha_t}\,\text{x0}[i] + \sqrt{1 - \bar\alpha_t}\,\epsilon
        ```  
      - Stores the result into the `out` buffer.

- State initialization: "curand_init(seed, idx, 0, &state)" ensures each thread’s RNG is reproducible.
- Noise sampling: "curand_normal(&state)" draws a standard Gaussian for every pixel.
- Diffusion compute: blends "x0" and ε using precomputed scalars "sa = √ᾱₜ" and "soa = √(1−ᾱₜ)".
- Memory writes: writes both ε and the blended output so that host code can validate GPU vs. CPU.

2) Implementation

Compiling the code:

<pre>nvcc .\forward_diffusion.cu -o fdiffusion</pre>

Running the code after compiling:

<pre>fdiffusion</pre>

<pre>CPU inference time (ms) : 3.9058 ms
GPU timings (ms):
  Host-to-Device copy time:      : 1.1135
  Kernel execution time:         : 0.770496
  Device-to-Host copy time:      : 1.91946
  Total GPU time:                : 3.79891</pre>