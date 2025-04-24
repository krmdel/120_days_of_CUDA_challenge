Day 53: Implementation of sampling and schedule optimization in CUDA

1) Summary of the daily tutorial

The code implements a fast 50‑step sampling loop for a latent diffusion model using CUDA. The code:

- Builds a square‑spaced timestep schedule (sqrt_schedule) and computes diffusion hyperparameters (β, α, cumulative α, and derived coefficients).
- Implements a tiny U‑Net denoiser via two 3×3 convolutions, ReLU activation, and a residual connection.
- Uses per‑element Philox RNG states to add Gaussian noise in the reverse diffusion step.
- Measures and compares CPU vs GPU performance, including host‑to‑device and device‑to‑host copy times, kernel execution time, and total GPU time.

- CUDA kernels:
  - conv3: Loads a tile (plus 1‑pixel halo) of the input into shared memory and computes a 3×3 convolution for each output channel in a tiny U‑Net layer.
  - relu: Applies element‑wise ReLU:  

    ```cpp
    t[i] = fmaxf(t[i], 0);
    ```
  - add_res: Adds the input latent, x_t, back into the network output (y += x) for the residual connection.
  - rng_init: Initializes a Philox-based, curandState, for each element of the latent tensor.
  - reverse_step: Applies the reverse diffusion update:  

    ```math
    x_{t-1} = \frac{1}{\sqrt{\alpha_t}}\bigl(x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\;\epsilon_\theta(x_t)\bigr)
              \;+\;\sigma_t\,z,\quad z\sim\mathcal{N}(0,1)
    ```

- CUDA kernels:
  - conv3: Predicts noise residuals by applying two successive 3×3 conv layers.
  - relu: Introduces non‑linearity after the first conv layer.
  - add_res: Merges the denoiser output with the original latent to form a skip connection.
  - rng_init: Seeds GPU RNG for stochastic noise sampling.
  - reverse_step: Transforms the noisy latent toward the clean data manifold according to the optimized schedule.

2) Implementation

Compiling the code:

<pre>nvcc .\sampling_schedule.cu -o sampling</pre>

Running the code after compiling:

<pre>sampling</pre>

<pre>CPU inference time           : 544.784 ms
GPU timings (ms):
  Host-to-Device copy time:      : 0.026624
  Kernel execution time:         : 2.35315
  Device-to-Host copy time:      : 0.023872
  Total GPU time:                : 2.43507</pre>