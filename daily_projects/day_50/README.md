Day 50: Implementation of reverse diffusion, step and loss in CUDA

1) Summary of the daily tutorial

The code implements the reverse diffusion step in CUDA. Given a noisy sample, x_t, at time step, t, and the original clean sample, x_0, below are computed:

- The true noise  
   ```math
   \epsilon = \frac{x_t - \sqrt{\alpha_{\mathrm{cum},t}}\;x_0}{\sqrt{1 - \alpha_{\mathrm{cum},t}}}
   ```
- A noisy prediction of that noise  
   ```math
   \hat{\epsilon} = \epsilon + \sigma_{\mathrm{pred}}\;\zeta
   ```
- The denoised previous sample  
   ```math
   x_{t-1} = \frac{1}{\sqrt{\alpha_t}}\Bigl(x_t - \underbrace{\frac{\beta_t}{\sqrt{1-\alpha_{\mathrm{cum},t}}}}_{\text{=coeff}_\epsilon}\,\hat{\epsilon}\Bigr) + \sigma_t\,\zeta'
   ```
- The per‑pixel loss for training  
   ```math
   \mathcal{L} = (\hat{\epsilon} - \epsilon)^2
   ```

These operations are all fused into a single CUDA kernel for maximum data locality and throughput.

- CUDA kernels:
  - reverse_gpu:
    A templated kernel which, for each of the H x W x C pixels:
    - Initializes a Philox RNG state per thread to draw two Gaussian samples zeta and zeta'.
    - Computes the true noise epsilon from (x_t, x_0).  
    - Forms the noisy noise prediction epsilon = epsilon + sigma * zeta.  
    - Computes the previous timestep sample x_t-1 using the schedule parameters.  
    - Emits both the predicted noise, the denoised sample, and the squared error loss into separate output buffers.

2) Implementation

Compiling the code:

<pre>nvcc .\reverse_diffusion.cu -o rdiffusion</pre>

Running the code after compiling:

<pre>rdiffusion</pre>

<pre>CPU inference time (ms) : 335.599 ms
GPU timings (ms):
  Host-to-Device copy time:      : 2.02515
  Kernel execution time:         : 0.348384
  Device-to-Host copy time:      : 2.87786
  Total GPU time:                : 5.29194</pre>