Day 51: Implementation of latent U-Net denoiser in CUDA

1) Summary of the daily tutorial

The code implements a core residual convolutional block of a latent U‑Net denoiser in CUDA. The block consists of two 3×3 convolutions (with padding), a ReLU activation in between, and a skip connection that adds the original input back into the final output. Constant memory is utilized for weights, shared‐memory tiling for efficient convolution, and custom kernels for activation and residual addition.

```math
y_{oc,y,x} = \sum_{ic=0}^{C_{in}-1}\sum_{k_y=-1}^{1}\sum_{k_x=-1}^{1} w_{oc,ic,k_y,k_x}\,\;x_{ic,y+k_y,x+k_x}
```

```math
\text{output} = \mathrm{Conv2}\bigl(\mathrm{ReLU}(\mathrm{Conv1}(x))\bigr) + x
```

- CUDA kernels:
  - conv3x3_kernel: Loads a (TILE + 2) x (TILE + 2) patch of the input into shared memory (including halo), performs a 3×3 convolution over all input channels, and writes one output channel per block.  
  - relu_kernel: Launches one thread per element to apply an elementwise ReLU, i.e. t_i = max(t_i,0).  
  - add_residual_kernel: Launches one thread per element to add the original input tensor to the convolved output, implementing the U‑Net skip connection.

2) Implementation

Compiling the code:

<pre>nvcc .\latent_unet.cu -o unet</pre>

Running the code after compiling:

<pre>unet</pre>

<pre>CPU inference time (ms) : 11.2384 ms
GPU timings (ms):
  Host-to-Device copy time:      : 0.068256
  Kernel execution time:         : 0.275808
  Device-to-Host copy time:      : 0.010368
  Total GPU time:                : 0.375808</pre>