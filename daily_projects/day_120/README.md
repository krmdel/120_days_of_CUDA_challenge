Day 120: Implementation of point-cloud generation from RGB-D image in CUDA

1) Summary of the daily tutorial

The code converts an RGB-D frame (a color image plus a per-pixel depth map) into a 3-D point cloud on the GPU. For every pixel (x, y) with depth d it:

```math
Z = d \times \text{scale},\quad
X = \frac{(x - c_x)\,Z}{f_x},\quad
Y = \frac{(y - c_y)\,Z}{f_y}
```

producing a metric 3-D position p = (X, Y, Z, 1) and a corresponding RGBA colour.
The code times CPU and GPU implementations, illustrating the speed-up gained from a simple but highly parallel CUDA kernel.

- CUDA kernels:
  - rgbd2pcKer: 2-D grid of 16 × 16 threads; one thread per pixel. Loads depth (uint16_t) and colour (uchar3) for its pixel. Computes 3-D coordinates with the intrinsic camera matrix (fₓ, fᵧ, cₓ, cᵧ). Writes float4 point positions to xyz and uchar4 colours to rgbOut. Inserts NaN coordinates when depth = 0 to mark invalid pixels.

2) Implementation

Compiling the code:

<pre>nvcc .\point_cloud.cu -o point_cloud</pre>

Running the code after compiling:

<pre>point_cloud</pre>

<pre>[Info] random RGB-D 3840×2160

Image size              : 3840 × 2160  (8294400 px)
Intrinsics fx fy cx cy  : 525 525 319.5 239.5

GPU H2D copy            : 9.10221 ms
GPU kernel              : 0.233472 ms
GPU D2H copy            : 34.9356 ms
GPU total               : 121.913 ms
CPU total               : 112.781 ms</pre>