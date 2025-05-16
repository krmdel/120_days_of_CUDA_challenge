Day 72: Implementation of two dimensional (2D) circular convolution using cuFFT in CUDA

1) Summary of the daily tutorial

The code performs a 2-D circular convolution of two real-valued images, x[m,n] and h[m,n]. It first casts each real image to a complex array. Thenm computes the forward 2-D FFT of both images with cuFFT and multiplies the resulting spectra element-wise in a custom kernel. It then computes the inverse 2-D FFT to obtain the convolved image and normalizes by the total number of pixels N = H x W.

```math
Y = \operatorname{IFFT2}\!\bigl(\,\operatorname{FFT2}(X)\;\odot\;\operatorname{FFT2}(H)\bigr),
\qquad  y[m,n] = \frac{1}{HW}\sum_{i=0}^{H-1}\sum_{j=0}^{W-1}
                x[i,j]\;h[(m-i)\!\bmod H,\,(n-j)\!\bmod W].
```

- CUDA kernels:
    - complexPointwiseMul2D: Per-pixel complex multiplication of the two frequency-domain images. Each thread multiplies a pair of complex numbers. Because cuFFT implements full wrap-around (periodic) boundaries, the result is a true circular convolution—no explicit zero-padding or region extraction is required.

2) Implementation

Compiling the code:

<pre>nvcc .\2d_circular_cufft.cu -o 2d_circular_cufft</pre>

Running the code after compiling:

<pre>2d_circular_cufft</pre>

<pre>Image 128×128 (16384 px)

Inference time: 0.144384 ms</pre>