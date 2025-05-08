Day 67: Implementation of one dimensional (1D) discrete Fourier transform (DFT) through shared memory in CUDA

1) Summary of the daily tutorial

The code performs computation of an arbitrary-length 1-D DFT and its inverse entirely on the GPU without relying on cuFFT.
  - Pre-computed twiddle factors stored in constant memory (d_twiddle) for fast, broadcast-wide look-ups.
	- Tiled processing with shared memory: each thread block cooperatively loads a contiguous tile of the input sequence, re-uses it for many output frequencies, and thus cuts global-memory traffic.
	- A single templated kernel (dft1d_kernel_sm) that works for both the forward (real → complex) and inverse (complex → complex) directions, selected by DIR = -1 or +1.
	- Built-in timing to report forward and inverse kernel throughput.

Forward DFT (DIR = −1):

	```math
	X[k] \;=\; \sum_{n=0}^{N-1} x[n]\,e^{-j\,2\pi k n / N}, \qquad k = 0,\dots,N-1
	```

Inverse DFT (DIR = +1):

	```math
	x_n \;=\; \frac{1}{N} \sum_{k=0}^{N-1} X_k \, e^{+j\,2\pi k n / N}, \qquad n = 0,\dots,N-1
	```

The kernel evaluates these sums directly but hides N terms of global memory behind one shared-memory load per tile.

  - CUDA kernels:
    - dft1d_kernel_sm: Loads TILE samples into shared memory. For every tile:
	    - All threads compute their contribution to X_k (or x_n) by multiplying the cached sample with the pre-tabulated twiddle.
	    - Uses #pragma unroll for the inner loop over the tile.
	    - Applies the 1/N scale factor when DIR == +1.
	    - Writes the complex result to X_out[k].
	    - Host helper fill_twiddle_table
	    - Generates the length-N table cos(2.pi.n/N) + jsin(2.pi.n/N) once on the CPU and uploads it to constant memory.

  Note that this code was executed on a A100 GPU, however, day 66 was executed on 3050Ti and thus, the difference in inference time can be recognized. 
  However, since the operation is compute bound, shared memory optimization only slightly improved the inference time. Moving the input into shared memory removed most of the DRAM reads, but the dominant cost is still evaluating sincosf() N² times.

2) Implementation

Compiling the code:

<pre>nvcc .\idft_1d_shared.cu -o idft_1d_shared</pre>

Running the code after compiling:

<pre>idft_1d_shared</pre>

<pre>Vector length: 16384 (shared-mem TILE = 256)
CPU forward DFT : 3760.72 ms
CPU inverse DFT : 3978.84 ms

GPU H2D copy     : 0.067584 ms
GPU forward kern : 3.06176 ms
GPU forward D2H  : 2.83853 ms
GPU inverse kern : 2.83034 ms
GPU inverse D2H  : 0.057696 ms</pre>