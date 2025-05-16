Day 74: Implementation of power-spectrogram with streams and kernel chaining in CUDA

1) Summary of the daily tutorial

The code implements computation of a long 1-D audio signal into a power spectrogram. It does so in a streaming, double-buffered fashion so that data transfers, windowing, FFT, and magnitude-squared steps overlap in flight on two CUDA streams. This hides most memory-latency and keeps the GPU busy end-to-end.

    - CUDA kernels
	    - apply_window: Multiplies every sample in every frame by the pre-computed Hann window held in constant memory. Each block processes a contiguous span of samples so global reads are coalesced.
	    - power_kernel: Converts the complex FFT output to power by computing

	- Two CUDA streams (STREAMS = 2) form a double buffer: while stream 0 is busy on the GPU, stream 1 can move the next chunk of frames across PCIe, and vice-versa.
	- Kernels are launched back-to-back in the same stream (apply_window → cuFFT → power_kernel), so the output of one becomes the input of the next without expensive synchronisations or extra copies.

2) Implementation

Compiling the code:

<pre>nvcc .\spectogram_stream.cu -o spectogram_stream</pre>

Running the code after compiling:

<pre>spectogram_stream</pre>

<pre>Singal length: 96000
Spectrogram time: 120.95 ms  (frames=372)</pre>