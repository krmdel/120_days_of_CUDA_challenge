Day 84: Implementation of FIR with down-sampling (polyphase decimator) in CUDA

1) Summary of the daily tutorial

The code implements method for acceleartion of a finite-impulse-response (FIR) filter followed by down-sampling by a factor D (a decimator) on the GPU. Instead of computing every FIR output sample and then discarding D – 1 of them, only the required outputs are produced:
- a naïve one-thread-per-output kernel for reference,
- an optimized shared-memory tiled kernel that loads the window of input samples cooperatively and re-uses it for a whole tile of outputs,
- taps stored in constant memory for fast broadcast to every thread,
	
The decimated FIR operation is:

```math
y[m] \;=\; \sum_{k=0}^{T-1} h[k] \, x[mD - k],
\qquad m = 0,\dots,\bigl\lceil\tfrac{N}{D}\bigr\rceil-1
```

where N: input‐signal length, T – number of filter taps, D – decimation factor, h[k] – filter coefficients, stored in constant memory, y[m] – decimated output sequence.

- CUDA kernels:
	- fir_decim_naive: One thread computes one output sample. Reads the needed T input points directly from global memory. Simple to understand but bandwidth-bound and latency-heavy.
	- fir_decim_tiled: Each thread block computes TILE_OUT consecutive outputs. Shared memory holds the window of TILE_OUT·D + (T-1) input samples so every tap reuse is on-chip. Threads cooperatively load the “tile + halo” region, then perform the inner sum. Provides much higher arithmetic intensity and significantly lower global-memory traffic.

2) Implementation

Compiling the code:

<pre>nvcc .\polyphase_decimator.cu -o polyphase_decimator</pre>

Running the code after compiling:

<pre>polyphase_decimator</pre>

<pre>Input signal lenght: 1048576,  taps: 63,  decim: 4,  output: 262144

Naïve decim:      H2D 1.084   Kern 0.068     D2H 0.801   Total 1.952 ms
Shared-mem decim:  H2D 1.084   Kern 0.052     D2H 0.403   Total 1.539 ms</pre>