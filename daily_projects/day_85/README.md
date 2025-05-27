Day 85: Implementation of biquad infinite-impulse response (IIR) filter in CUDA

1) Summary of the daily tutorial

The code implements a single-section biquad IIR filter. Two alternative engines are compared:
- Serial recursion (reference): one thread iterates through the entire signal, maintaining its own delay line.
- Frame recursion: the signal is split into fixed-length frames; each thread processes one frame, but also receives the correct initial history so that the whole stream is filtered seamlessly.

The biquad section applies the classical difference equation

```math
y[n] \;=\; b_0\,x[n] \;+\; b_1\,x[n-1] \;+\; b_2\,x[n-2]
           \;-\; a_1\,y[n-1] \;-\; a_2\,y[n-2],
\qquad (a_0 = 1).
```

- CUDA kernels:
	- biquad_serial: One block, one thread. Walks the entire input array, updating its private state (x₁,x₂,y₁,y₂) at every sample.
	- biquad_frame: Grid: ⌈Nframes / THREADS⌉ blocks, each with THREADS threads. Each thread filters one frame (F samples). Uses a per-frame history struct (Hist) passed from the host so that recursion is continuous across frame boundaries.

2) Implementation

Compiling the code:

<pre>nvcc .\biquad_iir.cu -o biquad_iir</pre>

Running the code after compiling:

<pre>biquad_iir</pre>

<pre>Samples 1048576,   frame 8192,   #frames 128

Max |serial-frame| = 0.872136

                    H2D      Kern     D2H     Total
Serial recursion  H2D 1.066   Kern 33.866   D2H 0.924   Total 35.856 ms
Frame  recursion  H2D 1.066   Kern 2.519    D2H 0.874   Total 4.459 ms</pre>