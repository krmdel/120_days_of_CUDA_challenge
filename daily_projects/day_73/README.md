Day 73: Implementation of overlap-add FFT-based convolution for long 1D signals in CUDA

1) Summary of the daily tutorial

The code implements how to convolve very long 1-D signals with finite-length filters completely on the GPU by combining cuFFT transforms with two lightweight custom kernels.
Instead of computing one huge FFT of the entire signal, the overlap-add (OLA) technique splits the input into hop-sized blocks (L samples each), transforms them independently, multiplies each block in the frequency domain with the (zero-padded) kernel spectrum, and finally stitches the inverse-FFT blocks together with an additive overlap equal to the kernel length − 1.

For each block b

Y_b[k] \;=\; X_b[k] \, H[k], \qquad
y[n] \;=\; \sum_{b=0}^{B-1} \operatorname{ifft}\!\bigl\{ Y_b[k] \bigr\}[\,n-bL\,],

where X_b[k] – FFT of the b-th hop-sized segment / H[k] – FFT of the zero-padded kernel (shared by every block) / L – hop size (non-overlapped region of each segment)

    - CUDA kernels
        - multiply_complex_batch: Per-block point-wise multiplication in the frequency domain. Each thread multiplies one complex spectrum bin of a signal block with the corresponding kernel bin:

        Y_b[k] = X_b[k] \cdot H[k]
        
        - overlap_add_accumulate: Adds the inverse-FFT time-domain blocks into the global output with correct offsets and normalises by 1/N. Atomic adds ensure safe accumulation when multiple blocks update overlapping regions.

Note that: 
    - Overlap-add is only faster than direct time-domain O(N·M) when each block is big enough that FFT launch overhead is dwarfed by arithmetic.
    - One large, batched cuFFT is usually best until the signal length exceeds GPU memory.

2) Implementation

Compiling the code:

<pre>nvcc .\overlap.cu -o overlap</pre>

Running the code after compiling:

<pre>overlap</pre>

<pre>Signal size: 65536 

Optimised Overlap-Add convolution: 125.085 ms</pre>