Day 91: Implementation of reconstruction, peak signal-to-noise ratio (PSNR) in CUDA

1) Summary of the daily tutorial

The code takes an 8-bit grayscale image (or a randomly generated one), compresses it with a classic JPEG-style pipeline entirely on the GPU, and reconstructs the image:
- Forward 8 × 8 DCT on every block.
- Zig-zag scan the 64 DCT coefficients into 1-D order.
- Scalar quantisation with a user‐selectable quality factor qf.
- Inverse steps: de-quantise, inverse zig-zag, inverse DCT.
- Quality metric & I/O: mean-squared-error → PSNR, and optional PNG export.

Quantisation of each coefficient C_{u,v} is

```math
Q_{u,v} \;=\; \operatorname{round}
\!\Bigl( \dfrac{C_{u,v}}{q_f \,\times\, Q_{\text{tbl}}[u,v]} \Bigr)
```

and the inverse step is

```math
\tilde{C}_{u,v} \;=\;
q_f \,\times\, Q_{\text{tbl}}[u,v] \times Q_{u,v}.
```

- CUDA kernels:
    - fwdDCT: computes the forward 2-D DCT of each 8 × 8 block using shared memory for both row- and column-wise passes.
    - invDCT: performs the inverse 2-D DCT to reconstruct spatial-domain pixels.
    - zigzag: copies the 8 × 8 block into a linear buffer following the JPEG zig-zag pattern.
    - quant: applies scalar quantisation to every coefficient: Q=\mathrm{round}(C/(q_f\cdot Q_{\text{tbl}})).
    - dequant_invzig: merges de-quantisation with inverse zig-zag, rebuilding an 8 × 8 block in raster order.
    - mseK: accumulates per-pixel squared error to produce an image-wide MSE, later converted to PSNR on the host.

2) Implementation

Compiling the code:

<pre>nvcc .\recon.cu -o recon</pre>

Running the code after compiling:

<pre>recon</pre>

<pre>GPU total 4.20234 ms,  PSNR 13.4749 dB
Saved recon.png</pre>