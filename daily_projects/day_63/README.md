Day 63: Integration of full pipeline for baseline SAM in CUDA

1) Summary of the daily tutorial

The code implements the full pipeline integration for baseline SAM in CUDA, from raw image input all the way through to a final mask output.

- Image encoder: an im2col transformation plus a cuBLAS GEMM to turn a 3 x 1024 x 1024 image into a 4096 x 256 token embedding.
- Prompt encoder: a small kernel that embeds 2D point coordinates plus a type lookup into the same 256-dim space.
- Sequence assembly & normalization: stacking image tokens, prompt tokens, and two zero “padding” tokens into one 4110 x 256 sequence, with a residual-style layer normalization.
- Transformer encoder: three cuBLAS GEMMs for Q, K, V, a custom attention kernel for scaled-dot-product attention + softmax, plus two more GEMMs for the MLP (with a GELU activation) and residual adds.
- Mask decoder: layer-norm the image features, upsample by $\times4$, then project each pixel’s feature vector against two learned "mask tokens" to produce two, 256 x 256 masks.

In each attention head, h:

```math
\text{score}_{t,j}^{(h)} 
= \frac{\sum_{k=1}^{HDIM} Q_{t,h,k}\,K_{j,h,k}}{\sqrt{HDIM}}
\quad\longrightarrow\quad
O_{t,h,d} 
= \sum_{j=1}^{DTOK} \mathrm{softmax}(\text{score}_{t,j}^{(h)})\;V_{j,h,d}
```

- CUDA kernels:
    - add_vec: adds a bias or positional‐embedding vector into a longer feature buffer in parallel (one thread per element).
    - layernorm: computes mean and variance over each token’s 256-d embedding, then normalizes with an RMS term for numerical stability.
    - patch2col: each block reads one 16 x 16 patch (×3 channels), flattens it into a 768-length vector, and writes it - into the 4096 x 768 "column" matrix.
    - attn: For each query token, t, and head, h, computes scaled‐dot‐product scores into shared memory, applies a row‐wise softmax, then accumulates the weighted sum of the value vectors.
    - gelu_kernel: Applies the GELU activation element‐wise across the MLP’s hidden activations.
    - point_embed: Linearly combines each 2D point’s (x,y) with two learned 256-d weight vectors plus a per-type lookup table to produce prompt tokens.
    - mask_downsample: A simple average‐pool over 16 x 16 blocks of the binary mask to produce a 64 x 64 feature grid.
    - mask_project: For each of the 4096 cells, multiplies its downsampled scalar by a 256-d weight vector plus bias to produce dense "prompt‐dense" tokens.
    - upsample4: Copies each coarse 256-d token into a 4 x 4 neighborhood, producing a full 256 x 256 x 256 feature map.
    - proj_mask: For each of the 65,536 output pixels, takes the inner product of its 256-d feature against one of two mask tokens to produce the final logits.

2) Implementation

Compiling the code:

<pre>nvcc .\sam.cu -o sam</pre>

Running the code after compiling:

<pre>sam</pre>

<pre>GPU timings (ms):
 Host-to-Device copy time:      :  1.41 ms
 Kernel execution time:         :  338.60 ms
 Device-to-Host copy time:      :  0.29 ms
 Total GPU time:                :  340.49 ms</pre>