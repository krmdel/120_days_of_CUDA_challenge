Day 62: Implementation of mask decoder in CUDA

1) Summary of the daily tutorial

The code implements mask decoder, converting a 64 × 64 grid of feature tokens (produced by the ViT encoder) plus a small set of learned "mask-tokens" into two full-resolution 256 × 256 segmentation masks.

```math
\hat{\mathbf{f}}_{p,d}
= \frac{\mathbf{f}_{p,d}-\mu_p}{\sqrt{\sigma_p^{2}+\varepsilon}}
\;(\text{layer-norm})
\quad\longrightarrow\quad
\mathbf{f}^{\uparrow}_{y,x,d}
= \hat{\mathbf{f}}_{\left\lfloor \tfrac{y}{4} \right\rfloor,
                     \left\lfloor \tfrac{x}{4} \right\rfloor,d}
\;(4\times\text{ up-sample})
\quad\longrightarrow\quad
m_k[y,x]
= \displaystyle\sum_{d=0}^{D-1}
    \mathbf{f}^{\uparrow}_{y,x,d}\,t_{k,d}
\;(\text{dot-product mask projection})
```

where p indexes the 4096 feature tokens, d ∈ [0,255] is the embedding dimension, (y,x) ∈ [0,255]^2 are output pixel coordinates, k ∈ {0,1} indexes the two mask-tokens.

- CUDA kernels:
    - Layer-norm: on every feature token to stabilise the dot-product that follows.
    - 4 × nearest-neighbour up-sampling (replicate each 64 × 64 token to a 4 × 4 block) to restore the original image resolution.
    - Dot-product projection between each up-sampled pixel feature and each mask-token to obtain one scalar logit per pixel and per mask.

2) Implementation

Compiling the code:

<pre>nvcc .\mask_decoder.cu -o mask_decoder</pre>

Running the code after compiling:

<pre>mask_decoder</pre>

<pre>GPU timings (ms):
 Host-to-Device copy time:      :  0.41 ms
 Kernel execution time:         :  1.93 ms
 Device-to-Host copy time:      :  0.16 ms
 Total GPU time:                :  2.66 ms</pre>