Day 68: Implementation of one dimensional (1D) discrete Fourier transform (DFT) in CUDA

1) Summary of the daily tutorial

The code implements the previous baseline SAM implementation by leveraging Flash Attention 1.0 for:
    - converting a 3-channel 1024 × 1024 image into a sequence of 4096 patch tokens + 2 prompt tokens (total 4,110 tokens)
    - linearly projects the tokens into an embedding space of size 256
    - applies two alternative single-head self-attention implementations

- CUDA kernels:
    - patch2col: gathers every 16 × 16 image patch into a column layout (TOK × PVEC) so cuBLAS can treat patch extraction as a GEMM.
    - add_vec: element-wise addition of learned vectors (e.g. positional or bias) to a large tensor.
    - layernorm: in-place Layer-Norm across the embedding dimension for each token:
    
    ```math
    \hat{x}_{t,d} 
    = \frac{x_{t,d}-\mu_t}{\sqrt{\sigma_t^{2}+\varepsilon}}
    \quad\text{(Layer normalization)}
    ```

    - attn_naive: baseline scaled-dot-product attention. Each block handles one (token, head) pair; probabilities are stored in dynamic shared memory.
    - flash_attn: Flash Attention 1.0 (CTA-tiled) processes a BLOCK_M × BLOCK_K tile at a time, maintaining (m, ℓ) running statistics in registers so only one soft-max pass is required. Achieves *O(N<sup>2</sup>)* arithmetic with *O(N)* memory.

    Workflow:

    Patch & embedding → Layer-Norm → Sequence build → (Q,K,V) GEMMs → Attention → context tensor

2) Implementation

Compiling the code:

<pre>nvcc .\flash_sam.cu -o flash_sam</pre>

Running the code after compiling:

<pre>flash_sam</pre>

<pre>Image size: 128 x 128 (16384 pixels)

CPU DFT     : 4787.070 ms
GPU H2D     : 1.132 ms
GPU Kernel  : 8.352 ms
GPU D2H     : 0.054 ms
GPU Total   : 9.537 ms</pre>