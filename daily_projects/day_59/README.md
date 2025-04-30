Day 59: Implementation of image encoder on vision transformer backbone in CUDA

1) Summary of the daily tutorial

The code implements an end-to-end image encoder stage for a vision transformer (ViT) backbone in CUDA. The code: 

- Loads a 3-channel (RGB) image of size 1024×1024 into device memory.
- Divides the image into 16×16 patches (total 64×64=4096 tokens).
- Projects each flattened patch into a 256-dimensional embedding, adding a learned positional embedding.
- Applies one transformer encoder block:  
  1. Layer-norm  
  2. Linear projections to Queries, Keys, Values  
  3. Multi-head scaled-dot-product attention  
  4. Residual addition + layer-norm  
  5. Two-layer MLP with GELU activation  
  6. Final residual addition  

Positional embedding addition:
```math
\text{tok}[p, d] \;=\; \sum_{c=0}^{2}\;\sum_{py=0}^{P-1}\;\sum_{px=0}^{P-1}
    \text{img}[c,\,gy\cdot P+py,\,gx\cdot P+px]\;\times\;W_p[(c,py,px),d]
\quad,\quad
\text{tok}[p, d]\;\mathrel{+}=\,\text{pos}[p, d]
```
- CUDA kernels:
  - patch_embed: Loads one 16×16 patch per block, flattens and multiplies it by the patch-projection weight matrix in register/fast memory. It then computes raw patch embeddings and immediately adds the corresponding positional embedding.
  - layernorm: For each token, computes mean and variance across the 256-dim embedding lanes, then normalizes to zero mean and unit variance. It then stabilizes training and inference by standardizing each token’s distribution.
  - linear: Performs a batched GEMV: for each token t and output dim d, computes        
  ```math
  \text{out}[t, d] \;=\; \sum_{k=0}^{ID-1} \text{in}[t, k]\; W[k, d] \;+\; B[d]
  ```
    then implements the Q, K, V projections (ID=OD=256) and could also serve as the final output projection.
  - attn: Each block computes one attention head for one query token:  
      1. Thread 0 loads dot-products <Q_h(t),K_h(j)> for all j, scales by 1/sqrt(d_h), and applies softmax.  
      2. All threads then use the shared probability vector to compute the weighted sum of V_h.  
    then implements multi-head scaled-dot-product attention without any library calls.
  - add_vec: Adds vector b into a element-wise over all tokens and embedding dimensions. It then realizes residual connections (token + attention output, and token + MLP output).
  - mlp: 
    - Two-layer feed-forward network:  
      1. Hidden = GELU(in·W1 + B1) where W1 is [256→1024]  
      2. Out = hidden·W2 + B2 where W2 is [1024→256]  
    then expands the representation, applies non-linearity, then projects back down, as in standard transformer MLP.

2) Implementation

Compiling the code:

<pre>nvcc .\image_encoder.cu -o image_encoder</pre>

Running the code after compiling:

<pre>image_encoder</pre>

<pre>GPU timings (ms):
 Host-to-Device copy time:      :  1.02 ms
 Kernel execution time:         :  10.17 ms
 Device-to-Host copy time:      :  0.06 ms
 Total GPU time:                :  11.30 ms</pre>