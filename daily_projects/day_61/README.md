Day 61: Implementation of two way attention decoder in CUDA

1) Summary of the daily tutorial

The code implements two-way attention decoder that fuses prompt tokens and image patch features. Starting from host-generated token, hTok, and feature, hFeat, buffers, copying data to device memory and then execute:

- Token normalization and self-attention over prompt tokens
- Cross-attention from tokens to image features
- Cross-attention from image features back to tokens
- A final token MLP for prediction

  Key operations:

  ```math
  \hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}
  \quad\text{(LayerNorm)}
  ```

  ```math
  a_i \;\leftarrow\; a_i + b_i
  \quad\text{(Residual addition)}
  ```

  ```math
  y = W\,x
  \quad\text{(Linear projection)}
  ```

  ```math
  \mathrm{Attention}(Q,K,V) = \mathrm{softmax}\!\Bigl(\frac{Q\,K^\top}{\sqrt{d_k}}\Bigr)\,V
  \quad\text{(Scaled dot-product attention)}
  ```

  ```math
  \mathrm{FFN}(x) = W_2\;\mathrm{GELU}\bigl(W_1\,x\bigr)
  \quad\text{(Two-layer feed-forward network)}
  ```

- CUDA kernels:
  - add_vec: Element-wise vector addition to apply residual connections. adds buffer B into A (A[i]+=B[i]).
  - ln_tok: Layer-normalizes each token embedding of length EMB. computes mean μ and variance σ² over EMB, then (x–μ)/√(σ²+ϵ).
  - ln_feat: Same as ln_tok, but applied to image feature vectors. 
  - linear: Dense projection: for each of M tokens/features, computes out[m,d] = ∑ₖ in[m,k]·W[k,d].
  - attn: Scaled dot-product attention with template parameters for query/key lengths:
    - Computes per-head dot products into an extern shared buffer
    - Applies softmax over KLEN scores
    - Accumulates context ∑ₖ probs[k]·V[k] into output.
  - mlp: Two-layer MLP with GELU activation:
    - Hidden = GELU(W1·in)
    - Output = W2·hidden

2. Implementation:

Compiling the code:

<pre>nvcc day61_decoder.cu -o decoder</pre>

Running the code after compiling:

<pre>./decoder</pre>

<pre>
GPU timings (ms):
  Host-to-Device copy time:      :  X.XX ms
  Kernel execution time:         :  Y.YY ms
  Device-to-Host copy time:      :  Z.ZZ ms
  Total GPU time:                :  W.WW ms
</pre>























2) Implementation

Compiling the code:

<pre>nvcc .\decoder.cu -o decoder</pre>

Running the code after compiling:

<pre>decoder</pre>

<pre>GPU timings (ms):
 Host-to-Device copy time:      :  0.40 ms
 Kernel execution time:         :  5.08 ms
 Device-to-Host copy time:      :  0.54 ms
 Total GPU time:                :  6.84 ms</pre>