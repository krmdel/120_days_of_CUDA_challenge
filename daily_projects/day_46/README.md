Day 47: Implementation of decoder in CUDA

1) Summary of the daily tutorial

The code implements decoder in CUDA, generating text one token at a time from visual features and a token embedding table. At each step, it performs:

  - Self‑attention: over the already‑generated tokens with causal masking  
  - Cross‑attention: to a fixed set of visual feature vectors  
  - A two‑layer feed‑forward network  
  - Greedy decoding by arg‑max over the vocabulary  

  ```math
  \text{Attention}(Q, K, V) \;=\; \mathrm{softmax}\!\Bigl(\tfrac{QK^T}{\sqrt{D_{\mathrm{head}}}}\Bigr)\,V
  ```

  ```math
  \mathrm{Context} \;=\; \mathrm{softmax}\!\Bigl(\tfrac{Q_c\,K_c^T}{\sqrt{D_{\mathrm{head}}}}\Bigr)\,V_c
  ```

  ```math
  \mathrm{FFN}(x) \;=\; \mathrm{ReLU}(x\,W_1)\,W_2
```

- CUDA kernels:
  - tiledMatMulKernel: Performs a tiled, shared‑memory matrix multiplication for all linear projections (Q, K, V, feed‑forward, logits).  
  - transposeKernel: Transposes a matrix to prepare K for dot‑product attention.  
  - scaleKernel: Scales the raw attention scores by 1/sqrt(D_head).  
  - softmaxKernel: Applies a row‑wise softmax, using a small shared‑memory buffer for max and sum.  
  - causalMaskKernel: Writes -inf (-10^9) above the diagonal to enforce causal masking in self‑attention.  
  - addInPlace: Adds the projected attention or FFN output back into the token sequence in place.  
  - reluKernel: Applies a ReLU activation to the first FFN layer outputs.  
  - extractQuery: Gathers the last token’s hidden state into a standalone Q vector for cross‑attention.  
  - argmaxKernel: Finds the index of the maximum logit (greedy sampling) across the full vocabulary in parallel.  
  - embedToken: Looks up a token embedding by index and writes it into the sequence buffer.  
  - writeId: Writes the newly selected token ID into the "ids" array for the next iteration.  

2) Implementation

Compiling the code:

<pre>nvcc .\decoder.cu -o decoder</pre>

Running the code after compiling:

<pre>decoder</pre>

<pre>Total CPU inference time (ms): 461.259 ms
GPU timings (ms):
  Host-to-Device copy time: 3.78982
  Kernel execution time:    7.78854
  Device-to-Host copy time: 0.1072
  Total GPU time:           11.7078</pre>