Day 48: Implementation of contrastive language-image pre-training (CLIP) in CUDA

1) Summary of the daily tutorial

The code integrates the vision transformer and text‐transformer to build a CLIP pipeline in CUDA. Starting from raw image and token inputs. The code,  

- Computes patch embeddings and adds sinusoidal positional encodings  
- Runs a transformer encoder over image patches, then mean‑pools and L2‑normalizes to get an image vector  
- Runs a transformer encoder over token embeddings, applies feed‑forward, residuals, layer‑norm pools and normalizes to get a text vector  
- Finally computes cosine similarity between the two vectors via a custom dot‑product kernel  

    - Positional embedding:  
    ```math
    \text{patch\_emb}[p, d] \mathrel{+}= \sin\bigl(\tfrac{p}{10000^{2\lfloor d/2\rfloor/D}}\bigr)\quad\text{or}\quad\cos\bigl(\tfrac{p}{10000^{2\lfloor d/2\rfloor/D}}\bigr)
    ```  

    - Contrastive similarity:
    ```math
    \text{sim}(v_{\rm img}, v_{\rm txt}) \;=\;\frac{v_{\rm img} \cdot v_{\rm txt}}{\|v_{\rm img}\|\,\|v_{\rm txt}\|}
    ```  

- CUDA kernels:
  - patch_embed_kernel: Loads each 32×32 image patch into shared memory and projects it into a 256‑dim embedding.  
  - add_pos_kernel: Adds element‑wise sinusoidal positional encodings to each patch embedding.  
  - tiledMatMulKernel: Performs tiled matrix multiplication for all linear projections (Q, K, V, output) and attention score computations.  
  - transposeKernel: Transposes the key matrix to enable efficient dot‑product attention.  
  - scaleKernel: Scales the raw attention scores by 1/√D to stabilize gradients.  
  - softmaxKernel: Applies a row‑wise softmax to convert scaled scores into attention weights.  
  - mean_pool_kernel: Averages the sequence of patch embeddings to form a single image feature vector.  
  - l2_norm_kernel: Applies L2 normalization so that image and text vectors lie on the unit hypersphere.  
  - dot_kernel: Computes the dot product (cosine similarity) between two normalized D‑dim vectors.  
  - addInPlace: Implements residual connections by element‑wise addition.  
  - reluKernel: Applies ReLU nonlinearity in the feed‑forward sublayer of the text encoder.  
  - layer_norm_kernel: Performs layer normalization across the hidden dimension after each residual block.  

2) Implementation

Compiling the code:

<pre>nvcc .\clip.cu -o clip</pre>

Running the code after compiling:

<pre>clip</pre>

<pre>CPU cosine similarity: 0.000426968
Total CPU inference time (ms): 3008.06

GPU cosine similarity: 0.000441317
GPU timings (ms):
  Host-to-Device copy time: : 4.16595
  Kernel execution time:    : 6.8175
  Device-to-Host copy time: : 0.070976
  Total GPU time:           : 11.07</pre>