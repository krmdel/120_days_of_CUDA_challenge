Day 42: Implementation of integration for vision transfomer in CUDA

1) Summary of the daily tutorial:

The code implements the integration of vision transformer components from previous tutorials (patch_embeddings, positional_embeddings and encoder).

- Patch embedding: The input image is divided into non-overlapping patches. Each patch is flattened and then multiplied with a learned weight matrix to produce a patch embedding. In the kernel, tiling and shared memory are employed to efficiently load and process each patch.

- Positional embedding addition:  Since the patch extraction loses spatial information, a sinusoidal positional encoding is added element‑wise to each patch embedding. The positional encoding is computed as follows:
  
  ```math
  \text{embedding}[p, d] \mathrel{+}= \text{PE}(p, d)
  ```
  
  where for each patch index \( p \) and embedding dimension \( d \), the positional encoding \( \text{PE}(p, d) \) is generated using sine and cosine functions with frequencies that vary with the embedding index.

- Transformer encoder: A simple transformer encoder layer is implemented to process the sequence of patch embeddings. The encoder performs:
  
  - Linear projections to compute queries, Q, keys, K, and values, V, from the patch embeddings
  - Calculation of the dot-product attention scores using Q and the transpose of K, with a scaling factor 1/sqrt(D) where D is the model dimension
  - Application of a row‑wise softmax to the attention scores
  - A final linear projection of the attention outputs via the output weight matrix

- CUDA kernels:

    - patch_embedding_kernel_tiled: A tiled kernel loads each patch into shared memory and computes a dot product with the weight matrix for each patch. Each CUDA block processes one patch by using all available threads (e.g., 1024 threads for a \(32 \times 32\) patch)
  
    - add_positional_embeddings_tiled:A dedicated kernel adds the sinusoidal positional encoding to each patch embedding in a tiled fashion. The element‑wise update follows the formula:
    
        ```math
        \text{embedding}[p, d] \mathrel{+}= \text{PE}(p, d)
        ```
  
        This operation updates each embedding with a computed sine or cosine value based on its patch index, p, and dimension, d.

  - tiledMatMulKernel: Tiled matrix multiplication kernels are used for the linear projections (computing, Q, K and V), as well as for calculating the attention and the final output
  - transposeKernel: A transpose kernel rearranges the key matrix to compute the dot-product attention scores efficiently
  - scaleKernel: A scaling kernel multiplies all elements in the score matrix by 1/sqrt(model_dim)
  - softmaxKernel: A custom kernel applies a row‑wise softmax to the scaled dot-product scores to obtain a probabilistic attention map

2) Implementation:

Compiling the code:  

<pre>nvcc .\integrated_vit.cu -o vit</pre>

Running the code after compiling: 

<pre> vit </pre>

<pre>CPU Full Inference Time (ms): 1003.64
GPU Timings (ms):
  Host-to-Device copy time: 0.59904
  Kernel execution time:   4.60698
  Device-to-Host copy time: 0.182816
  Total GPU time:          5.43149</pre>