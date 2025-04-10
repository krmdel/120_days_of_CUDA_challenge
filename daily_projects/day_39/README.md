Day 39: Implementation of image patch extraction and generating patch embeddings in CUDA

1) Summary of the daily tutorial:

The code performs extraction of non-overlapping patches from an input image and generation of patch embeddings by applying a linear projection for vision transfomers

- Image patch extraction: The input image, X, of size, H x W, 1024 x 1024 is divided into non-overlapping patches of size, P x P, with P = 8. Each patch is defined by its top‑left coordinates determined as:  
  \[
  r = \left\lfloor \frac{p}{N_w} \right\rfloor \times P, \quad c = \left(p \mod N_w\right) \times P,
  \]
  where N_w = W / P is the number of patches per row and p is the patch index.

- Patch flattening and embedding generation: Each (P x P) patch is flattened into a vector of size K with:
  
  \[
  K = P \times P.
  \]

  To compute the patch embedding, a linear projection is applied. For a given patch and an embedding dimension index, d, (with total embedding dimension, D = 128), the embedding is computed as:

  \[
  \text{embedding}[d] = \sum_{k=0}^{K-1} \text{patch}[k] \times W[k, d],
  \]

  where W is a weight matrix of size K x D

- Shared memory tiling: The patch data is loaded once into shared memory as tiles by a subset of threads. This minimizes redundant accesses to global memory by reusing the patch data for the dot-product computation. Each block processes one image patch while its threads compute the dot-product for each embedding dimension in parallel.  
   
2) Implementation:

Compiling the code:  

<pre>nvcc .\patch_embeddings.cu -o patch</pre>

Running the code after compiling: 

<pre> patch </pre>

<pre>CPU patch embedding time: 317.883 ms
GPU timings (in ms):
  Host-to-Device copy time: 0.525184
  Kernel execution time:   1.75309
  Device-to-Host copy time: 0.226048
  Total GPU time:          2.55174</pre>