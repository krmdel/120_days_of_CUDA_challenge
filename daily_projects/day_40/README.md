Day 40: Implementation of positional embeddings for image patches in CUDA

1) Summary of the daily tutorial:

The code performs computing positional embeddings for image patches. Because the patch extraction process discards information about the spatial location of patches, positional encodings are crucial to enable the model to preserve positional information.

- Positional encoding computation: For each patch embedding with index, p, and each embedding dimension index, d, (with total embedding dimension of D = 128), sinusoidal positional encoding values are computed as:
  
  ```math
  \text{PE}(p, d) = \begin{cases} 
  \sin\left(\frac{p}{10000^{\frac{2\lfloor d/2 \rfloor}{D}}}\right), & \text{if } d \text{ is even} \\
  \cos\left(\frac{p}{10000^{\frac{2\lfloor d/2 \rfloor}{D}}}\right), & \text{if } d \text{ is odd}
  \end{cases}
  ```

- Updating patch embeddings: The computed positional encoding is then added element‑wise to the patch embeddings:
  
  ```math
  \text{embedding}[p, d] \mathrel{+}= \text{PE}(p, d)
  ```

- Shared memory tiling: The patch embedding vector is processed in tiles using shared memory. Each CUDA block corresponds to one patch, and the embedding vector is processed in sub‑chunks (tiles) that are first loaded into shared memory. This tiling strategy minimizes redundant global memory accesses when computing the positional encodings and updating the embeddings.

 2) Implementation:

Compiling the code:  

<pre>nvcc .\patch_embeddings.cu -o patch</pre>

Running the code after compiling: 

<pre> patch </pre>

<pre>CPU positional embeddings addition time: 102.246 ms
GPU timings (in ms):
  Host-to-Device copy time: 0.765312
  Kernel execution time:   1.11142
  Device-to-Host copy time: 0.763552
  Total GPU time:          2.68141</pre>