Day 60: Implementation of prompt encoder for sparse and dense prompt embeddings in CUDA

1) Summary of the daily tutorial

The code implements how to build a prompt encoder that produces both sparse and dense prompt embeddings. Sparse embeddings are generated from a set of user‐defined points and a bounding box, while dense embeddings are derived from a full-resolution mask by downsampling and projection. By structuring the work into three specialized CUDA kernels, we achieve high throughput and minimize data movement.

- Operations performed are:

  ```math
  \text{token\_embed}[t, d]
    = x_t \cdot W_{\text{pos}}[0,d]
    + y_t \cdot W_{\text{pos}}[1,d]
    + T_{\text{type}_t,d}
  ```

  ```math
  m_{\text{down}}[o_y,o_x]
    = \frac{1}{16\times16} \sum_{dy=0}^{15}\sum_{dx=0}^{15}
      \text{mask}\bigl[o_y\cdot16+dy,\;o_x\cdot16+dx\bigr]
  ```

  ```math
  \text{mask\_proj}[c, d]
    = m_{\text{down}}[c]\;W_m[d] + B_m[d]
  ```

- CUDA kernels:
  - point_box_embed: Computes a 256-dimensional embedding for each token (points + box corners) by combining its (x,y) coordinates with a learnable type embedding lookup table.
  - mask_downsample: Reduces a 1024×1024 binary mask to 64×64 by averaging each non-overlapping 16×16 block, producing a coarse representation of the mask.
  - mask_project: Projects each downsampled scalar value into a 256-dimensional vector via a per-cell linear layer (element-wise multiplication by weights and addition of biases).

2) Implementation

Compiling the code:

<pre>nvcc .\prompt_encoder.cu -o prompt_encoder</pre>

Running the code after compiling:

<pre>prompt_encoder</pre>

<pre>GPU timings (ms):
 Host-to-Device copy time:      :  0.43 ms
 Kernel execution time:         :  0.96 ms
 Device-to-Host copy time:      :  0.05 ms
 Total GPU time:                :  1.47 ms</pre>