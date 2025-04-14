Day 43: Implementation of image encoder in CUDA

1) Summary of the daily tutorial:

The code implements image encoder for vision langugage models which processes an input image using both patch embedding and a transformer encoder layer.

- Patch embedding: The input image is divided into non-overlapping patches, each of size 32×32 pixels. Each patch is flattened into a vector of 1024 elements and then linearly projected into a 128-dimensional embedding vector. This projection is achieved by multiplying the flattened patch vector by a learned weight matrix of shape [PATCH_VEC_SIZE x MODEL_DIM].

- Positional embeddings: Since dividing the image into patches discards the spatial ordering, sinusoidal positional encodings are added to each patch embedding to preserve location information. The positional encoding is computed for each patch index, p, and each embedding dimension, d, using a sinusoidal function. The computed encoding is added to the corresponding patch embedding as follows:

  ```math
  \text{embedding}[p, d] \mathrel{+}= \text{PE}(p, d)
  ```

  This encoding uses sine for even dimensions and cosine for odd dimensions based on the formula:

  ```math
  \text{PE}(p, d) = 
  \begin{cases} 
  \sin\left(\frac{p}{10000^{\frac{2\lfloor d/2\rfloor}{\text{MODEL\_DIM}}}}\right), & \text{if } d \text{ is even} \\
  \cos\left(\frac{p}{10000^{\frac{2\lfloor d/2\rfloor}{\text{MODEL\_DIM}}}}\right), & \text{if } d \text{ is odd}
  \end{cases}
  ``` 

- Encoder: Once the patch embeddings are enhanced with positional information, they are passed through a transformer encoder layer. This layer performs the following steps:
  
  - Linear projections to obtain Q, K, and V: For the input, X, (the matrix of patch embeddings), the queries, Q, keys, K, and values, V, are computed using weight matrices W_q, W_k, and W_v:

  ```math
  \text{Q} = \text{X} \times \text{W}_q,\quad
  \text{K} = \text{X} \times \text{W}_k,\quad
  \text{V} = \text{X} \times \text{W}_v
  ``` 

  - Attention computation: The attention scores are computed by multiplying, Q, with the transpose of K:

    ```math
    \text{scores} = \text{Q} \times \text{K}^\top
    ```
    
    These scores are then scaled by the factor \( \frac{1}{\sqrt{\text{MODEL\_DIM}}} \):

    ```math
    \text{scores} \mathrel{*}= \frac{1}{\sqrt{\text{MODEL\_DIM}}}
    ```

    Next, a row-wise softmax is applied to normalize the scores.

  - Output projection: The normalized scores (attention weights) are used to weight, V, vectors:

    ```math
    \text{attention} = \text{scores} \times \text{V}
    ``` 
        
    Finally, the attention output is projected with the weight matrix, W_o, to yield the final encoder output:

    ```math
    \text{output} = \text{attention} \times \text{W}_o
    ```

- CUDA Kernels: kernels utilize tiling and shared memory
  - patch_embedding_kernel_tiled: Loads each patch into shared memory and performs the linear projection.
  - add_positional_embeddings_tiled: Processes patch embeddings in tiles to compute and add the sinusoidal positional encodings.
  - tiledMatMulKernel: Supports the transformer encoder’s linear projections, attention score computation, and final output projection.
  - transposeKernel, scaleKernel, softmaxKernel: Handle specific operations in the encoder, such as transposing the key matrix, scaling the scores, and performing the softmax normalization.

2) Implementation:

Compiling the code:  

<pre>nvcc .\image_encoder.cu -o image_encoder</pre>

Running the code after compiling: 

<pre> image_encoder </pre>

<pre>Total CPU inference time (ms): 1011.11
GPU Timings (ms):
  Host-to-Device copy time: 0.60416
  Kernel execution time:    4.99517
  Device-to-Host copy time: 0.221376
  Total GPU inference time: 5.89053</pre>