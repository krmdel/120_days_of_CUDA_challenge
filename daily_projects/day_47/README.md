Day 47: Implementation of vision language model for image captioning in CUDA

1) Summary of the daily tutorial

The code performs an autoregressive decoder to generate text captions from visual features. After extracting image patch embeddings and computing cross‑modal context in previous steps, the decoder:

- Initializes the generated sequence with a start‐token embedding. 
- Then iteratively performs:
  - Self‐Attention with causal masking: over the tokens produced so far, ensuring each position only attends to earlier positions.
  - Cross‐Modal Attention: between the latest token’s query and the fixed visual features.
  - Feed‐Forward Network (FFN): to project and refine the token representation.
  - Token Projection & Argmax: to predict the next token ID.
  - Token Embedding: to append the new token into the sequence buffer.

    ```math
    \text{Attention}(Q, K, V) \;=\; \text{Softmax}\!\Bigl(\tfrac{Q\,K^\top}{\sqrt{D_\text{head}}}\Bigr)\,V
    ```

    - Self‐attention with causal mask enforces  
    ```math
    \text{scores}[i,j] \;=\;
        \begin{cases}
        \tfrac{Q_i \cdot K_j}{\sqrt{D_\text{head}}}, & j \le i,\\
        -\infty, & j > i.
        \end{cases}
    ```

    - Feed‐forward network per token, x:  
    ```math
    \text{FFN}(x) = W_2\bigl(\max(0,\,x\,W_1)\bigr) + x.
    ```

- CUDA kernels:
  - patch_embedding_kernel: Loads each image patch into shared memory and computes its projection into the model dimension.
  - add_pos_emb_kernel: Adds the sinusoidal positional encoding in place on the patch embeddings.
  - tiledMatMulKernel: A shared‑memory tiling implementation of matrix multiplication, used throughout for projections, attention score computation, and output transforms.
  - transposeKernel: Reorders a matrix in global memory to enable efficient dot‑product attention.
  - scaleKernel: Multiplies every element of the scores matrix by 1/sqrt(D) or 1/sqrt(D_head).
  - softmaxKernel: Computes a row‑wise softmax on the scores matrix using a two‑phase shared‑memory reduction.
  - causal_mask_kernel: Applies the autoregressive mask by writing (-10^9) to all future positions (j>i).
  - addInPlace: Element‑wise adds one buffer into another (used for residual connections).
  - reluKernel: Applies the ReLU activation in the FFN’s first linear layer.
  - extractQuery: Gathers the last token’s vector into a small buffer for the cross‑attention query.
  - argmax_kernel: Performs a parallel reduction to find the index of the maximum logit (next token ID).
  - embed_token_kernel: Looks up the token embedding for the newly predicted ID and writes it into the sequence buffer.
  - write_id_kernel: Stores the predicted token ID into the output ID array at the correct position.

2) Implementation

Compiling the code:

<pre>nvcc .\image_captioning.cu -o capt</pre>

Running the code after compiling:

<pre>capt</pre>

<pre>Total CPU inference time (ms): 526.845 ms
GPU inference timings (ms):
  Host-to-Device copy time: 3.61677
  Kernel execution time:    7.26323
  Device-to-Host copy time: 0.052672
  Total GPU time:           12.7908</pre>