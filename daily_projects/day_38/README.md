Day 38: Implementation of encoder-decoder ingegration for sequence-to-sequence in CUDA

1) Summary of the daily tutorial:

The code demonstrates all the components of a Transformer‐based sequence-to-sequence (seq2seq) model. The integrated model consists of the following modules:

- Tokenizer and embedding lookup: The input sentences for the encoder and the decoder are tokenized and mapped to token IDs. For each token the corresponding embedding is looked up from a fixed embedding matrix. The embedding lookup is defined as:

  ```math
  \text{Embedding}(token\_id) = E[token\_id]
  ```

- Positional encodings: To incorporate the position information, sinusoidal positional encodings are computed and added to the token embeddings. The positional encoding function is defined as:

  ```math
  \text{PE}(pos, 2i) = \sin\left(\frac{pos}{10000^{\frac{2i}{d_{model}}}}\right)
  ```

  ```math
  \text{PE}(pos, 2i+1) = \cos\left(\frac{pos}{10000^{\frac{2i}{d_{model}}}}\right)
  ```

- Encoder block: The encoder block consists of:
  
    - Self-Attention: For an input:

      ```math
      Q = X \times W_{q}, \quad K = X \times W_{k}, \quad V = X \times W_{v}
      ```
      
      The multi-head attention (with heads and each head of dimension d_head = d_model / H, is computed as:

      ```math
      \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^{T}}{\sqrt{d_{\text{head}}}}\right)V
      ```

      The outputs of all heads are then concatenated and projected:

      ```math
      \text{SelfAttentionOutput} = \text{Attention}(Q, K, V) \times W_{o}
      ```
      
    - Residual connection and layer normalization:

      ```math
      X_{\text{attn}} = \text{LayerNorm}(X + \text{SelfAttentionOutput})
      ```
    
    - Feed forward network (FFN): FFN has two layers:
      
      ```math
      \text{FFN}_1(x) = \text{ReLU}(x \times W_1)
      ```

      ```math
      \text{FFN}(x) = \text{FFN}_1(x) \times W_2
      ```

      Followed by another residual connection and layer normalization:

      ```math
      \text{EncoderFinal} = \text{LayerNorm}(X_{\text{attn}} + \text{FFN}(X_{\text{attn}}))
      ```

- Decoder Block:

    The decoder block contains:
    
    - Self-Attention: Similar to the encoder, the decoder input is projected into Q, K, V and self-attention is computed:

      ```math
      \text{DecoderSelfAttention} = \text{softmax}\left(\frac{QK^T}{\sqrt{d_{\text{head}}}}\right)V \times W_{o}^{(\text{self})}
      ```

      Followed by a residual connection and layer normalization:
      
      ```math
      Y_{\text{norm}} = \text{LayerNorm}(Y + \text{DecoderSelfAttention})
      ```

    - Encoder-decoder attention: In this sub-layer, the queries come from the normalized output of self-attention, and the keys and values are obtained from the encoder’s final output. The projections are:
      
      ```math
      Q' = Y_{\text{norm}} \times W_{q}^{(\text{encdec})}
      ```

      ```math
      K = \text{EncFinal} \times W_{k}^{(\text{encdec})}, \quad V = \text{EncFinal} \times W_{v}^{(\text{encdec})}
      ```

      The scaled dot‑product attention is computed as:
      
      ```math
      \text{EncoderDecoderAttention} = \text{softmax}\left(\frac{Q'K^T}{\sqrt{d_{\text{head}}}}\right)V \times W_{o}^{(\text{encdec})}
      ```

      And then combined with a residual connection and layer normalization:
      
      ```math
      Z = \text{LayerNorm}(Y_{\text{norm}} + \text{EncoderDecoderAttention})
      ```

    - Feed forward network (FFN): FFN block in the decoder is similar to the encoder’s:
      
      ```math
      \text{FFN}_1(Z) = \text{ReLU}(Z \times W_1)
      ```

      ```math
      \text{FFN}(Z) = \text{FFN}_1(Z) \times W_2
      ```

      With the final output:
      
      ```math
      \text{DecoderFinal} = \text{LayerNorm}(Z + \text{FFN}(Z))
      ```

- CUDA Kernels: 
  - matmul_kernel: Tiled matrix multiplication for efficient linear projection and FFN computations
  - compute_scores_kernel:** Computes the scaled dot‑product attention scores
  - softmax_kernel: Applies softmax on the attention scores using parallel reduction for stability
  - weighted_sum_kernel: Computes the weighted sum over the values, V, using the softmax scores
  - add_kernel: Performs element‑wise addition for residual connections
  - relu_kernel: Applies the ReLU activation function
  - layer_norm_kernel: Computes layer normalization per token

2) Implementation:

Compiling the code:  

<pre>nvcc .\seq2seq.cu -o seq2seq</pre>

Running the code after compiling: 

<pre> seq2seq </pre>

<pre>CPU Total Inference Time: 3598.41 ms
GPU Inference Timings:
  Host-to-Device Copy Time: 2.14029 ms
  Encoder Kernel Time: 2.13709 ms
  Decoder Kernel Time: 1.664 ms
  Device-to-Host Copy Time: 0.138592 ms
  Total GPU Inference Time: 8.62422 ms</pre>