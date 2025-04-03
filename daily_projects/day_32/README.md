Day 32: Implementation of positional encoding in CUDA

1) Summary of the daily tutorial:

The code performs a positional encodings (a key component in Transformer architectures) in CUDA. The goal is to compute a positional encoding matrix on the GPU using CUDA kernels, map the computation from a 2D matrix of shape (seq_len, d_model) (expanded to (1, seq_len, d_model)) where each row corresponds to a position and each column to a feature and use sine functions for even-indexed dimensions and cosine functions for odd-indexed ones, based on the formulation from the original Transformer paper (Vaswani et al. Attention Is All You Need, 2017).

For each position pos and dimension j:

For even indices (where \( j = 2i \)):  

```math
\text{PE}(pos, 2i) = \sin\!\Bigl(pos \cdot \exp\!\Bigl(-\frac{\log(10000)}{d_{\text{model}}} \cdot i\Bigr)\Bigr)  
```

For odd indices (where \( j = 2i+1 \)):  

```math
\text{PE}(pos, 2i+1) = \cos\!\Bigl(pos \cdot \exp\!\Bigl(-\frac{\log(10000)}{d_{\text{model}}} \cdot i\Bigr)\Bigr)  
```

- GPU Kernels: 
    - positional_encoding_kernel: Each thread calculates its global index to determine its corresponding position (pos) and feature dimension (j). It computes an angle based on the position and a scaled term (div_term) derived from the feature index and applies sinf if j is even or cosf if j is odd.
    Input: 
        pe: Output array (flattened tensor representing a matrix of shape seq_len * d_model),
        seq_len: Number of positions in the sequence,
        d_model: Dimension of each positional encoding.

2) Implementation:

Compiling the code:  

<pre>nvcc .\positionalencoding.cu -o posenc</pre>

Running the code after compiling: 

<pre> posenc </pre>

-  seq_len = 5 & d_model = 5

<pre>Positional Encoding (shape: 1 x 5 x 5):
0       1       0       1       0
0.841471        0.540302        0.157827        0.987467        0.0251162
0.909297        -0.416147       0.311697        0.950181        0.0502166
0.14112 -0.989992       0.457755        0.889079        0.0752853
-0.756802       -0.653644       0.592338        0.80569 0.100306</pre>