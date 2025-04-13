Day 41: Implementation of encoder block for vision transfomer in CUDA

1) Summary of the daily tutorial:

The code implements the encoder block for vision transformers using CUDA. The encoder block takes an input matrix (representing patch embeddings) and performs a series of matrix operations to compute the final output.

- Computing queries, keys, and values: The input, X, is multiplied with three different weight matrices, W_q, W_k, and W_v, to generate the query, Q, key, K, and value, V, matrices respectively

  ```math
  Q = X \times W_q
  ```

  ```math
  K = X \times W_k
  ```

  ```math
  V = X \times W_v
  ```

- Transposing the key matrix: To prepare for the attention score calculation, the key matrix, K, is transposed to K^T:

  ```math
  K^T = \text{transpose}(K)
  ```

- Computing attention scores: The attention scores are obtained by performing a matrix multiplication between the query matrix, Q, and the transposed key matrix K^T:

  ```math
  \text{scores} = Q \times K^T
  ```

- Scaling the attention scores: The computed scores are scaled by the inverse square root of the model dimension to help stabilize gradients:

  ```math
  \text{scores} \mathrel{*}= \frac{1}{\sqrt{\text{MODEL\_DIM}}}
  ```

- Applying the softmax function: A row-wise softmax is applied to the scaled scores to convert them into a probability distribution:

  ```math
  \text{attention\_weights} = \text{softmax}(\text{scores})
  ```

- Computing the attention output: The final attention output is computed by multiplying the softmax-normalized scores with the value matrix, V:

  ```math
  \text{attention} = \text{attention\_weights} \times V
  ```

- Producing the final output: The attention output is then multiplied by an output weight matrix, W_o, to compute the final output of the encoder block:

  ```math
  \text{output} = \text{attention} \times W_o
  ```

CUDA kernels:
- tiledMatMulKernel: For efficient computation of the matrix multiplications.
- transposeKernel: For rearranging matrix K to K^T.
- scaleKernel: For element-wise scaling of the attention scores.
- softmaxKernel: For normalizing the attention scores row-wise.

2) Implementation:

Compiling the code:  

<pre>nvcc .\encoder.cu -o encoder</pre>

Running the code after compiling: 

<pre> encoder </pre>

<pre>GPU Timings (ms):
  Host -> Device copy: 0.768672
  Kernel execution:    1.64902
  Device -> Host copy: 0.163808
  Total GPU time:      2.5815
Total GPU Inference Time (ms): 4.68422
Total CPU Inference Time (ms): 679.742</pre>