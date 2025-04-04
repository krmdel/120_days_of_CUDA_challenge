Day 33: Implementation of tokenizer for transformers in CUDA

1) Summary of the daily tutorial:

The code performs a simple tokenizer and embedding lookup on CUDA. 

- Tokenization on the CPU: The code takes a list of sentences, splits them into tokens based on whitespace, and maps each token to a token ID using a predefined vocabulary. Missing tokens are padded with -1.

- Embedding Matrix: A small toy embedding matrix is defined where each row represents a learned embedding for a token.

- CUDA Kernel for Embedding Lookup: embedding_lookup_kernel takes the token IDs and, for each token, retrieves its corresponding embedding vector from the embedding matrix. If a token ID is invalid (i.e., -1 or out of range), the kernel returns a zero vector for that token.

2) Implementation:

Compiling the code:  

<pre>nvcc .\tokenizer.cu -o tokenizer</pre>

Running the code after compiling: 

<pre> tokenizer </pre>

<pre>CPU -> GPU copy time: 0.022528 ms
Kernel execution time: 0.8704 ms
GPU -> CPU copy time: 0.024864 ms
Total elapsed time: 0.958848 ms

Output Embeddings:
Sentence 0:
  Token 0: 0.1 0.2 0.3 0.4
  Token 1: 0.5 0.6 0.7 0.8
  Token 2: 0 0 0 0
  Token 3: 0 0 0 0
  Token 4: 0 0 0 0
Sentence 1:
  Token 0: 0.9 1 1.1 1.2
  Token 1: 1.3 1.4 1.5 1.6
  Token 2: 0 0 0 0
  Token 3: 0 0 0 0
  Token 4: 0 0 0 0</pre>