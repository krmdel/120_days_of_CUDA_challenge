Day 56: Implementation of flash attention in CUDA

1) Summary of the daily tutorial


2) Implementation

Compiling the code:

<pre>nvcc .\flash_attention.cu -o flash_attention</pre>

Running the code after compiling:

<pre>flash_attention</pre>

<pre>Baseline (L=512, d=64)
  H2D   : 0.188544 ms
  Kernels: 1.40992 ms
  D2H   : 0.094144 ms
  Total : 1.69261 ms

FlashAttention-CTA (L=512, d=64)
  H2D   : 0.121216 ms
  Kernel: 0.048672 ms
  D2H   : 0.047072 ms
  Total : 0.21696 ms</pre>