Day 41: Implementation of encoder block for vision transfomer in CUDA

1) Summary of the daily tutorial:

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