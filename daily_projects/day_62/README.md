Day 62: Implementation of mask decoder in CUDA

1) Summary of the daily tutorial

to be added

2) Implementation

Compiling the code:

<pre>nvcc .\mask_decoder.cu -o mask_decoder</pre>

Running the code after compiling:

<pre>mask_decoder</pre>

<pre>GPU timings (ms):
 Host-to-Device copy time:      :  0.41 ms
 Kernel execution time:         :  1.93 ms
 Device-to-Host copy time:      :  0.16 ms
 Total GPU time:                :  2.66 ms</pre>